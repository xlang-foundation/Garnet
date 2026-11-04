"""Execute explicit MTP graphs against saved independent FP32 references."""
import json,math,struct,sys
sys.path.insert(0,sys.argv[2]);sys.path.insert(0,sys.argv[3])
import garnet,tensor
from xModel.qwen38.vl_27b.metal_mtp import MetalMTPProgram,MetalMTPChunk
with open(sys.argv[1]) as stream:f=json.load(stream)
io=garnet.tensor_io();program=None;source=None;previous=None;profile=None;results=[];profiles=[];checks=0
def close_programs():
 global program,source
 if program is not None:program.close();program=None
 if source is not None:source.close();source=None
def compare(actual,expected,label):
 global checks
 a=actual.tolist();b=expected.tolist();assert len(a)==len(b),label
 worst=0.
 for x,y in zip(a,b):
  assert math.isfinite(x) and math.isfinite(y),label+' nonfinite'
  ratio=abs(x-y)/(f['atol']+f['rtol']*abs(y));worst=max(worst,ratio)
 assert worst<=1,(label,'numeric criterion',worst)
 checks+=1;return worst
try:
 for case in f['cases']:
  b=case['batch'];t=case['chunk'];cap=case['capacity'];h=f['config']['hidden_size'];d=f['config']['head_dim'];kv=f['config']['num_key_value_heads'];cache=kv*cap*d
  key=(b,t,cap)
  if profile!=key:
   close_programs();source=MetalMTPProgram(f['checkpoint'],f['config'],b,cap,1)
   program=MetalMTPProgram(f['checkpoint'],f['config'],b,cap,t,source=source)
   inventory=json.loads(garnet.memory_inventory_json([{'tag':'source','object':source.graph},{'tag':'child','object':program.graph}],None))
   owners={}
   for tag in ('source','child'):
    names={}
    for ref in inventory['references']:
     if ref['tag']==tag and ref['role']=='weight_record':
      if ref['name'] not in names:names[ref['name']]=[]
      names[ref['name']].append(ref['buffer_id'])
    owners[tag]=names
   assert owners['source']==owners['child'] and len(owners['source'])==15,'MTP shared weight identity'
   assert len([name for name in owners['source'] if name.endswith('|BF16')])==8,'MTP BF16 projection count'
   snapshot=inventory['device_snapshot'];assert snapshot['metal_allocated_bytes']<=snapshot['metal_recommended_working_set_bytes']-256*1024*1024
   inventory_path=f['output']+'.B'+str(b)+'T'+str(t)+'C'+str(cap)+'.inventory.json'
   with open(inventory_path,'w') as stream:json.dump(inventory,stream,indent=2)
   profiles.append(dict(batch=b,chunk=t,source_chunk=1,capacity=cap,inventory=inventory_path,prepared_snapshot=snapshot,shared_weights=15,shared_bf16_projections=8));checks+=3
   if not f['trained']:
    symbols=[tensor.input('mtp_negative_'+str(i),shape=s,dtype=tensor.float32) for i,s in enumerate(program.shapes)]
    captured=MetalMTPChunk(*symbols,f['config'],b,t,cap)
    for method,bindings in [('build_mtp',symbols[:6]),('build_bf16',symbols)]:
     bad_graph=garnet.compiled_graph();failed=False
     try:
      if method=='build_mtp':bad_graph.build_mtp(captured,bindings,f['checkpoint'])
      else:bad_graph.build_bf16(captured,bindings,f['checkpoint'])
     except Exception:failed=True
     assert failed and not bad_graph.ready,'MTP explicit build contract';bad_graph.close();checks+=1
    for index,axis,value in [(0,0,5),(0,1,5),(2,2,4097)]:
     shapes=[list(s) for s in program.shapes];shapes[index][axis]=value
     invalid=[tensor.input('mtp_bounds_'+str(i),shape=s,dtype=tensor.float32) for i,s in enumerate(shapes)]
     before=json.loads(garnet.memory_stats_json());bad_graph=garnet.compiled_graph();message=''
     try:bad_graph.build_mtp(captured,invalid,f['checkpoint'])
     except Exception as error:message=str(error)
     assert 'MTP profile requires' in message and not bad_graph.ready,'MTP native profile bounds'
     bad_graph.close();after=json.loads(garnet.memory_stats_json())
     assert before['metal_allocated_bytes']==after['metal_allocated_bytes'],'Invalid MTP profile allocated GPU buffers';checks+=2
   # The shared child must retain its weights after closing the source graph.
   source.close();source=None;profile=key;previous=None
  inputs=[io.read_f32(path,shape) for path,shape in zip(case['inputs'],program.shapes)]
  if case['carry']:
   assert previous is not None
   inputs[2]=io.slice(previous,t*h,[b,kv,cap,d]);inputs[3]=io.slice(previous,t*h+cache,[b,kv,cap,d])
  expected=io.read_f32(case['expected'],[b,t*h+2*cache]);actual=program.run(inputs);worst=compare(io.slice(actual,0,[b,t*h]),io.slice(expected,0,[b,t*h]),case['label']+' hidden')
  changed=[];untouched=0
  for row in range(b):
   # slice() operates on all physical rows. Compare numeric cache spans using
   # their union; compare untouched complements exactly, including inactive rows.
   changed.extend(int(s) for s in case['slots'][row][:case['lengths'][row]])
  changed=sorted(set(changed))
  for part in range(2):
   for head in range(kv):
    base=t*h+part*cache+head*cap*d;start=0
    for slot in changed+[cap]:
     if slot>start:
      a=io.slice(actual,base+start*d,[b,(slot-start)*d]);e=io.slice(previous if case['carry'] else expected,base+start*d,[b,(slot-start)*d])
      assert io.verification_f32_sha256(a)==io.verification_f32_sha256(e),(case['label'],'untouched cache',part,head,start,slot)
      untouched+=b*(slot-start)*d;checks+=1
     if slot<cap:
      selected=io.slice(actual,base+slot*d,[b,d]);worst=max(worst,compare(selected,io.slice(expected,base+slot*d,[b,d]),case['label']+' updated cache'))
      a=selected.tolist();e=io.slice(previous if case['carry'] else expected,base+slot*d,[b,d]).tolist()
      for row in range(b):
       if slot not in [int(s) for s in case['slots'][row][:case['lengths'][row]]]:
        assert struct.pack('<'+str(d)+'f',*a[row*d:(row+1)*d])==struct.pack('<'+str(d)+'f',*e[row*d:(row+1)*d]),'Inactive cache row changed'
        untouched+=d;checks+=1
     start=slot+1
  # Failed metadata must be rejected before upload; the next valid call is exact.
  if not case['carry']:
   for index,replacement in [(6,[.5]*(b*t)),(5,[-1.]*(b*t)),(4,[float('nan')]*(b*t*3))]:
    bad=list(inputs);bad[index]=tensor.tensor(replacement,shape=program.shapes[index],dtype=tensor.float32);failed=False
    try:program.run(bad)
    except Exception:failed=True
    assert failed,(case['label'],'missing control rejection',index);checks+=1
   if t>1:
    bad=list(inputs);flags=[0.]*(b*t);flags[1]=1.;bad[6]=tensor.tensor(flags,shape=program.shapes[6],dtype=tensor.float32);failed=False
    try:program.run(bad)
    except Exception:failed=True
    assert failed,(case['label'],'missing non-prefix rejection');checks+=1
    bad=list(inputs);flags=[0.]*(b*t);flags[0]=1.;flags[1]=1.;slots=[s for row in case['slots'] for s in row];slots[1]=slots[0]+2.
    bad[5]=tensor.tensor(slots,shape=program.shapes[5],dtype=tensor.float32);bad[6]=tensor.tensor(flags,shape=program.shapes[6],dtype=tensor.float32);failed=False
    try:program.run(bad)
    except Exception:failed=True
    assert failed,(case['label'],'missing noncontiguous-slot rejection');checks+=1
   repeated=program.run(inputs);assert io.verification_f32_sha256(repeated)==io.verification_f32_sha256(actual),(case['label'],'failed metadata altered graph');checks+=1
  previous=actual
  record=dict(label=case['label'],worst_element_ratio=worst,untouched_cache_words=untouched,output_sha256=io.verification_f32_sha256(actual))
  results.append(record);print('metal_mtp_case PASS',json.dumps(record),flush=True)
 result=dict(passed=True,cases=results,profiles=profiles,checks=checks,trained=f['trained'],sensitivity=f['sensitivity'],scope='Auxiliary graph FP32 oracle only; no target generation, draft acceptance, coding or speed admission',backend=garnet.backend_info())
 with open(f['output'],'w') as stream:json.dump(result,stream,indent=2)
finally:
 close_programs();garnet.shutdown();memory=json.loads(garnet.memory_stats_json())
 for key in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):assert memory[key]==0,(key,memory[key])
 print('metal_mtp_closed',json.dumps(memory),flush=True)
