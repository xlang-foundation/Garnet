"""Untimed B4/T4 state gate; ordinary outputs are proposal ORACLES for testing only."""
import sys,json,os
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet,tensor
from xModel.mistral.text_7b.metal_decoder import MistralMetalDecoder
with open(sys.argv[3]) as f:inputs=json.load(f)
output=sys.argv[4];mode=sys.argv[5];assert mode in ('ordinary','verify','reverse') and not os.path.exists(output)
assert 'native_dense_verify_b4t4_optional_v1' in garnet.backend_info()
with open(inputs['target_model']+'/config.json') as f:config=json.load(f)
reference=None
if mode!='ordinary':
 with open(inputs['reference_path']) as f:reference=json.load(f)
 assert reference['completed'] and reference['mode']=='ordinary'
 oracle={j['label']:j['generated'] for j in reference['jobs']}
decoder=None;policy=None;jobs={};all_jobs={};records=[];ticks=[];rejects=[];prepared=None;cleanup=None;calls=0
def val(a,shape):return tensor.tensor(a,shape=shape,dtype=tensor.float32)
def save(done):
 with open(output,'w') as f:json.dump(dict(completed=done,mode=mode,scope=__doc__,inputs=inputs,backend=garnet.backend_info(),library=garnet.native_library_path(),prepared=prepared,jobs=list(all_jobs.values()),ticks=ticks,records=records,rejections=rejects,cleanup=cleanup),f,indent=2)
def witness(label,owners):
 selected=[]
 for j in owners:selected.append(dict(label=j['label'],generated=len(j['generated']),row=j['row'],epoch=j['epoch'],status=j['status'],states=[]))
 for layer in range(32):
  pair=decoder.session.snapshot(layer)
  for j in selected:j['states'].append([decoder.io.verification_f32_sha256(decoder.io.rows(part,j['row'],1)) for part in pair])
 hidden=decoder.session.diagnostic_final_hidden()
 for j in selected:j['hidden']=decoder.io.verification_f32_sha256(decoder.io.rows(hidden,j['row'],1))
 records.append(dict(label=label,rows=selected,status=json.loads(policy.status_json())));save(False)
def reject(label,args):
 before=policy.status_json();failed=False
 try:policy.verify_decode_json(*args)
 except Exception:failed=True
 assert failed and policy.status_json()==before;rejects.append(label)
try:
 decoder=MistralMetalDecoder(inputs['target_model']+'/model.safetensors',config,4,4096);decoder.prepare_prefill(128)
 if mode=='reverse':decoder.prepare_verification(4);decoder.prepare_compact_prefill(512)
 else:decoder.prepare_compact_prefill(512);decoder.prepare_verification(4)
 assert decoder.verification_tokens==4 and decoder.session.compact_prefill_family_count==17
 decoder.prefill_compile_ms=0;base=[layer['graph'] for layer in decoder.layers];decoder._resident_prefill_graphs(128)
 assert decoder.session.configure_token_tracking(['d7182d886f93a9efa73d6d556ee33ffe839a4e09e15965eeef46f295ab5b503d',decoder.io.checkpoint_format,inputs['target_tokenizer_sha256'],'text_zero_based_3axis_v1','scheduled-B4-C4096-prefill-up512-verify4-fp32-v1','text'])
 policy=garnet.continuous_scheduler();assert policy.initialize(decoder.session,16);assert policy.set_prefill_chunk(512)
 prepared=dict(memory=json.loads(garnet.memory_stats_json()),target_graphs=len(decoder.prefill_graphs),compact_families=decoder.session.compact_prefill_family_count,verification_tokens=decoder.verification_tokens)
 assert prepared['memory']['metal_allocated_bytes']<=prepared['memory']['metal_recommended_working_set_bytes']-536870912
 for item in inputs['jobs']:
  identity=policy.submit_tokens(item['prompt_ids'],item['limit'],0);assert identity>0
  job=dict(item,id=identity,generated=[],prefilled=0,status='queued');jobs[identity]=job;all_jobs[identity]=job
 while jobs:
  tick=json.loads(policy.next_json());assert tick['batches'];ticks.append(dict(plan=tick,executions=[]));record=ticks[-1]
  for plan in tick['batches']:
   for q in plan['requests']:jobs[q['id']].update(row=q['row'],epoch=q['epoch'])
   if plan['stage']=='prefill':
    width=plan['prefill_chunk'];byrow={q['row']:q for q in plan['requests']};ids=[];slots=[];flags=[];final=[0]*4
    for row in range(4):
     q=byrow.get(row)
     if q is not None and q['prefill_offset']+q['prefill_tokens']==len(jobs[q['id']]['prompt_ids']):final[row]=q['epoch']
     for t in range(width):
      on=q is not None and t<q['prefill_tokens'];position=q['prefill_offset']+t if on else 0
      ids.append(jobs[q['id']]['prompt_ids'][position] if on else 0);slots.append(float(position));flags.append(float(on))
    hidden=decoder.io.stack_tokens([decoder.io.embedding([ids[row*width+t] for row in range(4)]) for t in range(width)])
    decoder.session.run_tokens(decoder.prefill_profile_graphs(width),hidden,val([s for s in slots for axis in range(3)],[4,width,3]),val(slots,[4,width]),val(flags,[4,width]),-1,plan['epochs'],ids)
    predicted=decoder.session.predict_rows(final) if any(final) else None;finished=[]
    for q in plan['requests']:
     job=jobs[q['id']];job['prefilled']=q['prefill_offset']+q['prefill_tokens']
     if final[q['row']]:
      token=predicted[q['row']];job['generated'].append(token);assert policy.complete_prefill(q['id'],token,False)==2;job['status']='decode';finished.append(job)
     else:assert policy.complete_prefill_chunk(q['id'],q['prefill_tokens'])
    record['executions'].append(dict(stage='prefill',width=width,compact_rows=decoder.session.compact_prefill_last_rows))
    if finished:witness('prefill',finished)
   else:
    assert plan['stage']=='decode';ids=[0]*4;slots=[0.]*4
    for q in plan['requests']:ids[q['row']]=q['token_id'];slots[q['row']]=float(q['slot']);assert ids[q['row']]==jobs[q['id']]['generated'][-1]
    if mode=='ordinary':
     predicted=decoder.session.run_tokens(base,decoder.io.embedding(ids),val([s for s in slots for axis in range(3)],[4,3]),val(slots,[4]),val([float(x) for x in plan['active_rows']],[4]),0,plan['epochs'],ids);completed=[]
     for q in plan['requests']:
      job=jobs[q['id']];token=predicted[q['row']];job['generated'].append(token);finished=len(job['generated'])==job['limit'];assert policy.complete_decode(q['id'],token,finished)==(3 if finished else 2);completed.append(dict(id=q['id'],row=q['row'],epoch=q['epoch'],emitted=[token],finished=finished))
     native=None;submitted=None
    else:
     byrow={q['row']:q for q in plan['requests']};flat=[];positions=[];active=[];submitted=[]
     for row in range(4):
      q=byrow.get(row);job=jobs[q['id']] if q is not None else None;remaining=job['limit']-len(job['generated']) if job else 0
      depth=min(remaining,4 if calls==0 else 1+(row+calls)%4);mismatch=(row if calls==0 else (row+calls)%4)
      for t in range(4):
       on=t<depth;token=0
       if on:
        token=ids[row] if t==0 else oracle[job['label']][len(job['generated'])+t-1]
        if t>0 and t==mismatch:token=(token+1)%32768
       flat.append(token);positions.append(slots[row]+float(t) if on else 0.);active.append(float(on))
      if q is not None:submitted.append(dict(id=q['id'],row=row,known=ids[row],generated_before=len(job['generated']),tokens=flat[row*4:row*4+depth],depth=depth))
     args=[decoder.io.stack_tokens([decoder.io.embedding([flat[row*4+t] for row in range(4)]) for t in range(4)]),val([s for s in positions for axis in range(3)],[4,4,3]),val(positions,[4,4]),val(active,[4,4]),flat,[]]
     if calls==0:
      witness('before_invalid',list(jobs.values()));bad=list(args);tokens=list(flat);tokens[0]=(tokens[0]+1)%32768;bad[4]=tokens;reject('wrong_known',bad)
      bad=list(args);flags=list(active);flags[1]=0.;bad[3]=val(flags,[4,4]);reject('noncontiguous_prefix',bad);witness('after_invalid',list(jobs.values()))
     native=json.loads(policy.verify_decode_json(*args));completed=native['requests'];assert len(completed)==len(plan['requests']);calls+=1
     assert native['verification']['target_affine_misses']==0 and native['verification']['target_indexed_decode_dispatches']==225
     for q in completed:
      job=jobs[q['id']];assert q['row']==job['row'] and q['epoch']==job['epoch'] and q['remaining_before']==job['limit']-len(job['generated']);assert q['consumed']==len(q['emitted']) and 1<=q['consumed']<=q['submitted']<=4
      job['generated'].extend(q['emitted']);assert job['generated']==oracle[job['label']][:len(job['generated'])];assert q['finished']==(len(job['generated'])==job['limit'])
     if calls==1:
      witness('before_stale',list(jobs.values()));reject('stale_tick',args);witness('after_stale',list(jobs.values()))
    for q in completed:
     if q['finished']:all_jobs[q['id']]['status']='finished';assert not decoder.session.release_request_row(q['row'],q['epoch']);del jobs[q['id']]
    record['executions'].append(dict(stage='decode',native=native,submitted=submitted,completed=completed));witness('decode',[all_jobs[q['id']] for q in completed])
  assert len(ticks)<100
 assert json.loads(policy.status_json())['active']==json.loads(policy.status_json())['tracked']==0
 assert decoder.prefill_compile_ms==0 and len(decoder.prefill_graphs)==224
finally:
 if policy is not None:policy.close()
 if decoder is not None:decoder.close()
 garnet.shutdown();cleanup=json.loads(garnet.memory_stats_json())
 for k in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):assert cleanup[k]==0
 assert cleanup['metal_allocated_bytes']<=4194304
 save(len(all_jobs)==4 and all(j['status']=='finished' for j in all_jobs.values()))
