"""Real MTP drafts, exact target tokens/state and standalone teacher-cache oracle."""
import json,sys
sys.path.insert(0,sys.argv[2]);sys.path.insert(0,sys.argv[3])
import garnet,tensor
from xModel.qwen38.vl_27b.model import create_decoder
from xModel.qwen38.vl_27b.metal_mtp import MetalMTPProgram
from xModel.qwen35.vl_0_8b.decoder_layer import T,norm
from xModel.qwen35.vl_0_8b.prefill_layer import PrefillLayer
with open(sys.argv[1]) as stream:f=json.load(stream)
with open(f['config']) as stream:config=json.load(stream)
with open(sys.argv[5]) as stream:prompt=json.load(stream)
c=config['text_config'];w=c['hidden_size'];cap=f['capacity'];chunk=4;kv=c['num_key_value_heads'];dim=c['head_dim'];cache=kv*cap*dim
decoder=None;reference=garnet.decoder_session();graphs=[];mtp=None;oracle=None;checks=0;records=[];mtp_records=[]
def val(data,shape):return tensor.tensor(data,shape=shape,dtype=tensor.float32)
def syms(shapes,name):return [tensor.input(name+str(i),shape=s,dtype=tensor.float32) for i,s in enumerate(shapes)]
def equal(a,b,label):
 global checks
 assert a==b,(label,a,b);checks+=1
try:
 decoder=create_decoder(f['checkpoint'],config,cap,linear_chunk=True);target=decoder.session;io=decoder.io;base=[l['graph'] for l in decoder.layers]
 reference.build(base,decoder.head);verify=[]
 for i,layer in enumerate(decoder.layers):
  inputs=syms([[1,4,w]]+layer['shapes'][1:3]+[[1,4,3],[1,4],[1,4]],'mtp_real_verify_'+str(i)+'_')
  @T.fusion(name='mtp_real_verify_'+str(i),boundary='required')
  def body(h,a,b,p,s,active):return PrefillLayer(h,a,b,p,s,active,c,i,1,4,cap,layer['kind'],parallel_attention=True,parallel_linear=True,attention_capacity_limit=4096)
  g=garnet.compiled_graph();g.build_verification_shared(body(*inputs),inputs,f['checkpoint'],base[i]);graphs.append(g);verify.append(g)
 for rows,kind in [(4,'verify'),(1,'projection'),(1,'norm')]:
  inputs=syms([[rows,w]],'mtp_real_'+kind+'_')
  @T.fusion(name='mtp_real_'+kind,boundary='required')
  def head(h):
   if kind=='norm':return norm(h,'model.language_model.norm',c)
   if kind=='projection':return h*T.unary_op('qwen_text_head',weight_name='lm_head.weight',bf16_weight=1)
   return norm(h,'model.language_model.norm',c)*T.unary_op('qwen_text_head',weight_name='lm_head.weight',bf16_weight=1)
  g=garnet.compiled_graph();g.build_shared(head(*inputs),inputs,f['checkpoint'],decoder.head);graphs.append(g)
 verification_head,projection_head,normalizer=graphs[-3:]
 target.prepare_verification(verify,verification_head);mtp=MetalMTPProgram(sys.argv[4],c,1,cap,4);oracle=MetalMTPProgram(sys.argv[4],c,1,cap,4,source=mtp)
 target.prepare_mtp(mtp.graph,projection_head,[1,4]);prepared=json.loads(garnet.memory_stats_json());print('trained_mtp_bridge_prepared',json.dumps(prepared),flush=True)
 inventory_path=f['output']+'.inventory.json'
 with open(inventory_path,'w') as stream:stream.write(garnet.memory_inventory_json([{'tag':'target_head','object':decoder.head},{'tag':'mtp_head','object':projection_head},{'tag':'mtp','object':mtp.graph}],target))
 key=io.zeros([1,kv,cap,dim]);value=io.zeros([1,kv,cap,dim]);last=io.zeros([1,w]);predicted=[0]
 def stack(ids):return io.stack_tokens([io.embedding([x]) for x in ids])
 def controls(start,length):
  slots=[float(start+j) for j in range(4)];return val([s+1 for s in slots for axis in range(3)],[1,4,3]),val(slots,[1,4]),val([float(j<length) for j in range(4)],[1,4])
 def step(owner,token,slot):return owner.run(base,io.embedding([token]),val([float(slot)]*3,[1,3]),val([float(slot)],[1]),val([1.],[1]),0)[0]
 def oracle_step(hidden,embedding,ctl,length):
  global key,value,last,predicted
  output=oracle.run([hidden,embedding,key,value]+list(ctl));key=io.slice(output,4*w,[1,kv,cap,dim]);value=io.slice(output,4*w+cache,[1,kv,cap,dim])
  if length:last=io.slice(output,(length-1)*w,[1,w])
  predicted=io.argmax(projection_head.run([last]));return predicted
 def compare(stage):
  snap=target.mtp_snapshot();a=[io.verification_f32_sha256(x) for x in snap[:3]];b=[io.verification_f32_sha256(x) for x in [key,value,last]];equal(a,b,stage+' MTP state');mtp_records.append(dict(stage=stage,actual=a,reference=b))
 first_draft=None
 for slot,token in enumerate(prompt['ids']):
  generated=step(target,token,slot);equal(generated,step(reference,token,slot),'prefill prediction')
  next_token=prompt['ids'][slot+1] if slot+1<len(prompt['ids']) else generated
  normalized=normalizer.run([reference.hidden()]);packed=io.stack_tokens([normalized]+[io.zeros([1,w])]*3);embedding=stack([next_token,0,0,0]);ctl=controls(slot,1)
  first_draft=target.mtp_teacher(embedding,*ctl,0,None)[0];equal([first_draft],oracle_step(packed,embedding,ctl,1),'shifted prefill MTP');compare('prefill')
 output_ids=[generated];base_slot=len(prompt['ids']);stops=c.get('eos_token_id',[]);stops=stops if isinstance(stops,list) else [stops]
 while len(output_ids)<prompt['max_output'] and output_ids[-1] not in stops:
  remaining=prompt['max_output']-len(output_ids);depth=min(3,remaining-1);proposals=[output_ids[-1]];draft=first_draft;base_key=key;base_value=value;base_last=last
  for j in range(depth):
   proposals.append(draft)
   if j+1<depth:
    embedding=stack([draft,0,0,0]);ctl=controls(base_slot+j,1);source=io.stack_tokens([last]+[io.zeros([1,w])]*3)
    draft=target.mtp_draft(embedding,*ctl,None)[0];equal([draft],oracle_step(source,embedding,ctl,1),'real draft prediction');compare('draft')
  expected=[];normalized=[]
  for j,token in enumerate(proposals):
   prediction=step(reference,token,base_slot+j);expected.append(prediction);normalized.append(normalizer.run([reference.hidden()]))
   if prediction in stops or j+1==len(proposals) or prediction!=proposals[j+1]:break
  padded=proposals+[0]*(4-len(proposals));slots=[float(base_slot+j) for j in range(4)];length=len(proposals)
  result=target.verify_greedy(stack(padded),val([s for s in slots for axis in range(3)],[1,4,3]),val(slots,[1,4]),val([float(j<length) for j in range(4)],[1,4]),padded,stops,[remaining],None)[0]
  equal(result,[expected,len(expected),expected[-1] in stops],'real draft target acceptance')
  actual_hidden=io.verification_f32_sha256(target.hidden());reference_hidden=io.verification_f32_sha256(reference.hidden());equal(actual_hidden,reference_hidden,'committed target hidden')
  actual_state=[];reference_state=[]
  for i in range(64):
   actual_state.append([io.verification_f32_sha256(x) for x in target.snapshot(i)]);reference_state.append([io.verification_f32_sha256(x) for x in reference.snapshot(i)]);equal(actual_state[-1],reference_state[-1],'committed target layer'+str(i))
  key=base_key;value=base_value;last=base_last;count=len(expected);embedding=stack(expected+[0]*(4-count));ctl=controls(base_slot,count);packed=io.stack_tokens(normalized+[io.zeros([1,w])]*(4-count))
  first_draft=target.mtp_teacher(embedding,*ctl,0,None)[0];equal([first_draft],oracle_step(packed,embedding,ctl,count),'real correction prediction');compare('correction')
  record=dict(base_slot=base_slot,proposals=proposals,emitted=result[0],reference_emitted=expected,consumed=count,accepted_drafts=count-1,mtp_stats=json.loads(target.mtp_stats_json()),actual_hidden_sha256=actual_hidden,reference_hidden_sha256=reference_hidden,actual_state_sha256=actual_state,reference_state_sha256=reference_state)
  records.append(record);print('trained_mtp_bridge_iteration PASS',json.dumps({k:v for k,v in record.items() if k not in ['actual_state_sha256','reference_state_sha256']}),flush=True);output_ids+=result[0];base_slot+=count
 result=dict(passed=True,checks=checks,prompt=prompt,output_ids=output_ids,iterations=records,mtp_states=mtp_records,prepared_memory_snapshot=prepared,inventory=inventory_path,scope='Actual pinned target and auxiliary short coding completion parity; diagnostic oracle downloads, not8-task coding or speed admission')
 with open(f['output'],'w') as stream:json.dump(result,stream,indent=2)
finally:
 reference.close()
 if decoder is not None:decoder.close()
 if mtp is not None:mtp.close()
 if oracle is not None:oracle.close()
 for graph in graphs:graph.close()
 garnet.shutdown();memory=json.loads(garnet.memory_stats_json())
 for name in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):assert memory[name]==0,(name,memory[name])
 print('trained_mtp_bridge_closed',json.dumps(memory),flush=True)
