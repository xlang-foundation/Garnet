"""Pinned trained target greedy accept/commit parity, not a speed/coding gate."""
import json,sys
sys.path.insert(0,sys.argv[2]);sys.path.insert(0,sys.argv[3])
import garnet,tensor
assert 'native_greedy_verification' in garnet.backend_info()
from xModel.qwen38.vl_27b.model import create_decoder
from xModel.qwen35.vl_0_8b.decoder_layer import T,norm
from xModel.qwen35.vl_0_8b.prefill_layer import PrefillLayer
with open(sys.argv[1]) as stream:f=json.load(stream)
with open(f['config']) as stream:config=json.load(stream)
c=config['text_config'];w=c['hidden_size'];capacity=f['capacity'];chunk=4
packed=[];flat=None;decoder=None;reference=garnet.decoder_session();records=[];checks=0

def val(data,shape):return tensor.tensor(data,shape=shape,dtype=tensor.float32)
def syms(shapes,name):return [tensor.input(name+str(i),shape=s,dtype=tensor.float32) for i,s in enumerate(shapes)]
def state(owner):return [[io.verification_f32_sha256(part) for part in owner.snapshot(i)] for i in range(64)]
def normal(owner,token,slot,predict=False):return owner.run(base,io.embedding([token]),val([float(slot)]*3,[1,3]),val([float(slot)],[1]),val([1.],[1]),0 if predict else -1)
def restore_reference(history):
 reference.reset()
 for token,slot in history:normal(reference,token,slot)
def seed(start):
 target.reset();reference.reset();history=[]
 for i,token in enumerate([220,1234,912]):normal(target,token,start+i);normal(reference,token,start+i);history.append([token,start+i])
 return history
try:
 decoder=create_decoder(f['checkpoint'],config,capacity,linear_chunk=True);target=decoder.session;io=decoder.io
 base=[layer['graph'] for layer in decoder.layers];reference.build(base,decoder.head)
 for i,layer in enumerate(decoder.layers):
  shapes=[[1,4,w]]+layer['shapes'][1:3]+[[1,4,3],[1,4],[1,4]]
  inputs=syms(shapes,'trained_greedy_layer_'+str(i)+'_')
  @T.fusion(name='trained_greedy_layer_'+str(i),boundary='required')
  def forward(h,a,b,positions,slots,active):
   return PrefillLayer(h,a,b,positions,slots,active,c,i,1,4,capacity,layer['kind'],parallel_attention=True,parallel_linear=True,attention_capacity_limit=4096)
  graph=garnet.compiled_graph();graph.build_verification_shared(forward(*inputs),inputs,f['checkpoint'],layer['graph']);packed.append(graph)
  if i%8==7:print('trained_greedy_compiled',i+1,'layers',flush=True)
 inputs=syms([[4,w]],'trained_greedy_head_')
 @T.fusion(name='trained_greedy_head',boundary='required')
 def head(h):return norm(h,'model.language_model.norm',c)*T.unary_op('qwen_text_head',weight_name='lm_head.weight',bf16_weight=1)
 flat=garnet.compiled_graph();flat.build_shared(head(*inputs),inputs,f['checkpoint'],decoder.head)
 target.prepare_verification(packed,flat);prepared=json.loads(garnet.memory_stats_json());print('trained_greedy_prepared',json.dumps(prepared),flush=True)
 assert prepared['metal_allocated_bytes']<=prepared['metal_recommended_working_set_bytes']-256*1024*1024
 def verify(history,known,length,requested,eos_position,label):
  global checks
  start=history[-1][1]+1;restore_reference(history);proposals=[];oracle=[];current=known
  for t in range(length):proposals.append(current);current=normal(reference,current,start+t,True)[0];oracle.append(current)
  if requested<length:proposals[requested]=(proposals[requested]+1)%c['vocab_size']
  stops=[] if eos_position<0 else [oracle[min(eos_position,length-1)]]
  expected=[]
  for t in range(length):
   expected.append(oracle[t])
   if oracle[t] in stops or t+1==length or oracle[t]!=proposals[t+1]:break
  consumed=len(expected);padded=proposals+[0]*(4-length);slots=[float(start+t) for t in range(4)];flags=[float(t<length) for t in range(4)]
  h=io.stack_tokens([io.embedding([token]) for token in padded])
  result=target.verify_greedy(h,val([s for s in slots for axis in range(3)],[1,4,3]),val(slots,[1,4]),val(flags,[1,4]),padded,stops,[length],None)
  expected_result=[[expected,consumed,expected[-1] in stops]]
  assert result==expected_result,('trained emitted tokens',label,result,expected_result)
  stats=json.loads(target.verification_stats_json())
  for key,value in [('target_conv_checkpoints',48),('target_delta_checkpoints',48),('target_kv_backups',32),('target_attention_dispatches',16),('borrowed_s1_banks',96),('target_affine_misses',0),('target_mpp_affine_dispatches',0),('target_indexed_prefill_dispatches',0),('target_reuse64_prefill_dispatches',0)]:assert stats[key]==value,(label,key,stats[key],value)
  if consumed<length:assert stats['commit_replay_layers']==48 and stats['commit_kv_restores']==32,(label,stats)
  else:assert stats['commit_replay_layers']==0 and stats['commit_kv_restores']==0,(label,stats)
  actual_state=state(target);actual_hidden=io.verification_f32_sha256(target.hidden())
  restore_reference(history)
  for t in range(consumed):normal(reference,proposals[t],start+t)
  wanted_state=state(reference);wanted_hidden=io.verification_f32_sha256(reference.hidden())
  record=dict(label=label,capacity=capacity,active_tokens=length,requested_consumed=requested,consumed=consumed,
   submitted_ids=padded,slots=slots,stops=stops,emitted=expected,hidden_sha256=actual_hidden,reference_hidden_sha256=wanted_hidden,
   state_sha256=actual_state,reference_state_sha256=wanted_state,metrics=stats)
  with open(f['output']+'.cases.jsonl','a') as stream:stream.write(json.dumps(record)+'\n')
  assert actual_state==wanted_state,('trained complete state mismatch',label)
  assert actual_hidden==wanted_hidden,('trained accepted hidden mismatch',label,actual_hidden,wanted_hidden)
  records.append(record);checks+=1
  print('trained_greedy_case PASS',json.dumps(dict(label=label,length=length,consumed=consumed,emitted=expected)),flush=True)
  return history+[[proposals[t],start+t] for t in range(consumed)],expected[-1]
 start=capacity-7 if capacity==4096 else 0
 for count in [1,2,3,4]:verify(seed(start),1000,4,count,-1,'prefix'+str(count))
 for length in [1,2,3]:verify(seed(start),1000,length,length,-1,'partial'+str(length))
 for stop in [0,3]:verify(seed(start),1000,4,4,stop,'eos'+str(stop))
 history,known=verify(seed(capacity-8 if capacity==4096 else 0),1000,4,2,-1,'carried-first')
 verify(history,known,3,3,-1,'carried-second')
 result=dict(passed=True,cases=records,checks=checks,capacity=capacity,batch=1,physical_tokens=4,
  prepared_memory_snapshot=prepared,backend=garnet.backend_info(),scope='Trained target native greedy acceptance/full-state parity; no MTP draft, coding gate or speed measurement',
  boundary_history='Three seed slots with older cache zero, final slot4095; not a full4K prompt' if capacity==4096 else 'Three seed tokens')
 with open(f['output'],'w') as stream:json.dump(result,stream,indent=2)
finally:
 reference.close()
 if decoder is not None:decoder.close()
 for graph in packed:graph.close()
 if flat is not None:flat.close()
 garnet.shutdown();owners=json.loads(garnet.memory_stats_json())
 for key in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):assert owners[key]==0,(key,owners[key])
 print('trained_greedy_closed',json.dumps(owners),flush=True)
