# SPDX-License-Identifier: Apache-2.0
"""Trained Mistral target verification/state parity; proposals here use an oracle.

Args: MODEL LIB SOURCE FIXTURE NEW_OUTPUT CAPACITY. No draft-acceptance, coding
quality or speed claim follows from oracle-generated proposals.
"""
import json,sys,os
sys.path.insert(0,sys.argv[2]);sys.path.insert(0,sys.argv[3])
import garnet,tensor
from xModel.mistral.text_7b.metal_decoder import MistralMetalDecoder
model,output,capacity=sys.argv[1],sys.argv[5],int(sys.argv[6])
assert capacity in (256,4096) and not os.path.exists(output)
assert garnet.native_library_path()==sys.argv[2]+'/libgarnet.dylib'
assert 'native_greedy_dense_verification' in garnet.backend_info()
assert 'linear_chunk_conv' not in garnet.backend_info() and 'linear_chunk_delta' not in garnet.backend_info()
os.mkdir(output)
with open(model+'/config.json') as stream:config=json.load(stream)
with open(sys.argv[4]) as stream:fixture=json.load(stream)
decoder=None;reference=garnet.decoder_session();records=[];prepared=None;closed=None
def val(values,shape):return tensor.tensor(values,shape=shape,dtype=tensor.float32)
def save(completed):
 with open(output+'/status.json','w') as stream:
  json.dump(dict(completed=completed,capacity=capacity,batch=1,verification_tokens=4,records=records,
   backend=garnet.backend_info(),library=garnet.native_library_path(),prepared=prepared,cleanup=closed,
   scope=__doc__),stream,indent=2)
def normal(owner,token,slot,predict=False):
 return owner.run(base,io.embedding([token]),val([float(slot)]*3,[1,3]),val([float(slot)],[1]),val([1.],[1]),0 if predict else -1)
def state(owner):return [[io.verification_f32_sha256(part) for part in owner.snapshot(i)] for i in range(32)]
def restore_reference(history):
 reference.reset()
 for token,slot in history:normal(reference,token,slot)
def seed(start):
 target.reset();reference.reset();history=[]
 for i,token in enumerate(fixture['cases'][0]['prompt_ids'][:3]):
  normal(target,token,start+i);normal(reference,token,start+i);history.append([token,start+i])
 return history
try:
 decoder=MistralMetalDecoder(model+'/model.safetensors',config,1,capacity,resident=True)
 target=decoder.session;io=decoder.io;base=[layer['graph'] for layer in decoder.layers]
 reference.build(base,decoder.head)
 decoder.prepare_verification(4)
 assert decoder.prepare_verification(4)
 for bad in (0,5,True,3):
  rejected=False
  try:decoder.prepare_verification(bad)
  except Exception:rejected=True
  assert rejected,('invalid/replaced profile accepted',bad)
 assert target.fused_decode_profiles>=1 and [g.fused_decode_sites for g in decoder.verification_graphs]==[1]*32
 prepared=json.loads(garnet.memory_stats_json());print('mistral_verify_prepared',json.dumps(prepared),flush=True)
 assert prepared['metal_allocated_bytes']<=prepared['metal_recommended_working_set_bytes']-256*1024*1024
 def verify(history,known,length,requested,eos_position,label,rebuild=True):
  start=history[-1][1]+1
  if rebuild:restore_reference(history)
  proposals=[];oracle=[];current=known
  for t in range(length):
   proposals.append(current);current=normal(reference,current,start+t,True)[0];oracle.append(current)
  if requested<length:proposals[requested]=(proposals[requested]+1)%config['vocab_size']
  stops=[] if eos_position<0 else [oracle[eos_position]]
  expected=[]
  for t in range(length):
   expected.append(oracle[t])
   if oracle[t] in stops or t+1==length or oracle[t]!=proposals[t+1]:break
  consumed=len(expected);padded=proposals+[0]*(4-length);slots=[float(start+t) for t in range(4)]
  hidden=io.stack_tokens([io.embedding([token]) for token in padded])
  result=target.verify_greedy(hidden,val([slot for slot in slots for axis in range(3)],[1,4,3]),
   val(slots,[1,4]),val([float(t<length) for t in range(4)],[1,4]),padded,stops,[length],None)
  assert result==[[expected,consumed,expected[-1] in stops]],('emitted IDs/commit count',label,result,expected)
  stats=json.loads(target.verification_stats_json())
  assert stats['target_fused_q4_decode_blocks']==(32 if length==4 else 0),('native fused verifier count',label,stats)
  assert 0<=stats['target_fused_q4_decode_input_copies']<=stats['target_fused_q4_decode_blocks']

  for key,wanted in [('target_affine_dispatches',225),('target_affine_misses',0),('target_kv_backups',64),
   ('target_attention_dispatches',32),('target_conv_checkpoints',0),('target_delta_checkpoints',0),
   ('target_mpp_affine_dispatches',0),('target_indexed_prefill_dispatches',0),('target_reuse64_prefill_dispatches',0),
   ('borrowed_s1_banks',0),('commit_replay_layers',0),('commit_kv_restores',64 if consumed<length else 0)]:
   assert stats[key]==wanted,(label,key,stats[key],wanted)
  actual=state(target);actual_hidden=io.verification_f32_sha256(target.hidden())
  restore_reference(history)
  for t in range(consumed):normal(reference,proposals[t],start+t)
  wanted=state(reference);wanted_hidden=io.verification_f32_sha256(reference.hidden())
  assert actual==wanted,('complete KV state',label)
  assert actual_hidden==wanted_hidden,('accepted hidden',label)
  record=dict(label=label,active_tokens=length,requested_consumed=requested,consumed=consumed,
   submitted_ids=padded,slots=slots,stops=stops,emitted=expected,state_sha256=actual,reference_state_sha256=wanted,
   hidden_sha256=actual_hidden,reference_hidden_sha256=wanted_hidden,metrics=stats)
  records.append(record);save(False)
  print('mistral_verify_case',label,'PASS',consumed,flush=True)
  return history+[[proposals[t],start+t] for t in range(consumed)],expected[-1]
 start=capacity-7 if capacity==4096 else 0
 for count in (1,2,3,4):verify(seed(start),1000,4,count,-1,'prefix'+str(count))
 for length in (1,2,3):verify(seed(start),1000,length,length,-1,'partial'+str(length))
 for stop in (0,3):verify(seed(start),1000,4,4,stop,'eos'+str(stop))
 history,known=verify(seed(capacity-8 if capacity==4096 else 0),1000,4,2,-1,'carried-first')
 verify(history,known,3,3,-1,'carried-second')
 if capacity==4096:
  # Full prompt state, not just a few seed tokens near the context boundary.
  ids=next(case for case in fixture['cases'] if case['id']=='context-256')['prompt_ids']
  for requested in (2,4):
   target.reset();reference.reset();history=[]
   for i,token in enumerate(ids):normal(target,token,i);normal(reference,token,i);history.append([token,i])
   verify(history,1000,4,requested,-1,'full-prompt-prefix'+str(requested),False)
finally:
 reference.close()
 if decoder is not None:decoder.close()
 garnet.shutdown();closed=json.loads(garnet.memory_stats_json())
 for key in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):
  assert closed[key]==0,(key,closed[key])
 save(len(records)==(13 if capacity==4096 else 11))
 print('mistral_verify_closed',json.dumps(closed),flush=True)
assert len(records)==(13 if capacity==4096 else 11)
