# SPDX-License-Identifier: Apache-2.0
"""Untimed actual-model hoist fault/poison/reset/prefix contracts.
XLang args: LIB SOURCE MODEL FROZEN_FIXTURE NEW_OUTPUT
No serving, performance or coding qualification.
"""
import sys,json,os
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet,tensor
from xModel.mistral.text_7b.metal_decoder import MistralMetalDecoder
lib,source,model,fixture_path,output=sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5]
assert not os.path.exists(output) and garnet.native_library_path()==lib+'/libgarnet.dylib'
assert 'chunk_kv_commit_hoist' in garnet.backend_info() and 'kv_hoist_test_faults' in garnet.backend_info()
with open(model+'/config.json') as f:config=json.load(f)
with open(fixture_path) as f:fixture=json.load(f)
decoder=None;completed=False;cases=[];events=[];cleanup=None;baseline={};closed_rejected=False;disarmed=None;memory=None;closed_rejection=None

def emit(e):
 events.append(e);print(json.dumps(e),flush=True)
def reject(label,fn):
 caught=False;message=''
 try:fn()
 except Exception as e:caught=True;message=str(e)
 assert caught,label+' was accepted'
 return dict(label=label,rejected=True,error=message)
try:
 decoder=MistralMetalDecoder(model+'/model.safetensors',config,4,4096,resident=True)
 decoder.prepare_compact_prefill(128);graphs=decoder._resident_prefill_graphs(128);s=decoder.session
 s.silu_profile32_set_enabled(False);s.paired_prefill_set_enabled(False);s.value_profile32_set_enabled(True)
 s.split_score_profile32_set_enabled(True);s.split_score_set_enabled(True);s.prefill_set_packed_affine(True)
 s.configure_token_prefix(['mistral7b-a4b8f870-fp16g64','kv-hoist-fault-contract-v1','mistral-v03-native-tokenizer','text_zero_based_3axis_v1','native-fault-contract','text'],8,384*1024*1024)
 def state():
  layers=[]
  for layer in range(32):layers.append([decoder.io.verification_f32_sha256(value) for value in s.snapshot(layer)])
  return dict(kv=layers,hidden=decoder.io.verification_f32_sha256(s.diagnostic_final_hidden()))
 def frame(offset):
  ids=[fixture['requests'][row]['token_ids'][offset+t] for row in range(4) for t in range(128)]
  h=decoder.io.stack_tokens([decoder.io.embedding([fixture['requests'][row]['token_ids'][offset+t] for row in range(4)]) for t in range(128)])
  return dict(ids=ids,hidden=h,positions=tensor.tensor([offset+t for row in range(4) for t in range(128) for axis in range(3)],shape=[4,128,3],dtype=tensor.float32),slots=tensor.tensor([offset+t for row in range(4) for t in range(128)],shape=[4,128],dtype=tensor.float32),active=tensor.tensor([1]*512,shape=[4,128],dtype=tensor.float32))
 first,second=frame(0),frame(128)
 def run(f,epochs):
  s.run_tokens(graphs,f['hidden'],f['positions'],f['slots'],f['active'],-1,epochs,f['ids'])
  assert s.chunk_kv_hoisted_commits==64 and s.chunk_attention_mps_dispatches==32
  assert [s.diagnostic_tracked_token_count(row,epochs[row]) for row in range(4)]==[int(f['slots'].tolist()[row*128])+128 for row in range(4)]
 def claim(request):
  epochs=[]
  for row in range(4):
   lease=s.claim_request_row(request+row);assert lease[0]==row;epochs.append(lease[1])
  return epochs
 initial_invalid=[reject('negative_fault',lambda:s.diagnostic_arm_kv_hoist_fault(-1)),reject('excess_fault',lambda:s.diagnostic_arm_kv_hoist_fault(3))]
 assert json.loads(s.diagnostic_kv_hoist_fault_witness_json())['mode']==0
 epochs=claim(100);run(first,epochs);seed=state();seed_predictions=s.predict_rows(epochs)
 for row in range(4):s.cache_token_prefix(row,epochs[row])
 stored=json.loads(s.token_prefix_stats_json());assert stored['entries']>0 and stored['entries']<=4
 run(second,epochs);healthy=state();healthy_predictions=s.predict_rows(epochs)
 baseline=dict(seed_state=seed,seed_predictions=seed_predictions,healthy_state=healthy,healthy_predictions=healthy_predictions,initial_cache=stored,initial_epochs=epochs,invalid_arm_calls=initial_invalid)
 emit(dict(event='healthy_control_complete',entries=stored['entries'],predictions=healthy_predictions))
 def restore_seed(request,old_epochs):
  s.reset();zero=state()
  stale=[reject('old_epoch_tracked',lambda:s.run_tokens(graphs,second['hidden'],second['positions'],second['slots'],second['active'],-1,old_epochs,second['ids'])),reject('old_epoch_leased',lambda:s.run_leased(graphs,second['hidden'],second['positions'],second['slots'],second['active'],-1,old_epochs))]
  new_epochs=claim(request);assert all(new_epochs[row]>old_epochs[row] for row in range(4))
  not_ready=reject('new_row_hidden_unready',lambda:s.predict_rows(new_epochs))
  for row in range(4):
   assert s.restore_token_prefix(row,new_epochs[row],fixture['requests'][row]['token_ids'][:128])==128
   assert s.diagnostic_tracked_token_count(row,new_epochs[row])==128
  restored=state();predictions=s.predict_rows(new_epochs);assert restored==seed and predictions==seed_predictions
  return new_epochs,dict(zero_state=zero,stale_rejections=stale,unready_rejection=not_ready,restored_state=restored,restored_predictions=predictions)
 for mode in [1,2]:
  old=epochs;epochs,reset_before=restore_seed(1000+mode*100,old)
  cache_before=json.loads(s.token_prefix_stats_json());assert s.diagnostic_arm_kv_hoist_fault(mode)
  failure=reject('injected_run',lambda:s.run_tokens(graphs,second['hidden'],second['positions'],second['slots'],second['active'],-1,epochs,second['ids']))
  witness=json.loads(s.diagnostic_kv_hoist_fault_witness_json())
  assert witness['mode']==mode and witness['armed']==0 and witness['encoded_commits']==2 and witness['native_abort_called']
  assert witness['pending_outputs']==0 and not witness['live_command'] and witness['live_submissions']==0
  if mode==1:assert witness['submitted_commands']==witness['drained_commands']==witness['completed_commands']==0
  else:assert witness['submitted_commands']>=1 and witness['drained_commands']==witness['completed_commands']==witness['submitted_commands'] and witness['observed_gpu_ms']>0
  emit(dict(event='fault_observed',mode=mode,witness=witness))
  assert s.chunk_kv_hoisted_commits==0 and s.chunk_kv_scatter_dispatches==0
  blocked=[]
  blocked.append(reject('tracked_generation',lambda:s.run_tokens(graphs,second['hidden'],second['positions'],second['slots'],second['active'],-1,epochs,second['ids'])))
  blocked.append(reject('leased_generation',lambda:s.run_leased(graphs,second['hidden'],second['positions'],second['slots'],second['active'],-1,epochs)))
  blocked.append(reject('predict_rows',lambda:s.predict_rows(epochs)))
  blocked.append(reject('snapshot',lambda:s.snapshot(0)))
  blocked.append(reject('final_hidden',lambda:s.diagnostic_final_hidden()))
  blocked.append(reject('hidden',lambda:s.hidden()))
  blocked.append(reject('row_reset',lambda:s.reset_row(0)))
  blocked.append(reject('row_release',lambda:s.release_request_row(0,epochs[0])))
  invalid_metadata=dict(tracked_count=s.diagnostic_tracked_token_count(0,epochs[0]),prefill_count=s.tracked_prefill_prefix_tokens(0,epochs[0]))
  assert invalid_metadata==dict(tracked_count=-1,prefill_count=-1)
  blocked.append(reject('prefix_capture',lambda:s.cache_token_prefix(0,epochs[0])))
  blocked.append(reject('prefix_restore',lambda:s.restore_token_prefix(0,epochs[0],fixture['requests'][0]['token_ids'][:128])))
  blocked.append(reject('arm_while_failed',lambda:s.diagnostic_arm_kv_hoist_fault(mode)))
  cache_after=json.loads(s.token_prefix_stats_json());assert cache_before==cache_after
  old=epochs;epochs,recovery=restore_seed(2000+mode*100,old)
  run(second,epochs);recovered=state();predictions=s.predict_rows(epochs);assert recovered==healthy and predictions==healthy_predictions
  result=dict(mode=mode,failure=failure,witness=witness,epochs_at_failure=old,reset_before=reset_before,blocked=blocked,invalid_metadata=invalid_metadata,cache_before=cache_before,cache_after=cache_after,recovery=recovery,recovered_state=recovered,recovered_predictions=predictions,new_epochs=epochs)
  cases.append(result);emit(dict(event='fault_case_complete',mode=mode,witness=witness,blocked_calls=len(blocked),exact_recovery=True))
 # A full reset must also disarm a fault that has not fired.
 s.diagnostic_arm_kv_hoist_fault(1);old=epochs;epochs,disarmed_reset=restore_seed(3000,old)
 assert json.loads(s.diagnostic_kv_hoist_fault_witness_json())['armed']==0
 run(second,epochs);disarmed_state=state();disarmed_predictions=s.predict_rows(epochs)
 assert disarmed_state==healthy and disarmed_predictions==healthy_predictions
 disarmed=dict(reset=disarmed_reset,state=disarmed_state,predictions=disarmed_predictions,epochs=epochs)
 memory=json.loads(garnet.memory_stats_json());completed=True
finally:
 if decoder is not None:
  closed=s;decoder.close();closed_rejection=reject('closed_fault_arm',lambda:closed.diagnostic_arm_kv_hoist_fault(1));closed_rejected=True
 garnet.shutdown();cleanup=json.loads(garnet.memory_stats_json())
 for key in ['garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets']:assert cleanup[key]==0
 assert cleanup['metal_allocated_bytes']<=4194304
 with open(output,'x') as f:json.dump(dict(completed=completed,backend=garnet.backend_info(),library=garnet.native_library_path(),baseline=baseline,cases=cases,disarmed=disarmed,closed_rejection=closed_rejection,memory=memory,cleanup=cleanup,events=events,scope=__doc__),f,indent=2)
assert completed
print('fault_model_complete',flush=True)
