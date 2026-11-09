# SPDX-License-Identifier: Apache-2.0
"""Replay fixed saved verifier inputs, checking complete target IDs/KV/hidden.
No draft inference or draft throughput is measured here. Instrumented timing is diagnostic.
Args: LIB SOURCE SAVED_RUN OUTPUT [CASE_LIMIT]. All inference sequential.
"""
import sys,json,os,time
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet,tensor
from xModel.mistral.text_7b.metal_decoder import MistralMetalDecoder
with open(sys.argv[3]) as f:saved=json.load(f)
inputs=saved['inputs'];output=sys.argv[4];assert not os.path.exists(output)
assert garnet.native_library_path()==sys.argv[1]+'/libgarnet.dylib'
with open(inputs['target_model']+'/config.json') as f:config=json.load(f)
cases=inputs['cases'][:int(sys.argv[5])] if len(sys.argv)>5 else inputs['cases']
records=[];decoder=None;prepared=None;cleanup=None

def val(v,shape):return tensor.tensor(v,shape=shape,dtype=tensor.float32)
def write(done):
 with open(output,'w') as f:json.dump(dict(completed=done,inputs=inputs,source_run=sys.argv[3],
  cases=[c['id'] for c in cases],library=garnet.native_library_path(),backend=garnet.backend_info(),
  timestamp_path=os.environ.get('GARNET_METAL_ENCODER_TIMESTAMPS_PATH'),prepared=prepared,
  records=records,cleanup=cleanup,scope=__doc__),f,indent=2)
try:
 start=time.monotonic();decoder=MistralMetalDecoder(inputs['target_model']+'/model.safetensors',config,1,4096)
 decoder.prepare_prefill(128);decoder.prepare_verification(4);graphs=len(decoder.prefill_graphs)
 prepared=dict(seconds=time.monotonic()-start,memory=json.loads(garnet.memory_stats_json()))
 print('replay_ready',prepared['memory']['metal_allocated_bytes'],flush=True)
 for case in cases:
  original=next(r for r in saved['records'] if r['mode']=='resident' and r['case_id']==case['id'])
  decoder.reset();prompt=case['prompt_ids'];start=time.monotonic()
  known=decoder.prefill_sequence([decoder.io.embedding([t]) for t in prompt],[[[i]*3 for i in range(len(prompt))]])[0]
  prefill=time.monotonic()-start;generated=[known];slot=len(prompt);rounds=[];start=time.monotonic()
  assert known==original['generated_ids'][0]
  for r in original['rounds']:
   assert r['committed_before']==len(generated) and r['slot']==slot and r['submitted_ids'][0]==known
   began=time.monotonic();metrics=None
   if r['proposal']:
    submitted=r['submitted_ids'];length=len(r['proposal'])+1;assert len(submitted)==4
    positions=[float(slot+t) for t in range(4)]
    result=decoder.session.verify_greedy(decoder.io.stack_tokens([decoder.io.embedding([t]) for t in submitted]),
     val([p for p in positions for axis in range(3)],[1,4,3]),val(positions,[1,4]),
     val([float(t<length) for t in range(4)],[1,4]),submitted,inputs['target_stops'],[inputs['output_limit']-len(generated)],None)[0]
    emitted=list(result[0]);consumed=result[1]
    assert result[2]==(emitted[-1] in inputs['target_stops'])
    metrics=json.loads(decoder.session.verification_stats_json())
    assert metrics['target_affine_misses']==0 and metrics['target_affine_dispatches']==225
   else:emitted=[decoder.step([known])[0]];consumed=1
   elapsed=time.monotonic()-began
   assert consumed==r['consumed'] and emitted==r['emitted_ids']
   generated.extend(emitted);known=emitted[-1];slot+=consumed;decoder.slots=[slot]
   rounds.append(dict(slot=r['slot'],consumed=consumed,submitted_ids=r['submitted_ids'],emitted_ids=emitted,
    target_seconds=elapsed,metrics=metrics))
  elapsed=time.monotonic()-start
  assert generated==original['generated_ids'] and slot==original['slots']
  state=[[decoder.io.verification_f32_sha256(p) for p in decoder.session.snapshot(i)] for i in range(32)]
  hidden=decoder.io.verification_f32_sha256(decoder.session.hidden())
  assert state==original['state_sha256'] and hidden==original['hidden_sha256']
  assert len(decoder.prefill_graphs)==graphs and decoder.prefill_compile_ms==0
  records.append(dict(case_id=case['id'],generated_ids=generated,state_sha256=state,hidden_sha256=hidden,
   rounds=rounds,prefill_seconds=prefill,replay_seconds=elapsed,memory=json.loads(garnet.memory_stats_json())))
  write(False);print('replay_case',case['id'],len(generated),'seconds',round(elapsed,3),flush=True)
finally:
 if decoder is not None:decoder.close()
 garnet.shutdown();cleanup=json.loads(garnet.memory_stats_json())
 for k in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):assert cleanup[k]==0,(k,cleanup[k])
 assert cleanup['metal_allocated_bytes']<=4*1024*1024
 write(len(records)==len(cases))
assert len(records)==len(cases)
