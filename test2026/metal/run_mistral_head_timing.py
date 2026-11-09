# SPDX-License-Identifier: Apache-2.0
"""Native head timing regression: a reported GPU span must fit its synchronous wall interval.
Args LIB SOURCE FIXTURE NEW_OUTPUT. Serial only. Full KV/hidden must remain unchanged.
This measures head telemetry, not model throughput or quality.
"""
import sys,os,json,time,math
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet,tensor
from xModel.mistral.text_7b.metal_decoder import MistralMetalDecoder
with open(sys.argv[3]) as f:fixture=json.load(f)
out=sys.argv[4];assert not os.path.exists(out);os.mkdir(out)
assert garnet.native_library_path()==sys.argv[1]+'/libgarnet.dylib'
with open(fixture['model']+'/config.json') as f:config=json.load(f)
def f32(v,s):return tensor.tensor(v,shape=s,dtype=tensor.float32)
def witness():return dict(kv=[[decoder.io.verification_f32_sha256(v) for v in decoder.session.snapshot(i)] for i in range(32)],hidden=decoder.io.verification_f32_sha256(decoder.session.hidden()))
decoder=None;heads=[];before=None;body_gpu=None;cleanup=None;completed=False
try:
 decoder=MistralMetalDecoder(fixture['model']+'/model.safetensors',config,4,4096,resident=True);decoder.prepare_prefill(128)
 assert decoder.session.configure_token_prefix(['mistral-a4b8f870','fp16g64','native-mistral-tokenizer','text_zero_based_3axis_v1','B4C4096T128-head-timing','text'],2,64*1024*1024)
 assert decoder.session.configure_prediction_confidence()
 epochs=[decoder.session.claim_request_row(100+i)[1] for i in range(4)]
 prompts=[p[:64] for p in fixture['cases'][1]['prompts']]
 hidden=decoder.io.stack_tokens([decoder.io.embedding([p[t] for p in prompts]) for t in range(64)])
 slots=[t for p in prompts for t in range(64)]
 decoder.session.run_tokens(decoder.prefill_profile_graphs(64),hidden,f32([v for v in slots for _ in range(3)],[4,64,3]),f32(slots,[4,64]),f32([1]*256,[4,64]),-1,epochs,[v for p in prompts for v in p])
 body_gpu=decoder.session.gpu_ms;before=witness();canonical=None
 for confidence,mask in [(False,[1,1,1,1]),(False,[1,0,0,0]),(False,[0,1,0,1]),(True,[1,1,1,1])]:
  selected=[epochs[i] if mask[i] else 0 for i in range(4)]
  start=time.monotonic();values=decoder.session.predict_rows_confidence(selected) if confidence else decoder.session.predict_rows(selected);wall=(time.monotonic()-start)*1000
  gpu=decoder.session.gpu_ms;ids=[v[0] for v in values] if confidence else list(values)
  state=witness();record=dict(confidence=confidence,mask=mask,ids=ids,gpu_ms=gpu,wall_ms=wall,state=state);heads.append(record)
  assert state==before,'Head mutated KV or hidden'
  assert math.isfinite(gpu) and 0<gpu<=wall+1.0,'Head GPU timer exceeds synchronous wall interval (stale body duration)'
  assert decoder.session.packed_prefill_dispatches==0 and decoder.session.affine_profile_dispatches==1
  if canonical is None:canonical=list(ids)
  for i in range(4):assert ids[i]==(canonical[i] if mask[i] else -1)
 completed=True
finally:
 if decoder is not None:decoder.close()
 garnet.shutdown();cleanup=json.loads(garnet.memory_stats_json())
 for k in ['garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets']:assert cleanup[k]==0
 assert cleanup['metal_allocated_bytes']<=4*1024*1024
 with open(out+'/results.json','w') as f:json.dump(dict(completed=completed,scope=__doc__,library=garnet.native_library_path(),body_gpu_ms=body_gpu,before=before,heads=heads,cleanup=cleanup),f,indent=2)
assert completed
print('head timing/state regression PASS',len(heads),flush=True)
