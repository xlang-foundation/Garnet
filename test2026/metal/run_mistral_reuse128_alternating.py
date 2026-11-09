# SPDX-License-Identifier: Apache-2.0
"""Same-session paired actual-weight prefill; no serving-speed admission.

Args: LIB SOURCE FIXTURE NEW_OUTPUT. One fixed compact128 engine prepares both
routes before readiness. Each owner group warms both routes, then alternates
four control/candidate pairs in AB/BA/AB/BA order. All runs reset private KV.
Timing excludes route selection, reset/claims, embedding, head and full hashes.
"""
import sys,os,json,time
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet,tensor
from xModel.mistral.text_7b.metal_decoder import MistralMetalDecoder
with open(sys.argv[3]) as f:fixture=json.load(f)
with open(fixture['model']+'/config.json') as f:config=json.load(f)
assert not os.path.exists(sys.argv[4]) and 'verification_affine_route_control' in garnet.backend_info()
decoder=None;prepared=None;closed=None;finished=False;groups=[];chunk=128;pipelines=0
prompt=next(c for c in fixture['cases'] if c['id']=='context-2048')['prompt_ids'][:2048]
def memory():return json.loads(garnet.memory_stats_json())
def f32(v,s):return tensor.tensor(v,shape=s,dtype=tensor.float32)
def save():
 with open(sys.argv[4],'w') as f:json.dump(dict(completed=finished,groups=groups,prepared=prepared,cleanup=closed,library=garnet.native_library_path(),backend=garnet.backend_info(),prepared_pipeline_count=pipelines,scope=__doc__),f,indent=2)
def inputs(live):
 result=[]
 for offset in range(0,2048,chunk):
  hidden=decoder.io.stack_tokens([decoder.io.embedding([prompt[offset+t] if row in live else 0 for row in range(4)]) for t in range(chunk)])
  slots=[offset+t if row in live else 0 for row in range(4) for t in range(chunk)];active=[int(row in live) for row in range(4) for t in range(chunk)]
  result.append([hidden,f32([s for s in slots for _ in range(3)],[4,chunk,3]),f32(slots,[4,chunk]),f32(active,[4,chunk])])
 return result
def hashes():return dict(KV=[[decoder.io.verification_f32_sha256(p) for p in decoder.session.snapshot(i)] for i in range(32)],hidden=decoder.io.verification_f32_sha256(decoder.session.hidden()))
try:
 decoder=MistralMetalDecoder(fixture['model']+'/model.safetensors',config,4,4096,resident=True);decoder.prepare_compact_prefill(128);prepared=memory();pipelines=decoder.session.affine_profile_pipeline_count
 assert decoder.session.compact_prefill_family_count==14 and [decoder.session.chunk_attention_mps_plan_count,decoder.session.chunk_attention_mps_scratch_bytes,decoder.session.chunk_attention_mps_bucket_count]==[21,1044162560,126]
 graphs=decoder.prefill_profile_graphs(128)
 for live in [[0],[0,1],[0,1,2,3]]:
  bundles=inputs(live);group=dict(live=live,runs=[]);groups.append(group);steady=None;expected=None;expected_prediction=None
  for pair in range(-1,4):
   order=[False,True] if pair<0 or pair%2==0 else [True,False]
   for enabled in order:
    decoder.session.reset();assert decoder.session.verification_set_affine_reuse128(enabled);epochs=[]
    for row in range(4):
     lease=decoder.session.claim_request_row(row+1);assert lease[0]==row;epochs.append(lease[1])
    gpu_ms=0;counts=[];start=time.monotonic()
    for args in bundles:
     decoder.session.run_leased(graphs,args[0],args[1],args[2],args[3],-1,epochs)
     assert decoder.session.compact_prefill_last_rows==(len(live) if len(live)<4 else 0)
     assert decoder.session.chunk_kv_scatter_dispatches==64
     count=dict(reuse64=decoder.session.mpp_reuse64_dispatches,reuse128=decoder.session.mpp_reuse128_dispatches)
     assert count==dict(reuse64=0 if enabled else 160,reuse128=160 if enabled else 0)
     assert decoder.session.affine_profile_misses==0 and decoder.session.affine_profile_pipeline_count==pipelines
     counts.append(count);gpu_ms+=decoder.session.gpu_ms
    wall=time.monotonic()-start;prediction=decoder.session.predict_rows([epochs[r] if r in live else 0 for r in range(4)])
    state=hashes();sample=memory();assert sample['metal_allocated_bytes']<=sample['metal_recommended_working_set_bytes']-256*1024*1024
    if steady is None:steady=sample['metal_allocated_bytes'];expected=state;expected_prediction=prediction
    else:assert sample['metal_allocated_bytes']==steady and state==expected and prediction==expected_prediction
    group['runs'].append(dict(pair=pair,variant='candidate' if enabled else 'control',warmup=pair<0,body_wall_seconds=wall,body_gpu_ms=gpu_ms,memory=sample,chunk_dispatches=counts,pipeline_count=decoder.session.affine_profile_pipeline_count,state_sha256=state,predictions=prediction))
    save();print('reuse128_alternating',live,pair,enabled,wall,flush=True)
  bundles.clear()
 assert decoder.session.verification_set_affine_reuse128(True);finished=True
finally:
 if decoder is not None:decoder.close()
 garnet.shutdown();closed=memory()
 for k in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):assert closed[k]==0
 assert closed['metal_allocated_bytes']<=4*1024*1024
 save()
assert finished
