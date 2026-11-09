# SPDX-License-Identifier: Apache-2.0
"""Actual-weight full2048-token prefill body comparison for cooperative projection tiles.

XLang args: LIB SOURCE FIXTURE NEW_OUTPUT VARIANT(control|candidate).
Both engines prepare compact128 families before any dispatch. One warmup and
three complete cold-cache prefills per owner group. Timing excludes reset,
leases, embedding, head and complete KV/hidden hashes. Changes in library policy
are admitted separately; these measurements are not serving throughput.
"""
import sys,os,json,time
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet,tensor
from xModel.mistral.text_7b.metal_decoder import MistralMetalDecoder
with open(sys.argv[3]) as f:fixture=json.load(f)
with open(fixture['model']+'/config.json') as f:config=json.load(f)
variant=sys.argv[5];assert variant in ('control','candidate') and not os.path.exists(sys.argv[4])
assert ('mpp_affine_reuse128_optional' in garnet.backend_info())==(variant=='candidate')
decoder=None;prepared=None;groups=[];closed=None;finished=False;chunk=128
prompt=next(c for c in fixture['cases'] if c['id']=='context-2048')['prompt_ids'][:2048]
def memory():return json.loads(garnet.memory_stats_json())
def f32(v,s):return tensor.tensor(v,shape=s,dtype=tensor.float32)
def save():
 with open(sys.argv[4],'w') as f:json.dump(dict(completed=finished,variant=variant,groups=groups,prepared=prepared,cleanup=closed,library=garnet.native_library_path(),backend=garnet.backend_info(),scope=__doc__),f,indent=2)
def inputs(live):
 result=[]
 for offset in range(0,2048,chunk):
  hidden=decoder.io.stack_tokens([decoder.io.embedding([prompt[offset+t] if row in live else 0 for row in range(4)]) for t in range(chunk)])
  slots=[offset+t if row in live else 0 for row in range(4) for t in range(chunk)];active=[int(row in live) for row in range(4) for t in range(chunk)]
  result.append([hidden,f32([s for s in slots for _ in range(3)],[4,chunk,3]),f32(slots,[4,chunk]),f32(active,[4,chunk])])
 return result
def hashes():return dict(KV=[[decoder.io.verification_f32_sha256(p) for p in decoder.session.snapshot(i)] for i in range(32)],hidden=decoder.io.verification_f32_sha256(decoder.session.hidden()))
try:
 decoder=MistralMetalDecoder(fixture['model']+'/model.safetensors',config,4,4096,resident=True);decoder.prepare_compact_prefill(128);prepared=memory()
 assert decoder.session.compact_prefill_family_count==14 and [decoder.session.chunk_attention_mps_plan_count,decoder.session.chunk_attention_mps_scratch_bytes,decoder.session.chunk_attention_mps_bucket_count]==[21,1044162560,126]
 graphs=decoder.prefill_profile_graphs(128)
 for live in [[0],[0,1,2,3]]:
  bundles=inputs(live);group=dict(live=live,measured=[]);groups.append(group);steady=None;expected=None;expected_prediction=None
  for repeat in range(-1,3):
   decoder.session.reset();epochs=[]
   for row in range(4):
    lease=decoder.session.claim_request_row(row+1);assert lease[0]==row;epochs.append(lease[1])
   gpu_ms=0;start=time.monotonic()
   for args in bundles:
    decoder.session.run_leased(graphs,args[0],args[1],args[2],args[3],-1,epochs)
    assert decoder.session.compact_prefill_last_rows==(1 if len(live)==1 else 0)
    assert decoder.session.chunk_kv_scatter_dispatches==64
    counts=dict(reuse64=decoder.session.mpp_reuse64_dispatches,reuse128=decoder.session.mpp_reuse128_dispatches if variant=='candidate' else 0)
    assert counts==dict(reuse64=0 if variant=='candidate' else 160,reuse128=160 if variant=='candidate' else 0)
    group['dispatches_per_chunk']=counts
    gpu_ms+=decoder.session.gpu_ms
   wall=time.monotonic()-start;prediction=decoder.session.predict_rows([epochs[r] if r in live else 0 for r in range(4)])
   state=hashes();sample=memory();assert sample['metal_allocated_bytes']<=sample['metal_recommended_working_set_bytes']-256*1024*1024
   if repeat==-1:steady=sample['metal_allocated_bytes'];expected=state;expected_prediction=prediction
   else:
    assert sample['metal_allocated_bytes']==steady and state==expected and prediction==expected_prediction
    group['measured'].append(dict(repeat=repeat,body_wall_seconds=wall,body_gpu_ms=gpu_ms,memory=sample))
   group.update(state_sha256=state,predictions=prediction);save();print('reuse128_stage',variant,live,repeat,wall,flush=True)
  bundles.clear()
 finished=True
finally:
 if decoder is not None:decoder.close()
 garnet.shutdown();closed=memory()
 for k in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):assert closed[k]==0
 assert closed['metal_allocated_bytes']<=4*1024*1024
 save()
assert finished
