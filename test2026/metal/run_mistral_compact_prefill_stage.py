# SPDX-License-Identifier: Apache-2.0
"""Paired actual-weight compact-owner late-chunk study, not serving throughput.

XLang args: LIB SOURCE FIXTURE NEW_OUTPUT REVERSE(0|1). One B4/C4096
session prepares all B1/B2 families before execution. Each owner group seeds
15 real prompt chunks. Control uses unleased B4 execution; candidate uses
leased native compact selection on the same weights, cache and scratch.
Three warmups and nine measured replays overwrite the identical final chunk.
Wall time includes session execution/completion, excludes embedding, head and
hashes. Full physical KV/hidden hashes must agree after both variants. Replays
do not measure independent prompts, arrival scheduling or HTTP token speed.
"""
import os,sys,json,time
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet,tensor
from xModel.mistral.text_7b.metal_decoder import MistralMetalDecoder
lib,source,fixture_path,output=sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]
reverse=int(sys.argv[5]);assert reverse in (0,1) and not os.path.exists(output)
assert garnet.native_library_path()==lib+'/libgarnet.dylib'
with open(fixture_path) as f:fixture=json.load(f)
case=next(c for c in fixture['cases'] if c['id']=='context-2048')
with open(fixture['model']+'/config.json') as f:config=json.load(f)
os.environ['GARNET_METAL_PROFILE_OPS']='0'
decoder=None;prepared=None;cleanup=None;completed=False;groups=[]
def memory():return json.loads(garnet.memory_stats_json())
def inventory():
 s=decoder.session
 return [s.compact_prefill_family_count,s.chunk_attention_mps_plan_count,
         s.chunk_attention_mps_scratch_bytes,s.chunk_attention_mps_bucket_count]
def hashes():
 return dict(KV=[[decoder.io.verification_f32_sha256(p) for p in decoder.session.snapshot(i)] for i in range(32)],
             hidden=decoder.io.verification_f32_sha256(decoder.session.hidden()))
try:
 decoder=MistralMetalDecoder(fixture['model']+'/model.safetensors',config,4,4096,resident=True)
 decoder.prepare_compact_prefill(128);graphs=decoder._resident_prefill_graphs(128)
 prepared=memory();plans=inventory();assert plans==[14,21,1044162560,126]
 assert prepared['metal_allocated_bytes']<=prepared['metal_recommended_working_set_bytes']-256*1024*1024
 for label,live in [('owner0',[0]),('owner3',[3]),('owners12',[1,2])]:
  decoder.session.reset();epochs=[]
  for row in range(4):
   lease=decoder.session.claim_request_row(row+1);assert lease[0]==row;epochs.append(lease[1])
  seeds=[]
  for offset in range(0,2048,128):
   hidden=decoder.io.stack_tokens([decoder.io.embedding([case['prompt_ids'][offset+t] if row in live else 0 for row in range(4)]) for t in range(128)])
   active=tensor.tensor([int(row in live) for row in range(4) for t in range(128)],shape=[4,128],dtype=tensor.float32)
   slots=tensor.tensor([offset+t if row in live else 0 for row in range(4) for t in range(128)],shape=[4,128],dtype=tensor.float32)
   positions=tensor.tensor([offset+t if row in live else 0 for row in range(4) for t in range(128) for axis in range(3)],shape=[4,128,3],dtype=tensor.float32)
   if offset<1920:
    began=time.monotonic();decoder.session.run(graphs,hidden,positions,slots,active,-1)
    seeds.append(dict(offset=offset,wall_seconds=time.monotonic()-began))
  variants=[];order=['control','compact'] if not reverse else ['compact','control']
  for variant in order:
   measured=[];steady=None
   for repeat in range(-3,9):
    began=time.monotonic()
    if variant=='control':decoder.session.run(graphs,hidden,positions,slots,active,-1)
    else:decoder.session.run_leased(graphs,hidden,positions,slots,active,-1,epochs)
    wall=time.monotonic()-began;s=decoder.session
    rows=len(live) if variant=='compact' else 0
    assert (s.compact_prefill_last_rows,s.compact_prefill_last_first)==(rows,live[0] if rows else 4)
    assert s.kv_inplace_updates==64 and s.chunk_kv_scatter_dispatches==64
    assert s.layout_vector_dispatches==256+int(bool(rows))
    assert inventory()==plans
    current=memory()
    if repeat==-1:steady=current
    if repeat>=0:
     assert current['metal_allocated_bytes']==steady['metal_allocated_bytes']
     measured.append(dict(repeat=repeat,wall_seconds=wall,gpu_ms=s.gpu_ms,
                          compact_rows=rows,first=s.compact_prefill_last_first,
                          layout_dispatches=s.layout_vector_dispatches))
   variants.append(dict(variant=variant,measured=measured,state_sha256=hashes(),steady=steady,after=memory()))
   print('compact_stage_variant',label,variant,'PASS',flush=True)
  assert variants[0]['state_sha256']==variants[1]['state_sha256'],('physical state differs',label)
  groups.append(dict(label=label,live=live,seeds=seeds,variants=variants))
 completed=True
finally:
 if decoder is not None:decoder.close()
 garnet.shutdown();cleanup=memory()
 for key in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):assert cleanup[key]==0
 assert cleanup['metal_allocated_bytes']<=4*1024*1024
 with open(output,'x') as f:json.dump(dict(completed=completed,library=garnet.native_library_path(),backend=garnet.backend_info(),
  reverse=reverse,case_id=case['id'],batch=4,capacity=4096,chunk=128,offset=1920,warmups=3,repeats=9,
  prepared=prepared,inventory=plans if prepared is not None else None,groups=groups,cleanup=cleanup,scope=__doc__),f,indent=2)
assert completed
print('compact_stage_complete',reverse,len(groups),flush=True)
