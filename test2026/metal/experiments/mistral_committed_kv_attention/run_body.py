# SPDX-License-Identifier: Apache-2.0
"""Actual-weight late-chunk committed KV attention study; no serving/TTFT result.
XLang args: LIB SOURCE FIXTURE NEW_OUTPUT OWNERS(1|4) ENABLED(0|1).
Native profiles precompile all bucket pipelines, seed15 real prompt chunks,
then replay the same offset1920 chunk3warmup+9measured times. All embedding,
head, complete state hashing and HTTP/scheduling are outside the body spans.
FP32 operands are retained; reduction grouping differs. No whole-model bit-identity claim.
"""
import os,sys,json,time
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet,tensor
from xModel.mistral.text_7b.metal_decoder import MistralMetalDecoder
lib,source,fixture_path,output=sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]
owners,enabled=int(sys.argv[5]),int(sys.argv[6]);assert owners in (1,4) and enabled in (0,1) and not os.path.exists(output)
assert garnet.native_library_path()==lib+'/libgarnet.dylib'
assert ('committed_kv_mps_fp32_v1' in garnet.backend_info())==bool(enabled)
assert 'chunk_flash_attention_fp32_operands' not in garnet.backend_info()
with open(fixture_path) as f:fixture=json.load(f)
case=next(c for c in fixture['cases'] if c['id']=='context-2048')
with open(fixture['model']+'/config.json') as f:config=json.load(f)
os.environ['GARNET_METAL_PROFILE_OPS']='0'
decoder=None;completed=False;cleanup=None;seeds=[];measured=[];state=[];prepared=None;after=None;predictions=None;hidden_hash=None;hidden_values=None
try:
 decoder=MistralMetalDecoder(fixture['model']+'/model.safetensors',config,4,4096,resident=True)
 decoder.prepare_compact_prefill(128);graphs=decoder._resident_prefill_graphs(128);epochs=[]
 # Match the admitted current profile explicitly; do not inherit optional defaults.
 decoder.session.silu_profile32_set_enabled(False)
 decoder.session.paired_prefill_set_enabled(False)
 decoder.session.value_profile32_set_enabled(True)
 decoder.session.split_score_profile32_set_enabled(True)
 decoder.session.split_score_set_enabled(True)
 decoder.session.prefill_set_packed_affine(True)
 assert decoder.session.packed_prefill_plans==3
 for row in range(4):
  lease=decoder.session.claim_request_row(row+1);assert lease[0]==row;epochs.append(lease[1])
 for offset in range(0,2048,128):
  hidden=decoder.io.stack_tokens([decoder.io.embedding([case['prompt_ids'][offset+t] if row<owners else 0 for row in range(4)]) for t in range(128)])
  active=tensor.tensor([int(row<owners) for row in range(4) for t in range(128)],shape=[4,128],dtype=tensor.float32)
  slots=tensor.tensor([offset+t if row<owners else 0 for row in range(4) for t in range(128)],shape=[4,128],dtype=tensor.float32)
  positions=tensor.tensor([offset+t if row<owners else 0 for row in range(4) for t in range(128) for axis in range(3)],shape=[4,128,3],dtype=tensor.float32)
  if offset<1920:
   began=time.monotonic();decoder.session.run_leased(graphs,hidden,positions,slots,active,-1,epochs)
   seeds.append(dict(offset=offset,wall_seconds=time.monotonic()-began))
 for repeat in range(-3,9):
  began=time.monotonic();decoder.session.run_leased(graphs,hidden,positions,slots,active,-1,epochs);elapsed=time.monotonic()-began
  s=decoder.session;specialized=s.chunk_committed_kv_attention_dispatches
  assert specialized==32*enabled and s.chunk_attention_mps_dispatches==32
  assert s.chunk_kv_hoisted_commits==64
  assert s.chunk_attention_mps_matrices==32*owners*8*(2 if enabled else 4)
  assert s.chunk_attention_mps_skipped_matrices==0
  assert s.affine_profile_dispatches==224 and s.affine_profile_misses==0
  assert s.packed_prefill_dispatches==160 and s.mpp_affine_dispatches==64
  assert s.compact_prefill_last_rows==(1 if owners==1 else 0) and s.compact_prefill_last_first==(0 if owners==1 else 4)
  assert s.kv_inplace_updates==64 and s.chunk_kv_scatter_dispatches==64 and s.layout_vector_dispatches==256+int(owners==1)
  assert [s.compact_prefill_family_count,s.chunk_attention_mps_plan_count,s.chunk_attention_mps_scratch_bytes,s.chunk_attention_mps_bucket_count]==[14,21,1044162560,126]
  memory=json.loads(garnet.memory_stats_json())
  if repeat==-1:prepared=memory
  if repeat>=0:
   assert memory['metal_allocated_bytes']==prepared['metal_allocated_bytes']
   measured.append(dict(repeat=repeat,wall_seconds=elapsed,gpu_ms=s.gpu_ms,committed_dispatches=specialized,mps_dispatches=s.chunk_attention_mps_dispatches))
 after=json.loads(garnet.memory_stats_json())
 predictions=decoder.session.predict_rows([epochs[r] if r<owners else 0 for r in range(4)])
 for i in range(32):state.append([decoder.io.verification_f32_sha256(p) for p in decoder.session.snapshot(i)])
 hidden=decoder.session.hidden();hidden_hash=decoder.io.verification_f32_sha256(hidden);hidden_values=hidden.tolist();completed=True
finally:
 if decoder is not None:decoder.close()
 garnet.shutdown();cleanup=json.loads(garnet.memory_stats_json())
 for key in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):assert cleanup[key]==0
 assert cleanup['metal_allocated_bytes']<=4*1024*1024
 with open(output,'x') as f:json.dump(dict(completed=completed,library=garnet.native_library_path(),backend=garnet.backend_info(),
  owners=owners,enabled=enabled,batch=4,capacity=4096,chunk=128,offset=1920,warmups=3,repeats=9,
  seeds=seeds,measured=measured,state_sha256=state,hidden_sha256=hidden_hash,hidden_values=hidden_values,predictions=predictions,
  prepared=prepared,after=after,cleanup=cleanup,scope=__doc__),f,indent=2)
assert completed
print('committed_stage_complete',owners,enabled,flush=True)
