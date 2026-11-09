# SPDX-License-Identifier: Apache-2.0
"""Within-engine alternating real-weight softmax body diagnostic.
XLang args: LIB SOURCE FIXTURE NEW_OUTPUT OWNERS INITIAL_ORDER(0|1).
No HTTP/TTFT/decode/coding claim. All pipelines compile before leases;
selectors change only between drained native batches, before the timer.
"""
import os,sys,json,time
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet,tensor
from xModel.mistral.text_7b.metal_decoder import MistralMetalDecoder
lib,source,fixture_path,output=sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]
owners,initial=int(sys.argv[5]),int(sys.argv[6]);assert owners in (1,4) and initial in (0,1) and not os.path.exists(output)
assert garnet.native_library_path()==lib+'/libgarnet.dylib'
assert 'chunk_mps_contiguous_softmax_optional' in garnet.backend_info()
assert 'chunk_flash_attention_fp32_operands' not in garnet.backend_info()
with open(fixture_path) as f:fixture=json.load(f)
case=next(c for c in fixture['cases'] if c['id']=='context-2048')
with open(fixture['model']+'/config.json') as f:config=json.load(f)
os.environ['GARNET_METAL_PROFILE_OPS']='0'
decoder=None;completed=False;cleanup=None;seeds=[];measured=[];state=[];prepared=None;after=None;predictions=None;hidden_hash=None;warm_state=[];warm_hidden=None;closed_rejected=False
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
 # Warm both routes before recording allocation/state witnesses.
 for pair in range(-3,9):
  for order in range(2):
   enabled=(pair+order+initial)%2
   assert decoder.session.diagnostic_contiguous_softmax_set_enabled(bool(enabled))
   assert decoder.session.diagnostic_contiguous_softmax_enabled==bool(enabled)
   began=time.monotonic();decoder.session.run_leased(graphs,hidden,positions,slots,active,-1,epochs);elapsed=time.monotonic()-began
   s=decoder.session;specialized=s.chunk_attention_contiguous_softmax_dispatches
   assert specialized==32*enabled and s.chunk_attention_mps_dispatches==32
   assert s.affine_profile_dispatches==224 and s.affine_profile_misses==0
   assert s.packed_prefill_dispatches==160 and s.mpp_affine_dispatches==64
   assert s.compact_prefill_last_rows==(1 if owners==1 else 0) and s.compact_prefill_last_first==(0 if owners==1 else 4)
   assert s.kv_inplace_updates==64 and s.chunk_kv_scatter_dispatches==64 and s.layout_vector_dispatches==256+int(owners==1)
   assert [s.compact_prefill_family_count,s.chunk_attention_mps_plan_count,s.chunk_attention_mps_scratch_bytes,s.chunk_attention_mps_bucket_count]==[14,21,1044162560,126]
   memory=json.loads(garnet.memory_stats_json())
   if pair>=0:
    assert memory['metal_allocated_bytes']==prepared['metal_allocated_bytes']
    measured.append(dict(pair=pair,order=order,enabled=enabled,wall_seconds=elapsed,gpu_ms=s.gpu_ms,contiguous_dispatches=specialized,mps_dispatches=s.chunk_attention_mps_dispatches))
  if pair==-1:
   for i in range(32):warm_state.append([decoder.io.verification_f32_sha256(p) for p in decoder.session.snapshot(i)])
   warm_hidden=decoder.io.verification_f32_sha256(decoder.session.hidden())
   prepared=json.loads(garnet.memory_stats_json())
 after=json.loads(garnet.memory_stats_json())
 predictions=decoder.session.predict_rows([epochs[r] if r<owners else 0 for r in range(4)])
 for i in range(32):state.append([decoder.io.verification_f32_sha256(p) for p in decoder.session.snapshot(i)])
 hidden_hash=decoder.io.verification_f32_sha256(decoder.session.hidden());completed=True
finally:
 if decoder is not None:
  closed_session=decoder.session
  decoder.close()
  assert not closed_session.diagnostic_contiguous_softmax_enabled
  try:closed_session.diagnostic_contiguous_softmax_set_enabled(True)
  except Exception:closed_rejected=True
  assert closed_rejected
 garnet.shutdown();cleanup=json.loads(garnet.memory_stats_json())
 for key in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):assert cleanup[key]==0
 assert cleanup['metal_allocated_bytes']<=4*1024*1024
 with open(output,'x') as f:json.dump(dict(completed=completed,library=garnet.native_library_path(),backend=garnet.backend_info(),
  owners=owners,initial_order=initial,batch=4,capacity=4096,chunk=128,offset=1920,warmup_pairs=3,measured_pairs=9,
  seeds=seeds,measured=measured,warm_state_sha256=warm_state,warm_hidden_sha256=warm_hidden,closed_selector_rejected=closed_rejected,state_sha256=state,hidden_sha256=hidden_hash,predictions=predictions,
  prepared=prepared,after=after,cleanup=cleanup,scope=__doc__),f,indent=2)
assert completed
print('paired_contiguous_stage_complete',owners,initial,flush=True)
