# SPDX-License-Identifier: Apache-2.0
"""Paired actual-weight late-chunk layout study, not serving token throughput.

Args: LIB SOURCE FIXTURE NEW_OUTPUT OWNERS(1|4). B4/C4096/T128.
Seed15 real prompt chunks, warm/replay the identical final chunk3+9 times.
Timings include native session.run and completion, exclude embedding and all
state hashes. Replaying identical inputs/slots tests steady late-chunk work;
this is not an independent prompt/latency distribution or HTTP benchmark.
"""
import sys,os,json,time
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet,tensor
from xModel.mistral.text_7b.metal_decoder import MistralMetalDecoder
lib,src,fixture_path,output,owners=sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],int(sys.argv[5])
assert owners in (1,4) and not os.path.exists(output)
assert garnet.native_library_path()==lib+'/libgarnet.dylib'
with open(fixture_path) as f:fixture=json.load(f)
case=next(c for c in fixture['cases'] if c['id']=='context-2048')
with open(fixture['model']+'/config.json') as f:config=json.load(f)
batch=4;chunk=128;decoder=None;seeds=[];measured=[];state=[];hidden_hash=None;prepared=None;cleanup=None;completed=False
try:
 os.environ['GARNET_METAL_PROFILE_OPS']='0'
 decoder=MistralMetalDecoder(fixture['model']+'/model.safetensors',config,batch,4096,resident=True)
 decoder.prepare_prefill(chunk);graphs=decoder._resident_prefill_graphs(chunk);graph_count=len(decoder.prefill_graphs)
 for offset in range(0,2048,chunk):
  values=[decoder.io.embedding([case['prompt_ids'][offset+t] if row<owners else 0 for row in range(batch)]) for t in range(chunk)]
  hidden=decoder.io.stack_tokens(values)
  active=tensor.tensor([int(row<owners) for row in range(batch) for t in range(chunk)],shape=[batch,chunk],dtype=tensor.float32)
  slots=tensor.tensor([offset+t if row<owners else 0 for row in range(batch) for t in range(chunk)],shape=[batch,chunk],dtype=tensor.float32)
  positions=tensor.tensor([offset+t if row<owners else 0 for row in range(batch) for t in range(chunk) for axis in range(3)],shape=[batch,chunk,3],dtype=tensor.float32)
  if offset<1920:
   started=time.monotonic();decoder.session.run(graphs,hidden,positions,slots,active,-1);seeds.append(dict(offset=offset,wall_seconds=time.monotonic()-started))
  else:
   prepared=json.loads(garnet.memory_stats_json())
   for repeat in range(-3,9):
    started=time.monotonic();decoder.session.run(graphs,hidden,positions,slots,active,-1);wall=time.monotonic()-started
    assert decoder.session.kv_inplace_updates==64 and decoder.session.chunk_kv_scatter_dispatches==64 and len(decoder.prefill_graphs)==graph_count
    layouts=decoder.session.layout_vector_dispatches if 'chunk_layout_vector4' in garnet.backend_info() else 0
    assert layouts==(256 if 'chunk_layout_vector4' in garnet.backend_info() else 0)
    if repeat>=0:measured.append(dict(repeat=repeat,wall_seconds=wall,gpu_ms=decoder.session.gpu_ms,layout_vector_dispatches=layouts))
   after=json.loads(garnet.memory_stats_json())
   assert after['metal_allocated_bytes']==prepared['metal_allocated_bytes']
 # Independent full-state hashes are collected after every timed sample.
 for i in range(32):state.append([decoder.io.verification_f32_sha256(part) for part in decoder.session.snapshot(i)])
 hidden_hash=decoder.io.verification_f32_sha256(decoder.session.hidden())
 completed=True
finally:
 if decoder is not None:decoder.close()
 garnet.shutdown();cleanup=json.loads(garnet.memory_stats_json())
 for key in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):assert cleanup[key]==0
 assert cleanup['metal_allocated_bytes']<=4*1024*1024
 with open(output,'x') as f:json.dump(dict(completed=completed,library=garnet.native_library_path(),backend=garnet.backend_info(),case_id=case['id'],batch=batch,capacity=4096,chunk=chunk,owners=owners,offset=1920,warmups=3,seeds=seeds,measured=measured,state_sha256=state,hidden_sha256=hidden_hash,prepared=prepared,after=after if completed else None,cleanup=cleanup,scope=__doc__),f,indent=2)
assert completed
print('actual_chunk_complete',owners,len(measured),flush=True)
