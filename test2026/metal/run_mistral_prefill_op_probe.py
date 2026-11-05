# SPDX-License-Identifier: Apache-2.0
"""Serialize operations in one late real-weight prefill chunk for diagnosis.

XLang args: LIB SOURCE FIXTURE CASE OUTPUT BATCH CHUNK PROFILE_OFFSET.
Only physical row0 is active, matching a single request in a B4 serving engine.
Earlier chunks populate real KV with profiling disabled. The selected chunk is
split into individual submissions; its timings cannot establish serving speed.
No token/quality assertion: the output is an operation-cost diagnostic only.
"""
import sys, os, json, time
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet,tensor
from xModel.mistral.text_7b.metal_decoder import MistralMetalDecoder

with open(sys.argv[3]) as stream:fixture=json.load(stream)
case=[c for c in fixture['cases'] if c['id']==sys.argv[4]][0]
output=sys.argv[5];batch=int(sys.argv[6]);chunk=int(sys.argv[7]);target=int(sys.argv[8])
assert not os.path.exists(output) and batch in (1,2,4) and batch*chunk<=512
assert target%chunk==0 and 0<=target<len(case['prompt_ids'])<=4096
with open(fixture['model']+'/config.json') as stream:config=json.load(stream)
decoder=None;records=[];closed=None;backend=garnet.backend_info();library=garnet.native_library_path()
try:
 os.environ['GARNET_METAL_PROFILE_OPS']='0'
 decoder=MistralMetalDecoder(fixture['model']+'/model.safetensors',config,batch,4096,resident=True)
 decoder.prepare_prefill(chunk);graphs=decoder._resident_prefill_graphs(chunk)
 graph_count=len(decoder.prefill_graphs)
 for offset in range(0,target+1,chunk):
  count=min(chunk,len(case['prompt_ids'])-offset)
  values=[decoder.io.embedding([case['prompt_ids'][offset+t] if row==0 and t<count else 0
                              for row in range(batch)]) for t in range(chunk)]
  hidden=decoder.io.stack_tokens(values)
  active=tensor.tensor([int(row==0 and t<count) for row in range(batch) for t in range(chunk)],shape=[batch,chunk],dtype=tensor.float32)
  slots=tensor.tensor([offset+t if row==0 and t<count else 0 for row in range(batch) for t in range(chunk)],shape=[batch,chunk],dtype=tensor.float32)
  positions=tensor.tensor([offset+t if row==0 and t<count else 0 for row in range(batch) for t in range(chunk) for axis in range(3)],shape=[batch,chunk,3],dtype=tensor.float32)
  os.environ['GARNET_METAL_PROFILE_OPS']='1' if offset==target else '0'
  started=time.monotonic();decoder.session.run(graphs,hidden,positions,slots,active,-1)
  assert decoder.session.kv_inplace_updates==64 and len(decoder.prefill_graphs)==graph_count
  records.append(dict(offset=offset,valid_tokens=count,profiled=offset==target,
                      wall_seconds=time.monotonic()-started,memory=json.loads(garnet.memory_stats_json())))
  print('mistral_prefill_op_probe',offset,count,offset==target,flush=True)
finally:
 os.environ['GARNET_METAL_PROFILE_OPS']='0'
 if decoder is not None:decoder.close()
 garnet.shutdown();closed=json.loads(garnet.memory_stats_json())
 for key in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):
  assert closed[key]==0,(key,closed[key])
 assert closed['metal_allocated_bytes']<=4*1024*1024
 with open(output,'x') as stream:
  json.dump(dict(completed=len(records)==target//chunk+1,backend=backend,library=library,
                 case_id=case['id'],batch=batch,chunk=chunk,profile_offset=target,records=records,cleanup=closed,
                 scope=__doc__),stream,indent=2)
assert len(records)==target//chunk+1
