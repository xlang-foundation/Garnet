# SPDX-License-Identifier: Apache-2.0
"""Native B1 prefill A/B collection, fixed512 up-to profile; no HTTP/quality admission.

Args LIB SOURCE FIXTURE NEW_OUTPUT control|candidate. One process at a time.
Includes ordinary128-token continuation, private all-layer KV/hidden witnesses,
short and ragged fallbacks. Preparation and embedding materialization are outside
prefill timing. Frozen serving references are not rerun or altered.
"""
import json,os,sys,time
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet
from xModel.mistral.text_7b.metal_decoder import MistralMetalDecoder
with open(sys.argv[3]) as f:fixture=json.load(f)
out=sys.argv[4];mode=sys.argv[5];assert mode in ('control','candidate') and not os.path.exists(out);os.mkdir(out)
assert garnet.native_library_path()==sys.argv[1]+'/libgarnet.dylib'
assert 'mistral_streamed_prefill_optional' in garnet.backend_info()
with open(fixture['model']+'/config.json') as f:config=json.load(f)
decoder=None;results=[];prepared=None;cleanup=None

def save(completed):
 with open(out+'/results.json','w') as f:json.dump(dict(completed=completed,scope=__doc__,mode=mode,fixture=sys.argv[3],backend=garnet.backend_info(),library=garnet.native_library_path(),prepared=prepared,cases=results,cleanup=cleanup),f,indent=2)

def witness():
 return dict(state_sha256=[[decoder.io.verification_f32_sha256(part) for part in decoder.session.snapshot(layer)] for layer in range(32)],hidden_sha256=decoder.io.verification_f32_sha256(decoder.session.hidden()),slots=list(decoder.slots))

try:
 decoder=MistralMetalDecoder(fixture['model']+'/model.safetensors',config,1,4096)
 decoder.session.prefill_set_streamed_affine(mode=='candidate');decoder.prepare_prefill(512)
 compiled=decoder.prefill_compile_ms;decoder.prefill_compile_ms=0;graphs=len(decoder.prefill_graphs)
 assert decoder.session.streamed_prefill_plans==3
 inventory=json.loads(garnet.memory_inventory_json([],decoder.session))
 scratch=[r for r in inventory['references'] if r['role'].startswith('streamed_prefill_')]
 assert len(scratch)==3 and len(set(r['buffer_id'] for r in scratch))==3
 assert sum(inventory['buffers'][r['buffer_id']]['length_bytes'] for r in scratch)==161480704
 with open(out+'/prepared-inventory.json','w') as f:json.dump(inventory,f,indent=2)
 prepared=dict(graphs=graphs,compile_ms=compiled,plans=decoder.session.streamed_prefill_plans,memory=json.loads(garnet.memory_stats_json()))
 print('streamed_model_ready',mode,prepared['memory']['metal_allocated_bytes'],flush=True)
 for case in fixture['cases']:
  ids=case['prompt_ids'];hidden=[decoder.io.embedding([token]) for token in ids];samples=[];first=None
  for rep in range(-1,3):
   decoder.reset();chunks=[];start=time.monotonic();prediction=None
   for offset in range(0,len(ids),512):
    length=min(512,len(ids)-offset)
    prediction=decoder.prefill_sequence(hidden[offset:offset+length],[[[offset+i]*3 for i in range(length)]])
    expected=160 if mode=='candidate' and length==512 else 0
    assert decoder.session.streamed_prefill_dispatches==expected and decoder.session.affine_profile_misses==0
    assert decoder.session.streamed_prefill_plans==3 and decoder.prefill_compile_ms==0 and len(decoder.prefill_graphs)==graphs
    chunks.append(dict(length=length,streamed_dispatches=decoder.session.streamed_prefill_dispatches,gpu_ms=decoder.prefill_profile['gpu_ms']))
   elapsed=time.monotonic()-start
   assert decoder.slots==[len(ids)]
   if first is None:first=prediction[0]
   assert prediction==[first]
   if rep>=0:samples.append(dict(seconds=elapsed,chunks=chunks,gpu_ms=sum(x['gpu_ms'] for x in chunks)))
  after_prefill=witness();generated=[prediction[0]];start=time.monotonic()
  while len(generated)<128 and generated[-1]!=2:generated.append(decoder.step([generated[-1]])[0])
  decode_seconds=time.monotonic()-start;after_decode=witness()
  assert decoder.slots==[len(ids)+len(generated)-1]
  results.append(dict(id=case['id'],prompt_ids=ids,prefill=samples,generated_ids=generated,decode_seconds=decode_seconds,after_prefill=after_prefill,after_decode=after_decode,memory=json.loads(garnet.memory_stats_json())))
  save(False);print('streamed_model_case',mode,case['id'],len(generated),flush=True)
finally:
 if decoder is not None:decoder.close()
 garnet.shutdown();cleanup=json.loads(garnet.memory_stats_json())
 for key in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):assert cleanup[key]==0
 assert cleanup['metal_allocated_bytes']<=4*1024*1024
 save(len(results)==len(fixture['cases']))
assert len(results)==len(fixture['cases'])
