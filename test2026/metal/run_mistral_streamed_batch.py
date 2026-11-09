# SPDX-License-Identifier: Apache-2.0
"""Native B4/C4096/T128 leased streaming prefill and ordinary greedy decode.

Args LIB SOURCE FIXTURE NEW_OUTPUT control|candidate. Fresh processes run
serially. Frozen C4 prompt IDs unchanged; readiness/embedding construction and
state witnesses are outside timing. No HTTP, coding quality or serving admission.
"""
import sys,json,os,time
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet,tensor
from xModel.mistral.text_7b.metal_decoder import MistralMetalDecoder
with open(sys.argv[3]) as f:fixture=json.load(f)
out=sys.argv[4];mode=sys.argv[5];assert mode in ('control','candidate') and not os.path.exists(out);os.mkdir(out)
assert garnet.native_library_path()==sys.argv[1]+'/libgarnet.dylib' and 'mistral_streamed_prefill_optional' in garnet.backend_info()
with open(fixture['model']+'/config.json') as f:config=json.load(f)
decoder=None;records=[];prepared=None;cleanup=None

def f32(v,s):return tensor.tensor(v,shape=s,dtype=tensor.float32)
def save(done):
 with open(out+'/results.json','w') as f:json.dump(dict(completed=done,scope=__doc__,mode=mode,fixture=sys.argv[3],library=garnet.native_library_path(),backend=garnet.backend_info(),prepared=prepared,cases=records,cleanup=cleanup),f,indent=2)
def witness(positions):
 return dict(slots=list(positions),state_sha256=[[decoder.io.verification_f32_sha256(p) for p in decoder.session.snapshot(i)] for i in range(32)],hidden_sha256=decoder.io.verification_f32_sha256(decoder.session.hidden()))
try:
 decoder=MistralMetalDecoder(fixture['model']+'/model.safetensors',config,4,4096,resident=True)
 decoder.session.prefill_set_streamed_affine(mode=='candidate');decoder.prepare_prefill(128)
 compiled=decoder.prefill_compile_ms;decoder.prefill_compile_ms=0;graphs=len(decoder.prefill_graphs);base=[l['graph'] for l in decoder.layers]
 assert graphs==224 and decoder.session.streamed_prefill_plans==3
 inventory=json.loads(garnet.memory_inventory_json([],decoder.session));scratch=[x for x in inventory['references'] if x['role'].startswith('streamed_prefill_')]
 assert len(scratch)==3 and len(set(x['buffer_id'] for x in scratch))==3 and sum(inventory['buffers'][x['buffer_id']]['length_bytes'] for x in scratch)==161480704
 with open(out+'/prepared-inventory.json','w') as f:json.dump(inventory,f,indent=2)
 prepared=dict(graphs=graphs,plans=3,compile_ms=compiled,memory=json.loads(garnet.memory_stats_json()))
 print('streamed_batch_ready',mode,prepared['memory']['metal_allocated_bytes'],flush=True)
 for case in fixture['cases']:
  prompts=case['prompts'];lengths=[len(p) for p in prompts];maximum=max(lengths);inputs=[]
  for offset in range(0,maximum,128):
   remaining=min(128,maximum-offset);chunk=2
   while chunk<remaining:chunk*=2
   slots=[offset+t if offset+t<len(p) else 0 for p in prompts for t in range(chunk)]
   active=[int(offset+t<len(p)) for p in prompts for t in range(chunk)];ids=[p[offset+t] if offset+t<len(p) else 0 for p in prompts for t in range(chunk)]
   hidden=decoder.io.stack_tokens([decoder.io.embedding([ids[row*chunk+t] for row in range(4)]) for t in range(chunk)])
   inputs.append(dict(chunk=chunk,offset=offset,active=active,hidden=hidden,position=f32([v for v in slots for _ in range(3)],[4,chunk,3]),slots=f32(slots,[4,chunk]),flags=f32(active,[4,chunk])))
  samples=[];first=None;epochs=None
  for rep in range(-1,3):
   decoder.reset();leases=[decoder.session.claim_request_row(1000+(rep+1)*4+row) for row in range(4)];assert [l[0] for l in leases]==list(range(4));epochs=[l[1] for l in leases]
   frames=[];start=time.monotonic()
   for x in inputs:
    decoder.session.run_leased(decoder.prefill_profile_graphs(x['chunk']),x['hidden'],x['position'],x['slots'],x['flags'],-1,epochs)
    count=sum(x['active']);wanted=160 if mode=='candidate' and count==512 else 0
    assert decoder.session.streamed_prefill_dispatches==wanted and decoder.session.affine_profile_misses==0
    assert decoder.prefill_compile_ms==0 and len(decoder.prefill_graphs)==graphs and decoder.session.streamed_prefill_plans==3
    frames.append(dict(offset=x['offset'],chunk=x['chunk'],active_tokens=count,streamed_dispatches=decoder.session.streamed_prefill_dispatches,gpu_ms=decoder.session.gpu_ms))
   predictions=decoder.session.predict_rows(epochs);head_ms=decoder.session.gpu_ms;elapsed=time.monotonic()-start
   if first is None:first=list(predictions)
   assert predictions==first
   if rep>=0:samples.append(dict(seconds=elapsed,frames=frames,head_gpu_ms=head_ms,gpu_ms=sum(x['gpu_ms'] for x in frames)+head_ms,predictions=list(predictions),epochs=list(epochs)))
  positions=list(lengths);after_prefill=witness(positions);generated=[[p] for p in predictions];done=[p==2 for p in predictions];decode_frames=[];start=time.monotonic()
  while not all(done):
   active=[int(not x) for x in done];ids=[generated[row][-1] if active[row] else 0 for row in range(4)];slots=[positions[row] if active[row] else 0 for row in range(4)]
   predictions=decoder.session.run_leased(base,decoder.io.embedding(ids),f32([v for v in slots for _ in range(3)],[4,3]),f32(slots,[4]),f32(active,[4]),0,[epochs[row] if active[row] else 0 for row in range(4)])
   assert decoder.session.streamed_prefill_dispatches==0 and decoder.session.affine_profile_misses==0
   for row in range(4):
    if active[row]:generated[row].append(predictions[row]);positions[row]+=1;done[row]=predictions[row]==2 or len(generated[row])==128
   decode_frames.append(dict(active=active,gpu_ms=decoder.session.gpu_ms))
  decode_seconds=time.monotonic()-start;after_decode=witness(positions)
  assert positions==[lengths[row]+len(generated[row])-1 for row in range(4)]
  records.append(dict(id=case['id'],prompt_ids=prompts,prefill=samples,generated_ids=generated,decode_seconds=decode_seconds,decode_frames=decode_frames,after_prefill=after_prefill,after_decode=after_decode,memory=json.loads(garnet.memory_stats_json())))
  save(False);print('streamed_batch_case',mode,case['id'],[len(g) for g in generated],flush=True)
  del inputs
finally:
 if decoder is not None:decoder.close()
 garnet.shutdown();cleanup=json.loads(garnet.memory_stats_json())
 for key in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):assert cleanup[key]==0
 assert cleanup['metal_allocated_bytes']<=4*1024*1024
 save(len(records)==len(fixture['cases']))
assert len(records)==len(fixture['cases'])
