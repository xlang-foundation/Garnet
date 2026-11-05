# SPDX-License-Identifier: Apache-2.0
"""Paired full2048-token prefill body study over the same prepared native engine.

XLang args: LIB SOURCE FIXTURE NEW_OUTPUT REVERSE(0|1). Compares compact128
with B1/T512 or B2/T256, using the same weights, physical owners and preallocated
profiles. One warmup and three complete cold-cache prefills per variant/group.
Timing excludes reset/claim, input construction, vocabulary head and hashing.
Different chunk boundaries can reassociate FP32 attention; state hashes are
retained, while final predictions must match. This is not serving throughput.
"""
import sys,os,json,time
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet,tensor
from xModel.mistral.text_7b.metal_decoder import MistralMetalDecoder
with open(sys.argv[3]) as f:fixture=json.load(f)
with open(fixture['model']+'/config.json') as f:config=json.load(f)
reverse=int(sys.argv[5]);assert reverse in (0,1) and not os.path.exists(sys.argv[4]);decoder=None;groups=[];closed=None;finished=False;prepared=None
prompt=next(c for c in fixture['cases'] if c['id']=='context-2048')['prompt_ids'][:2048]
def memory():return json.loads(garnet.memory_stats_json())
def f32(v,s):return tensor.tensor(v,shape=s,dtype=tensor.float32)
def save():
 with open(sys.argv[4],'w') as f:json.dump(dict(completed=finished,reverse=reverse,groups=groups,prepared=prepared,cleanup=closed,library=garnet.native_library_path(),scope=__doc__),f,indent=2)
def inputs(live,chunk):
 result=[]
 for offset in range(0,2048,chunk):
  hidden=decoder.io.stack_tokens([decoder.io.embedding([prompt[offset+t] if row in live else 0 for row in range(4)]) for t in range(chunk)])
  slots=[offset+t if row in live else 0 for row in range(4) for t in range(chunk)];active=[int(row in live) for row in range(4) for t in range(chunk)]
  result.append([hidden,f32([s for s in slots for _ in range(3)],[4,chunk,3]),f32(slots,[4,chunk]),f32(active,[4,chunk])])
 return result
def hashes():return [[decoder.io.verification_f32_sha256(p) for p in decoder.session.snapshot(i)] for i in range(32)]
try:
 decoder=MistralMetalDecoder(fixture['model']+'/model.safetensors',config,4,4096,resident=True);decoder.prepare_compact_prefill(512);prepared=memory()
 assert decoder.session.compact_prefill_family_count==17
 # Retain a deduplicated native resource inventory before the first dispatch.
 if not reverse:
  descriptors=[dict(tag='decode-'+str(i),object=l['graph']) for i,l in enumerate(decoder.layers)]
  descriptors += [dict(tag='base-'+str(i),object=g) for i,g in decoder.prefill_graphs.items()]
  descriptors += [dict(tag='compact-'+str(i),object=g) for i,g in decoder.compact_prefill_graphs.items()]
  descriptors.append(dict(tag='head',object=decoder.head))
  with open(sys.argv[4]+'.inventory.json','x') as f:f.write(garnet.memory_inventory_json(descriptors,decoder.session))
 for live,wide in [([0],512),([1,2],256)]:
  bundles={128:inputs(live,128),wide:inputs(live,wide)};group=dict(live=live,wide_chunk=wide,variants=[]);groups.append(group)
  for chunk in ([128,wide] if not reverse else [wide,128]):
   variant=dict(chunk=chunk,measured=[]);group['variants'].append(variant);steady=None;expected_state=None;expected_prediction=None
   for repeat in range(-1,3):
    decoder.session.reset();epochs=[]
    for row in range(4):
     lease=decoder.session.claim_request_row(row+1);assert lease[0]==row;epochs.append(lease[1])
    gpu_ms=0;start=time.monotonic()
    for args in bundles[chunk]:
     decoder.session.run_leased(decoder.prefill_profile_graphs(chunk),args[0],args[1],args[2],args[3],-1,epochs)
     assert decoder.session.compact_prefill_last_rows==len(live) and decoder.session.compact_prefill_last_first==live[0]
     gpu_ms+=decoder.session.gpu_ms
    wall=time.monotonic()-start;prediction=decoder.session.predict_rows([epochs[r] if r in live else 0 for r in range(4)])
    state=hashes();sample=memory();assert sample['metal_allocated_bytes']<=sample['metal_recommended_working_set_bytes']-256*1024*1024
    if repeat==-1:steady=sample['metal_allocated_bytes'];expected_state=state;expected_prediction=prediction
    else:
     assert sample['metal_allocated_bytes']==steady and state==expected_state and prediction==expected_prediction
     variant['measured'].append(dict(repeat=repeat,body_wall_seconds=wall,body_gpu_ms=gpu_ms,memory=sample))
    variant.update(state_sha256=state,predictions=prediction);save();print('wide_stage',live,chunk,repeat,wall,flush=True)
  assert group['variants'][0]['predictions']==group['variants'][1]['predictions']
  bundles.clear()
 finished=True
finally:
 if decoder is not None:decoder.close()
 garnet.shutdown();closed=memory()
 for k in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):assert closed[k]==0
 assert closed['metal_allocated_bytes']<=4*1024*1024
 save()
assert finished
