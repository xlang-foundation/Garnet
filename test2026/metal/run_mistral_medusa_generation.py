# SPDX-License-Identifier: Apache-2.0
"""Causal trained-head prototype vs ordinary greedy decode, sequentially.

Args: MODEL MEDUSA LIB SOURCE FIXTURE CODING_CONTROL CONTEXT_CONTROL NEW_OUTPUT.
Batch1, context4096, prepared up-to4 verifier. One run/mode/case, no warmups;
timings include host orchestration and trace capture. Different from frozen B4
HTTP serving profiles: this is correctness/acceptance and descriptive A/B only.
"""
import json,sys,os,time
sys.path.insert(0,sys.argv[3]);sys.path.insert(0,sys.argv[4])
import garnet,tensor
from xModel.mistral.text_7b.metal_decoder import MistralMetalDecoder
from xModel.mistral.text_7b.medusa_draft import MistralMedusaDraft
model,aux,output=sys.argv[1],sys.argv[2],sys.argv[8]
assert garnet.native_library_path()==sys.argv[3]+'/libgarnet.dylib'
assert not os.path.exists(output);os.mkdir(output)
with open(model+'/config.json') as stream:config=json.load(stream)
with open(sys.argv[5]) as stream:fixture=json.load(stream)
assert fixture['model']==model
controls=dict(coding=sys.argv[6],context=sys.argv[7]);decoder=None;draft=None;records=[];closed=None;prepared=None
def val(values,shape):return tensor.tensor(values,shape=shape,dtype=tensor.float32)
def state():return [[decoder.io.verification_f32_sha256(part) for part in decoder.session.snapshot(i)] for i in range(32)]
def save(completed):
 with open(output+'/status.json','w') as stream:
  json.dump(dict(completed=completed,records=records,prepared=prepared,cleanup=closed,fixture=sys.argv[5],
   profile=dict(batch=1,capacity=4096,prefill_chunk=128,verification_tokens=4,prefix_cache=False),
   library=garnet.native_library_path(),backend=garnet.backend_info(),scope=__doc__),stream,indent=2)
def prefill(ids):
 decoder.reset();offset=0;last=None
 while offset<len(ids):
  chunk=decoder._prefill_chunk_size(len(ids)-offset,128);valid=min(chunk,len(ids)-offset)
  hidden=decoder.io.stack_tokens([decoder.io.embedding([ids[offset+t] if t<valid else 0]) for t in range(chunk)])
  decoder.session.run(decoder._resident_prefill_graphs(chunk),hidden,
   val([float(offset+t) if t<valid else 0. for t in range(chunk) for axis in range(3)],[1,chunk,3]),
   val([float(offset+t) if t<valid else 0. for t in range(chunk)],[1,chunk]),
   val([float(t<valid) for t in range(chunk)],[1,chunk]),-1)
  raw=decoder.session.hidden().tolist();last=val(raw[(valid-1)*4096:valid*4096],[1,4096]);offset+=valid
 decoder.slots=[len(ids)]
 known=decoder.io.argmax(decoder.head.run([last]))[0]
 return known,last
try:
 decoder=MistralMetalDecoder(model+'/model.safetensors',config,1,4096,resident=True)
 decoder.prepare_prefill(128);decoder.prepare_verification(4)
 draft=MistralMedusaDraft(decoder,aux,3)
 prepared=json.loads(garnet.memory_stats_json())
 assert prepared['metal_allocated_bytes']<=prepared['metal_recommended_working_set_bytes']-256*1024*1024
 print('medusa_generation_ready',json.dumps(prepared),flush=True)
 for index,case in enumerate(fixture['cases']):
  with open(controls[case['kind']]+'/'+case['id']+'/request-0.result.json') as stream:control=json.load(stream)
  assert control['status']=='completed' and control['prompt_ids']==case['prompt_ids']
  pair=[]
  for mode in (['ordinary','medusa'] if index%2==0 else ['medusa','ordinary']):
   started=time.monotonic();known,last=prefill(case['prompt_ids']);prefill_seconds=time.monotonic()-started
   generated=[known];rounds=[];slot=len(case['prompt_ids']);began=time.monotonic()
   while known not in fixture['stops'] and len(generated)<case['max_tokens']:
    if mode=='ordinary':
     known=decoder.step([known],active=[1])[0];generated.append(known);slot+=1
    else:
     remaining=case['max_tokens']-len(generated);length=min(4,remaining)
     proposal=[]
     if length>1:
      normalized=draft.normalizer.run([last])
      proposal=[decoder.io.argmax(draft.heads[h].run([normalized]))[0] for h in range(length-1)]
     submitted=[known]+proposal+[0]*(4-length)
     hidden=decoder.io.stack_tokens([decoder.io.embedding([token]) for token in submitted])
     positions=[float(slot+t) for t in range(4)]
     result=decoder.session.verify_greedy(hidden,val([p for p in positions for axis in range(3)],[1,4,3]),
      val(positions,[1,4]),val([float(t<length) for t in range(4)],[1,4]),submitted,fixture['stops'],[remaining],None)[0]
     emitted=list(result[0]);consumed=result[1]
     assert consumed==len(emitted) and 1<=consumed<=length and result[2]==(emitted[-1] in fixture['stops'])
     stats=json.loads(decoder.session.verification_stats_json())
     if 'mistral_decode_reuse_four_rows' in garnet.backend_info():
      # The target head evaluates all four prepared rows; body projections
      # select only active token rows on a clipped tail.
      assert stats['target_affine_decode_block_dispatches']==(161 if length==4 else 1),('wrong reuse route',length,stats)
     for key,wanted in [('target_affine_dispatches',225),('target_affine_misses',0),('target_mpp_affine_dispatches',0),
      ('target_indexed_prefill_dispatches',0),('target_reuse64_prefill_dispatches',0),('target_kv_backups',64),
      ('target_attention_dispatches',32),('commit_kv_restores',64 if consumed<length else 0)]:assert stats[key]==wanted,(key,stats)
     rounds.append(dict(slot=slot,active_tokens=length,submitted_ids=submitted,emitted_ids=emitted,
      consumed_inputs=consumed,accepted_proposal_inputs=consumed-1,metrics=stats))
     generated.extend(emitted);known=emitted[-1];slot+=consumed;decoder.slots=[slot]
     last=decoder.session.hidden()
   decode_seconds=time.monotonic()-began
   assert generated==control['generated_ids'],('frozen native output changed',case['id'],mode)
   assert slot==len(case['prompt_ids'])+len(generated)-1 and decoder.slots==[slot]
   record=dict(id=case['id'],mode=mode,prompt_ids=case['prompt_ids'],budget=case['max_tokens'],
    generated_ids=generated,slots=slot,finish_reason='stop' if generated[-1] in fixture['stops'] else 'length',
    state_sha256=state(),hidden_sha256=decoder.io.verification_f32_sha256(decoder.session.hidden()),
    prefill_seconds=prefill_seconds,decode_seconds=decode_seconds,
    decode_tokens_per_second=(len(generated)-1)/decode_seconds,rounds=rounds,memory=json.loads(garnet.memory_stats_json()))
   records.append(record);pair.append(record);save(False)
   print('medusa_generation_case',case['id'],mode,len(generated),record['decode_tokens_per_second'],len(rounds),flush=True)
  assert pair[0]['state_sha256']==pair[1]['state_sha256'] and pair[0]['hidden_sha256']==pair[1]['hidden_sha256'],('ordinary/speculative final state differs',case['id'])
finally:
 if draft is not None:draft.close()
 if decoder is not None:decoder.close()
 garnet.shutdown();closed=json.loads(garnet.memory_stats_json())
 for key in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):assert closed[key]==0,(key,closed[key])
 save(len(records)==2*len(fixture['cases']))
 print('medusa_generation_closed',json.dumps(closed),flush=True)
assert len(records)==2*len(fixture['cases'])
