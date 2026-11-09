# SPDX-License-Identifier: Apache-2.0
"""Causal cross-tokenizer draft / native target verification experiment.

Args: LIB SOURCE INPUT OUTPUT. B1/C4096, up-to1024 output tokens, prepared profiles;
Three sequential modes compare ordinary target and the same GPU-resident
draft with scalar or ordered vector4 value sums in compiled-layout target attention, within one prepared engine. Complete coding outputs; native target
accepted-prefix verification. The resident lease release is inside decode timing. This is a complete-output native stage experiment, not frozen HTTP qualification.
The draft sees only the user messages and already-emitted target text. The
frozen future target stream is accessed only after each sequence completes.
"""
import sys,json,os,time
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet,tensor
from xModel.mistral.text_7b.metal_decoder import MistralMetalDecoder
from xModel.qwen.metal_dense.decoder import DenseMetalDecoder
with open(sys.argv[3]) as stream:inputs=json.load(stream)
output=sys.argv[4];assert not os.path.exists(output)
assert garnet.native_library_path()==sys.argv[1]+'/libgarnet.dylib'
assert 'mistral_verification_value_vector4' in garnet.backend_info()
assert inputs['capacity']==4096 and inputs['output_limit']==1024 and inputs['draft_token_limit']==8
with open(inputs['target_model']+'/config.json') as stream:config=json.load(stream)
with open(inputs['draft_model']+'/config.json') as stream:dconfig=json.load(stream)
target_tokenizer=garnet.tokenizer();target_tokenizer.open(inputs['target_model'])
draft_tokenizer=garnet.tokenizer();draft_tokenizer.open(inputs['draft_model'])
decoder=None;draft=None;records=[];prepared=None;cleanup=None;request=0;draft_lease=None

def val(values,shape):return tensor.tensor(values,shape=shape,dtype=tensor.float32)
def save(completed):
 with open(output,'w') as stream:
  json.dump(dict(completed=completed,inputs=inputs,backend=garnet.backend_info(),library=garnet.native_library_path(),
   prepared=prepared,records=records,cleanup=cleanup,scope=__doc__),stream,indent=2)
def state(owner,io,layers):return [[io.verification_f32_sha256(part) for part in owner.snapshot(i)] for i in range(layers)]
def tracked_prefill(ids,start,epoch,policy):
 offset=start;last=None;plans=[]
 while offset<len(ids):
  remaining=len(ids)-offset
  chunk=16
  if policy in ('compact','resident'):
   chunk=1
   while chunk<remaining and chunk<16:chunk*=2
  length=min(chunk,remaining)
  if chunk==1:
   last=draft.session.run_tokens(draft_base,draft.io.embedding([ids[offset]]),
    val([float(offset)]*3,[1,3]),val([float(offset)],[1]),val([1.],[1]),0,[epoch],[ids[offset]])
   offset+=1;plans.append(1);continue
  tokens=ids[offset:offset+length]+[0]*(chunk-length)
  hidden=draft.io.stack_tokens([draft.io.embedding([token]) for token in tokens])
  last=draft.session.run_tokens(draft._resident_prefill_graphs(chunk),hidden,
   val([float(offset+t) if t<length else 0. for t in range(chunk) for axis in range(3)],[1,chunk,3]),
   val([float(offset+t) if t<length else 0. for t in range(chunk)],[1,chunk]),
   val([float(t<length) for t in range(chunk)],[1,chunk]),length-1 if offset+length==len(ids) else -1,[epoch],tokens)
  offset+=length;plans.append(chunk)
 if last is None:last=draft.session.predict_rows([epoch])
 return last,plans

def propose(case,generated,limit,policy):
 global request,draft_lease
 began=time.monotonic();prefix=target_tokenizer.decode_ex(generated,False)
 # Reject a noncanonical target prefix instead of silently changing token IDs.
 roundtrip=target_tokenizer.encode_ex(prefix,False)==generated
 if not roundtrip:return [],dict(aligned=False,reason='noncanonical_prefix',roundtrip=False,seconds=time.monotonic()-began)
 prompt=draft_tokenizer.encode_ex(case['draft_prompt']+prefix,False)
 if len(prompt)+inputs['draft_token_limit']>inputs.get('draft_capacity',inputs['capacity']):
  return [],dict(aligned=False,reason='draft_capacity',roundtrip=True,seconds=time.monotonic()-began)
 reconcile=None
 if policy=='resident':
  if draft_lease is None:
   request+=1;draft_lease=draft.session.claim_request_row(request);assert draft_lease[0]==0
  epoch=draft_lease[1];cached=draft.session.reconcile_dense_tokens(0,epoch,prompt)
  reconcile=json.loads(draft.session.dense_reconcile_stats_json())
  assert reconcile['retained_tokens']==cached and reconcile['host_snapshot_bytes']==0
 else:
  request+=1;lease=draft.session.claim_request_row(request);assert lease[0]==0;epoch=lease[1]
  cached=draft.session.restore_token_prefix(0,epoch,prompt)
 assert 0<=cached<=len(prompt)
 predict,plans=tracked_prefill(prompt,cached,epoch,policy)
 if policy!='resident':assert draft.session.cache_token_prefix(0,epoch)
 decode_started=time.monotonic();draft_ids=[];executed=[];proposal=[];translated=[];aligned=False;decode_samples=[]
 for i in range(inputs['draft_token_limit']):
  if predict[0] in inputs['draft_stops']:break
  draft_ids.append(predict[0]);text=draft_tokenizer.decode_ex(draft_ids,False)
  translated=target_tokenizer.encode_ex(prefix+text,False);aligned=translated[:len(generated)]==generated
  if aligned:proposal=translated[len(generated):len(generated)+limit]
  else:proposal=[]
  if len(proposal)==limit:break
  if i+1<inputs['draft_token_limit']:
   executed.append(predict[0]);slot=len(prompt)+len(draft_ids)-1;step=time.monotonic()
   predict=draft.session.run_tokens(draft_base,draft.io.embedding([predict[0]]),
    val([float(slot)]*3,[1,3]),val([float(slot)],[1]),val([1.],[1]),0,[epoch],[predict[0]])
   decode_samples.append(dict(wall_seconds=time.monotonic()-step,gpu_ms=draft.session.gpu_ms))
 if policy!='resident':assert draft.session.release_request_row(0,epoch)
 return proposal,dict(aligned=aligned,reason='proposal' if proposal else 'boundary',roundtrip=roundtrip,
  prefix_source='gpu_resident' if policy=='resident' else 'host_snapshot',reconcile=reconcile,executed_draft_ids=executed,
  prompt_ids=prompt,cached_tokens=cached,missing_tokens=len(prompt)-cached,prefill_chunks=plans,
  generated_ids=draft_ids,draft_text=draft_tokenizer.decode_ex(draft_ids,False),translated_ids=translated,
  decode_samples=decode_samples,decode_seconds=time.monotonic()-decode_started,seconds=time.monotonic()-began,
  prefix_cache_stats=json.loads(draft.session.token_prefix_stats_json()))
try:
 began=time.monotonic()
 decoder=MistralMetalDecoder(inputs['target_model']+'/model.safetensors',config,1,inputs['capacity'])
 decoder.prepare_prefill(128);decoder.prepare_verification(4)
 draft=DenseMetalDecoder(inputs['draft_model']+'/model.safetensors',dconfig,1,inputs.get('draft_capacity',inputs['capacity']))
 assert draft.io.checkpoint_format==inputs.get('draft_checkpoint_format','bf16')
 draft.enable_prefill(16);draft.prefill_compile_ms=0
 for width in (2,4,8,16):draft._resident_prefill_graphs(width)
 draft_base=[layer['graph'] for layer in draft.layers]
 draft.session.configure_token_prefix([inputs['draft_revision'],draft.io.checkpoint_format,inputs['draft_tokenizer_sha256'],
  'text_zero_based_3axis_v1','B1D'+str(inputs.get('draft_capacity',inputs['capacity']))+'T16','text'],2,512*1024*1024)
 target_graphs=len(decoder.prefill_graphs);draft_graphs=len(draft.prefill_graphs)
 prepared=dict(seconds=time.monotonic()-began,target_graphs=target_graphs,draft_graphs=draft_graphs,
  memory=json.loads(garnet.memory_stats_json()))
 assert prepared['memory']['metal_allocated_bytes']<=prepared['memory']['metal_recommended_working_set_bytes']-256*1024*1024
 print('cross_draft_ready',prepared['memory']['metal_allocated_bytes'],flush=True)
 for index,case in enumerate(inputs['cases']):
  pair=[]
  for mode in (['ordinary','control','layout'] if index%2==0 else ['layout','control','ordinary']):
   assert draft_lease is None
   decoder.reset();assert decoder.session.verification_set_mistral_layout(True);assert decoder.session.verification_set_value_vector4(mode=='layout')
   draft.session.clear_token_prefixes();started=time.monotonic()
   prompt=case['prompt_ids'];assert target_tokenizer.encode(case['rendered'])==prompt
   known=decoder.prefill_sequence([decoder.io.embedding([token]) for token in prompt],[[[i]*3 for i in range(len(prompt))]])[0]
   prefill_seconds=time.monotonic()-started;generated=[known];rounds=[];slot=len(prompt);began=time.monotonic()
   while known not in inputs['target_stops'] and len(generated)<inputs['output_limit']:
    remaining=inputs['output_limit']-len(generated)
    if mode=='ordinary':known=decoder.step([known])[0];generated.append(known);slot+=1
    else:
     proposal,trace=propose(case,generated,min(3,remaining-1),'resident') if remaining>1 else ([],dict(seconds=0.,reason='clipped_tail',aligned=False))
     assert len(proposal)<=min(3,remaining-1)
     target_started=time.monotonic();length=len(proposal)+1
     if proposal:
      submitted=[known]+proposal+[0]*(4-length);positions=[float(slot+t) for t in range(4)]
      result=decoder.session.verify_greedy(decoder.io.stack_tokens([decoder.io.embedding([token]) for token in submitted]),
       val([p for p in positions for axis in range(3)],[1,4,3]),val(positions,[1,4]),
       val([float(t<length) for t in range(4)],[1,4]),submitted,inputs['target_stops'],[remaining],None)[0]
      emitted=list(result[0]);consumed=result[1]
      assert consumed==len(emitted) and 1<=consumed<=length and result[2]==(emitted[-1] in inputs['target_stops'])
      metrics=json.loads(decoder.session.verification_stats_json())
      assert metrics['target_affine_misses']==0 and metrics['target_affine_dispatches']==225
      assert metrics['target_attention_layout_dispatches']==32
      assert metrics['target_attention_vector4_dispatches']==(32 if mode=='layout' else 0)
      assert metrics['target_mpp_affine_dispatches']==0 and metrics['target_kv_backups']==64
      assert metrics['commit_kv_restores']==(64 if consumed<length else 0)
     else:
      submitted=[known];emitted=[decoder.step([known])[0]];consumed=1;metrics=None
     elapsed=time.monotonic()-target_started
     rounds.append(dict(committed_before=len(generated),slot=slot,proposal=proposal,submitted_ids=submitted,
      emitted_ids=emitted,consumed=consumed,accepted_draft=consumed-1,target_seconds=elapsed,draft=trace,metrics=metrics))
     generated.extend(emitted);known=emitted[-1];slot+=consumed;decoder.slots=[slot]
   if draft_lease is not None:
    assert draft.session.release_request_row(0,draft_lease[1]);draft_lease=None
   elapsed=time.monotonic()-began
   # This control is never consulted by the proposer or the native verifier.
   with open(case['control']) as stream:control=json.load(stream)
   assert control['prompt_ids']==prompt and generated==control['generated_ids'][:inputs['output_limit']]
   assert slot==len(prompt)+len(generated)-1 and decoder.slots==[slot]
   assert len(decoder.prefill_graphs)==target_graphs and len(draft.prefill_graphs)==draft_graphs
   assert decoder.prefill_compile_ms==0
   record=dict(case_id=case['id'],mode=mode,prompt_ids=prompt,generated_ids=generated,slots=slot,
    state_sha256=state(decoder.session,decoder.io,32),hidden_sha256=decoder.io.verification_f32_sha256(decoder.session.hidden()),
    prefill_seconds=prefill_seconds,decode_seconds=elapsed,decode_tokens_per_second=(len(generated)-1)/elapsed,
    rounds=rounds,memory=json.loads(garnet.memory_stats_json()))
   records.append(record);pair.append(record);save(False)
   print('cross_draft_case',case['id'],mode,len(generated),round(record['decode_tokens_per_second'],3),
    'accepted',sum(r['accepted_draft'] for r in rounds),'calls',len(rounds),flush=True)
  assert all(row['state_sha256']==pair[0]['state_sha256'] and row['hidden_sha256']==pair[0]['hidden_sha256'] for row in pair)
finally:
 if draft is not None:draft.close()
 if decoder is not None:decoder.close()
 garnet.shutdown();cleanup=json.loads(garnet.memory_stats_json())
 for key in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):
  assert cleanup[key]==0,(key,cleanup[key])
 assert cleanup['metal_allocated_bytes']<=4*1024*1024
 save(len(records)==3*len(inputs['cases']))
assert len(records)==3*len(inputs['cases'])
