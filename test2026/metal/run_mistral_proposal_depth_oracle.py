# SPDX-License-Identifier: Apache-2.0
"""Causal proposal-depth feasibility only; the target is a frozen token oracle.

Args LIB SOURCE INPUT OUTPUT. Native Qwen0.6 proposes from user messages and
already-committed Mistral text. A separate frozen oracle supplies acceptance and
next target tokens; no native target verifier is run. Recorded draft timings
exclude target cache eviction and are NOT full-generation performance evidence.
"""
import sys,json,os,time
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet,tensor
from xModel.qwen.metal_dense.decoder import DenseMetalDecoder
with open(sys.argv[3]) as stream:inputs=json.load(stream)
output=sys.argv[4];assert not os.path.exists(output)
assert garnet.native_library_path()==sys.argv[1]+'/libgarnet.dylib'
target_tokenizer=garnet.tokenizer();target_tokenizer.open(inputs['target_model'])
draft_tokenizer=garnet.tokenizer();draft_tokenizer.open(inputs['draft_model'])
with open(inputs['draft_model']+'/config.json') as stream:dconfig=json.load(stream)
draft=None;draft_lease=None;request=0;records=[];prepared=None;cleanup=None
def val(values,shape):return tensor.tensor(values,shape=shape,dtype=tensor.float32)
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
def oracle_commit(oracle,cursor,proposal,remaining):
 accepted=0
 for token in proposal:
  if cursor+accepted>=len(oracle) or oracle[cursor+accepted]!=token:break
  accepted+=1
  if token in inputs['target_stops']:break
 consumed=min(accepted+1,remaining,len(oracle)-cursor)
 emitted=oracle[cursor:cursor+consumed]
 for i,token in enumerate(emitted):
  if token in inputs['target_stops']:emitted=emitted[:i+1];break
 return emitted

def save(completed):
 with open(output,'w') as stream:json.dump(dict(completed=completed,inputs=inputs,records=records,prepared=prepared,cleanup=cleanup,
  library=garnet.native_library_path(),backend=garnet.backend_info(),scope=__doc__),stream,indent=2)
try:
 draft=DenseMetalDecoder(inputs['draft_model']+'/model.safetensors',dconfig,1,inputs['draft_capacity'])
 draft.enable_prefill(16);draft.prefill_compile_ms=0
 for width in (2,4,8,16):draft._resident_prefill_graphs(width)
 prepared_compile_ms=draft.prefill_compile_ms;draft.prefill_compile_ms=0
 draft_base=[layer['graph'] for layer in draft.layers];graph_count=len(draft.prefill_graphs)
 draft.session.configure_token_prefix([inputs['draft_revision'],draft.io.checkpoint_format,inputs['draft_tokenizer_sha256'],
  'text_zero_based_3axis_v1','B1D512T16oracle-feasibility','text'],2,512*1024*1024)
 prepared=dict(graphs=graph_count,prefill_compile_ms=prepared_compile_ms,memory=json.loads(garnet.memory_stats_json()))
 for index,case in enumerate(inputs['cases']):
  for depth in ([3,7] if index%2==0 else [7,3]):
   inputs['draft_token_limit']=8 if depth==3 else 16
   draft.session.clear_token_prefixes();assert draft_lease is None
   with open(case['control']) as stream:raw=json.load(stream)
   oracle=raw['generated_ids'];assert raw['prompt_ids']==case['prompt_ids']
   generated=[oracle[0]];rounds=[];began=time.monotonic()
   while generated[-1] not in inputs['target_stops'] and len(generated)<inputs['output_limit']:
    cursor=len(generated);remaining=inputs['output_limit']-cursor
    proposal,trace=propose(case,generated,min(depth,remaining-1),'resident') if remaining>1 else ([],dict(seconds=0.,reason='clipped_tail',aligned=False))
    emitted=oracle_commit(oracle,cursor,proposal,remaining);assert emitted
    rounds.append(dict(committed_before=cursor,proposal=proposal,emitted_ids=emitted,consumed=len(emitted),accepted_draft=len(emitted)-1,draft=trace))
    generated.extend(emitted)
   if draft_lease is not None:assert draft.session.release_request_row(0,draft_lease[1]);draft_lease=None
   assert generated==oracle,'oracle token mismatch'
   assert len(draft.prefill_graphs)==graph_count,'new graph after readiness'
   assert draft.prefill_compile_ms==0,'compilation after readiness'
   records.append(dict(case_id=case['id'],depth=depth,draft_budget=inputs['draft_token_limit'],generated_ids=generated,
    draft_only_seconds=time.monotonic()-began,rounds=rounds,memory=json.loads(garnet.memory_stats_json())))
   save(False);print('oracle_depth',case['id'],depth,'target_calls',len(rounds),'accepted',sum(t['accepted_draft'] for t in rounds),'draft_steps',sum(len(t['draft'].get('executed_draft_ids',[])) for t in rounds),flush=True)
finally:
 if draft is not None:draft.close()
 garnet.shutdown();cleanup=json.loads(garnet.memory_stats_json())
 for key in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):assert cleanup[key]==0
 assert cleanup['metal_allocated_bytes']<=4*1024*1024
 save(len(records)==16)
assert len(records)==16
