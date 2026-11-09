# SPDX-License-Identifier: Apache-2.0
"""Native matched2K target with a fixed bounded helper context.
Args LIB SOURCE INPUT OUTPUT qualify|ordinary|bounded.
The full target prompt remains unchanged; the helper sees the declared user tail.
All output controls are opened only after complete native generation.
Native stages only: no HTTP, quality or comparator speed admission.
"""
import sys,json,os,time,math
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet,tensor
from xModel.mistral.text_7b.metal_decoder import MistralMetalDecoder
from xModel.qwen.metal_dense.decoder import DenseMetalDecoder
with open(sys.argv[3]) as stream:inputs=json.load(stream)
assert len(sys.argv)==6
experiment=sys.argv[5];assert experiment=='qualify'
occupancy='occupancy-control'
mode_names=['ordinary','base'] if experiment=='qualify' else ['ordinary'] if experiment=='ordinary' else ['base']
output=sys.argv[4];assert not os.path.exists(output)
fused_enabled='native_fused_q4_decode_rows4_default_v1' in garnet.backend_info()
assert garnet.native_library_path()==sys.argv[1]+'/libgarnet.dylib'
assert 'mistral_verification_value_vector4' in garnet.backend_info() and 'prediction_confidence_top2_v1' in garnet.backend_info()
if occupancy:assert ('mistral_decode_reuse_three_rows' in garnet.backend_info())==(occupancy=='occupancy-candidate')
assert inputs['capacity']==4096 and inputs['output_limit']==128 and inputs['draft_token_limit']==8
assert inputs['draft_context_policy']['name']=='fixed_user_tail192_v1' and inputs['draft_capacity']==512
assert not fused_enabled
split_enabled='native_mistral_verification_split_scores128_default_v1' in garnet.backend_info()
with open(inputs['target_model']+'/config.json') as stream:config=json.load(stream)
with open(inputs['draft_model']+'/config.json') as stream:dconfig=json.load(stream)
target_tokenizer=garnet.tokenizer();target_tokenizer.open(inputs['target_model'])
draft_tokenizer=garnet.tokenizer();draft_tokenizer.open(inputs['draft_model'])
decoder=None;draft=None;policy=None;records=[];prepared=None;cleanup=None;request=0;draft_lease=None;use_confidence=False;stop_threshold=None;current_confidence=None

def val(values,shape):return tensor.tensor(values,shape=shape,dtype=tensor.float32)
def save(completed):
 with open(output,'w') as stream:
  json.dump(dict(completed=completed,inputs=inputs,backend=garnet.backend_info(),library=garnet.native_library_path(),
   prepared=prepared,records=records,cleanup=cleanup,scope=__doc__,projection_occupancy=occupancy,fused_decode_enabled=fused_enabled,experiment=experiment,split_scores_enabled=split_enabled),stream,indent=2)
def state(owner,io,layers):return [[io.verification_f32_sha256(part) for part in owner.snapshot(i)] for i in range(layers)]
def confidence_ids(values):
 global current_confidence
 assert len(values)==1 and len(values[0])==4
 current_confidence=values[0]
 assert type(current_confidence[0]) is int and all(math.isfinite(x) for x in current_confidence[1:])
 assert current_confidence[3]==current_confidence[1]-current_confidence[2] and current_confidence[3]>=0
 return [current_confidence[0]]
def draft_step(epoch,token,slot):
 args=[draft_base,draft.io.embedding([token]),val([float(slot)]*3,[1,3]),val([float(slot)],[1]),val([1.],[1]),0,[epoch],[token]]
 return confidence_ids(draft.session.run_tokens_confidence(*args)) if use_confidence else draft.session.run_tokens(*args)
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
   last=draft_step(epoch,ids[offset],offset)
   offset+=1;plans.append(1);continue
  tokens=ids[offset:offset+length]+[0]*(chunk-length)
  hidden=draft.io.stack_tokens([draft.io.embedding([token]) for token in tokens])
  last=draft.session.run_tokens(draft._resident_prefill_graphs(chunk),hidden,
   val([float(offset+t) if t<length else 0. for t in range(chunk) for axis in range(3)],[1,chunk,3]),
   val([float(offset+t) if t<length else 0. for t in range(chunk)],[1,chunk]),
   val([float(t<length) for t in range(chunk)],[1,chunk]),length-1 if offset+length==len(ids) and not use_confidence else -1,[epoch],tokens)
  if use_confidence and offset+length==len(ids):last=confidence_ids(draft.session.predict_rows_confidence([epoch]))
  offset+=length;plans.append(chunk)
 if last is None:last=confidence_ids(draft.session.predict_rows_confidence([epoch])) if use_confidence else draft.session.predict_rows([epoch])
 return last,plans

def propose(case,generated,limit,policy):
 global request,draft_lease,current_confidence
 current_confidence=None
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
 decode_started=time.monotonic();draft_ids=[];executed=[];proposal=[];translated=[];aligned=False;decode_samples=[];confidences=[];stop_reason='draft_budget'
 for i in range(inputs['draft_token_limit']):
  if predict[0] in inputs['draft_stops']:stop_reason='draft_eos';break
  if use_confidence:
   assert current_confidence[0]==predict[0];confidences.append(list(current_confidence))
  draft_ids.append(predict[0]);text=draft_tokenizer.decode_ex(draft_ids,False)
  translated=target_tokenizer.encode_ex(prefix+text,False);aligned=translated[:len(generated)]==generated
  if aligned:proposal=translated[len(generated):len(generated)+limit]
  else:proposal=[]
  if len(proposal)==limit:stop_reason='target_limit';break
  if stop_threshold is not None and proposal and current_confidence[3]<stop_threshold:stop_reason='margin';break
  if i+1<inputs['draft_token_limit']:
   executed.append(predict[0]);slot=len(prompt)+len(draft_ids)-1;step=time.monotonic()
   predict=draft_step(epoch,predict[0],slot)
   decode_samples.append(dict(wall_seconds=time.monotonic()-step,gpu_ms=draft.session.gpu_ms))
 if policy!='resident':assert draft.session.release_request_row(0,epoch)
 return proposal,dict(aligned=aligned,reason='proposal' if proposal else 'boundary',roundtrip=roundtrip,
  prefix_source='gpu_resident' if policy=='resident' else 'host_snapshot',reconcile=reconcile,executed_draft_ids=executed,
  prompt_ids=prompt,cached_tokens=cached,missing_tokens=len(prompt)-cached,prefill_chunks=plans,
  generated_ids=draft_ids,draft_text=draft_tokenizer.decode_ex(draft_ids,False),translated_ids=translated,
  terminal_prediction=predict[0],terminal_confidence=list(current_confidence) if use_confidence else None,confidences=confidences,stop_reason=stop_reason,threshold=stop_threshold,decode_samples=decode_samples,decode_seconds=time.monotonic()-decode_started,seconds=time.monotonic()-began,
  prefix_cache_stats=json.loads(draft.session.token_prefix_stats_json()))
try:
 began=time.monotonic()
 decoder=MistralMetalDecoder(inputs['target_model']+'/model.safetensors',config,1,inputs['capacity'])
 decoder.prepare_prefill(128);decoder.prepare_verification(4)
 if fused_enabled:assert [g.fused_decode_sites for g in decoder.verification_graphs]==[1]*32 and 1<=decoder.session.fused_decode_profiles<=2
 draft=DenseMetalDecoder(inputs['draft_model']+'/model.safetensors',dconfig,1,inputs.get('draft_capacity',inputs['capacity']))
 assert draft.io.checkpoint_format==inputs.get('draft_checkpoint_format','bf16')
 draft.enable_prefill(16);draft.prefill_compile_ms=0
 for width in (2,4,8,16):draft._resident_prefill_graphs(width)
 draft_base=[layer['graph'] for layer in draft.layers]
 draft.session.configure_token_prefix([inputs['draft_revision'],draft.io.checkpoint_format,inputs['draft_tokenizer_sha256'],
  'text_zero_based_3axis_v1','B1D'+str(inputs.get('draft_capacity',inputs['capacity']))+'T16','text'],2,512*1024*1024)
 assert draft.session.configure_prediction_confidence()
 draft.prefill_compile_ms=0
 target_graphs=len(decoder.prefill_graphs);draft_graphs=len(draft.prefill_graphs)
 prepared=dict(seconds=time.monotonic()-began,target_graphs=target_graphs,draft_graphs=draft_graphs,
  memory=json.loads(garnet.memory_stats_json()),fused_decode_profiles=decoder.session.fused_decode_profiles if fused_enabled else 0)
 assert prepared['memory']['metal_allocated_bytes']<=prepared['memory']['metal_recommended_working_set_bytes']-256*1024*1024
 print('cross_draft_ready',prepared['memory']['metal_allocated_bytes'],flush=True)

 assert 'native_scheduled_dense_verify_v1' in garnet.backend_info()
 assert decoder.session.configure_token_tracking(['d7182d886f93a9efa73d6d556ee33ffe839a4e09e15965eeef46f295ab5b503d',decoder.io.checkpoint_format,inputs['target_tokenizer_sha256'],'text_zero_based_3axis_v1','scheduled-B1-C4096-T128-verify4','text'])
 policy=garnet.continuous_scheduler();assert policy.initialize(decoder.session,16);assert policy.set_prefill_chunk(128)
 base=[layer['graph'] for layer in decoder.layers]
 def prefilling(case,budget):
  prompt=case['prompt_ids'];identity=policy.submit_tokens(prompt,budget,0);assert identity>0
  ticks=[]
  while True:
   tick=json.loads(policy.next_json());assert len(tick['batches'])==1;plan=tick['batches'][0];assert plan['stage']=='prefill' and len(plan['requests'])==1
   request=plan['requests'][0];assert request['id']==identity and request['row']==0 and request['cached_tokens']==0
   width=plan.get('prefill_chunk',128);offset=request['prefill_offset'];count=request['prefill_tokens'];assert 0<count<=width<=128
   ids=prompt[offset:offset+count]+[0]*(width-count);slots=[float(offset+t) if t<count else 0. for t in range(width)]
   graphs=decoder._resident_prefill_graphs(width)
   decoder.session.run_tokens(graphs,decoder.io.stack_tokens([decoder.io.embedding([token]) for token in ids]),val([p for p in slots for a in range(3)],[1,width,3]),val(slots,[1,width]),val([float(t<count) for t in range(width)],[1,width]),-1,[request['epoch']],ids)
   final=offset+count==len(prompt)
   ticks.append(dict(plan=plan,final=final,token_count=decoder.session.tracked_token_count(0,request['epoch'])))
   if final:
    known=decoder.session.predict_rows([request['epoch']])[0];assert policy.complete_prefill(identity,known,False)==2;return identity,request['epoch'],known,ticks
   assert policy.complete_prefill_chunk(identity,count)
 def scheduled_arguments(known,proposal,slot):
  submitted=[known]+proposal+[0]*(3-len(proposal));positions=[float(slot+t) for t in range(4)]
  return [decoder.io.stack_tokens([decoder.io.embedding([token]) for token in submitted]),val([p for p in positions for axis in range(3)],[1,4,3]),val(positions,[1,4]),val([float(t<len(proposal)+1) for t in range(4)],[1,4]),submitted,inputs['target_stops']]
 rejections=[]
 def rejected(name,args):
  before=policy.status_json();failed=False
  try:policy.verify_decode_json(*args)
  except Exception:failed=True
  assert failed and policy.status_json()==before
  rejections.append(name)
 case=inputs['cases'][0];prompt=case['prompt_ids'];assert target_tokenizer.encode(case['rendered'])==prompt
 began_request=time.monotonic();identity,epoch,known,prefill_ticks=prefilling(case,129);prefill_seconds=time.monotonic()-began_request
 generated=[known];rounds=[];slot=len(prompt);began_decode=time.monotonic()
 # One output-token headroom keeps this request leased for the complete final
 # KV/hidden witness. It is cancelled after the128-token capped audit. A separate
 # actual-budget2 request below proves terminal release; no performance claim.
 rejected('no_scheduled_tick',scheduled_arguments(known,[],slot))
 first=True
 while len(generated)<128 and known not in inputs['target_stops']:
  remaining=128-len(generated);tick=json.loads(policy.next_json());assert len(tick['batches'])==1
  plan=tick['batches'][0];assert plan['stage']=='decode' and len(plan['requests'])==1;request_plan=plan['requests'][0]
  assert request_plan['id']==identity and request_plan['row']==0 and request_plan['epoch']==epoch and request_plan['token_id']==known and request_plan['slot']==slot and request_plan['generated_tokens']==len(generated)
  proposal,trace=propose(case,generated,min(3,remaining-1),'resident') if remaining>1 else ([],dict(seconds=0.,reason='clipped_tail',aligned=False))
  args=scheduled_arguments(known,proposal,slot)
  if first:
   witness=(state(decoder.session,decoder.io,32),decoder.io.verification_f32_sha256(decoder.session.hidden()))
   bad=list(args);bad[4]=list(args[4]);bad[4][0]=(known+1)%32768;rejected('wrong_known',bad)
   bad=list(args);bad[2]=val([float(slot-1),float(slot+1),float(slot+2),float(slot+3)],[1,4]);rejected('wrong_slot',bad)
   bad=list(args);bad[3]=val([0.,1.,0.,0.],[1,4]);rejected('nonprefix_mask',bad)
   bad=list(args);bad[3]=val([0.,0.,0.,0.],[1,4]);rejected('no_active_scheduled_row',bad)
   bad=list(args);bad[3]=val([float('nan'),0.,0.,0.],[1,4]);rejected('nonfinite_mask',bad)
   bad=list(args);p=[float(slot+t) for t in range(4) for a in range(3)];p[2]+=1;bad[1]=val(p,[1,4,3]);rejected('wrong_position_axis',bad)
   bad=list(args);bad[4]=list(args[4]);bad[4][1]=-1;rejected('negative_token',bad)
   bad=list(args);bad[4]=list(args[4])[:3];rejected('wrong_token_count',bad)
   bad=list(args);bad[3]=val([1.,0.,0.,0.],[4]);rejected('wrong_mask_shape',bad)
   assert witness==(state(decoder.session,decoder.io,32),decoder.io.verification_f32_sha256(decoder.session.hidden()));first=False
  if remaining==1:
   # Native remaining budget is2 here, so depth3 must fail before execution.
   rejected('depth_exceeds_native_budget',scheduled_arguments(known,[0,0],slot))
  target_started=time.monotonic();result=json.loads(policy.verify_decode_json(*args));elapsed=time.monotonic()-target_started
  assert len(result['requests'])==1;q=result['requests'][0];metrics=result['verification'];emitted=q['emitted'];consumed=q['consumed']
  assert q['id']==identity and q['row']==0 and q['epoch']==epoch and q['known']==known and q['slot_before']==slot and q['remaining_before']==remaining+1 and q['submitted']==len(proposal)+1 and not q['finished']
  assert 1<=consumed==len(emitted)<=len(proposal)+1 and metrics['target_attention_split_score_dispatches']==32
  rounds.append(dict(committed_before=len(generated),slot=slot,proposal=proposal,submitted_ids=args[4],emitted_ids=emitted,consumed=consumed,accepted_draft=consumed-1,target_seconds=elapsed,draft=trace,metrics=metrics,native_schedule=plan,native_commit=q))
  generated.extend(emitted);known=emitted[-1];slot+=consumed
 elapsed=time.monotonic()-began_decode
 with open(case['control']) as stream:control=json.load(stream)
 assert generated==control['generated_ids'] and len(generated)==128 and slot==2175
 record=dict(case_id=case['id'],mode='base',prompt_ids=prompt,generated_ids=generated,slots=slot,state_sha256=state(decoder.session,decoder.io,32),hidden_sha256=decoder.io.verification_f32_sha256(decoder.session.hidden()),prefill_seconds=prefill_seconds,native_request_seconds=prefill_seconds+elapsed,decode_seconds=elapsed,decode_tokens_per_second=127/elapsed,rounds=rounds,memory=json.loads(garnet.memory_stats_json()),native_output_budget=129,prefill_ticks=prefill_ticks,rejections=rejections)
 assert decoder.session.tracked_token_count(0,epoch)==slot
 assert policy.cancel(identity);assert not decoder.session.request_row_matches(0,epoch)
 record['after_cancel']=json.loads(policy.status_json())
 # Actual budget2 terminal boundary, no helper or saved future output consulted.
 identity2,epoch2,known2,ticks2=prefilling(case,2);tick2=json.loads(policy.next_json());assert tick2['batches'][0]['stage']=='decode'
 rejected('actual_terminal_budget_overflow',scheduled_arguments(known2,[0],len(prompt)))
 terminal=json.loads(policy.verify_decode_json(*scheduled_arguments(known2,[],len(prompt))));q=terminal['requests'][0]
 assert q['id']==identity2 and q['epoch']==epoch2 and q['remaining_before']==1 and q['consumed']==1 and q['finished']
 assert q['emitted']==generated[1:2] and not decoder.session.request_row_matches(0,epoch2)
 assert not json.loads(policy.next_json())['batches']
 record['terminal_budget2']=dict(prefill_ticks=ticks2,commit=terminal,status=json.loads(policy.status_json()),epoch_released=True)
 records.append(record)
 if draft_lease is not None:assert draft.session.release_request_row(0,draft_lease[1]);draft_lease=None
 save(False)
finally:
 if policy is not None:policy.close()
 if draft is not None:draft.close()
 if decoder is not None:decoder.close()
 garnet.shutdown();cleanup=json.loads(garnet.memory_stats_json())
 for key in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):assert cleanup[key]==0
 assert cleanup['metal_allocated_bytes']<=4*1024*1024
 save(len(records)==1)
assert len(records)==1
