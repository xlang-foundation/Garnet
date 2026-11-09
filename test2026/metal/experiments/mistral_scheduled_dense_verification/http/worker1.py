"""Persistent diagnostic B1 native scheduled speculation HTTP worker."""
import sys,json,os,time,math
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet,tensor
from xModel.mistral.text_7b.metal_decoder import MistralMetalDecoder
from xModel.qwen.metal_dense.decoder import DenseMetalDecoder
with open(sys.argv[3]) as f:profile=json.load(f)
assert len(sys.argv)==4 and profile['active_rows']==1 and not profile['prefix_cache']
inputs=profile['runtime_inputs'];fused_enabled=False
with open(inputs['target_model']+'/config.json') as f:config=json.load(f)
with open(inputs['draft_model']+'/config.json') as f:dconfig=json.load(f)
target_tokenizer=garnet.tokenizer();target_tokenizer.open(inputs['target_model'])
draft_tokenizer=garnet.tokenizer();draft_tokenizer.open(inputs['draft_model'])
decoder=None;draft=None;policy=None;request=0;draft_lease=None;use_confidence=False;stop_threshold=None;current_confidence=None

def val(values,shape):return tensor.tensor(values,shape=shape,dtype=tensor.float32)
def emit(d):print(json.dumps(d,separators=(',',':')),flush=True)
def messages():
 pending=b''
 while True:
  block=os.read(0,65536)
  if not block:
   assert not pending
   return
  pending+=block
  while b'\n' in pending:
   line,pending=pending.split(b'\n',1)
   if len(line)>131072:raise ValueError('Worker request frame too large')
   yield json.loads(line)
  if len(pending)>131072:raise ValueError('Worker request frame too large')
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
   ticks.append(dict(plan=plan,final=final,token_count=decoder.session.diagnostic_tracked_token_count(0,request['epoch'])))
   if final:
    known=decoder.session.predict_rows([request['epoch']])[0];assert policy.complete_prefill(identity,known,known in inputs['target_stops'])==(3 if known in inputs['target_stops'] else 2);return identity,request['epoch'],known,ticks
   assert policy.complete_prefill_chunk(identity,count)
 def scheduled_arguments(known,proposal,slot):
  submitted=[known]+proposal+[0]*(3-len(proposal));positions=[float(slot+t) for t in range(4)]
  return [decoder.io.stack_tokens([decoder.io.embedding([token]) for token in submitted]),val([p for p in positions for axis in range(3)],[1,4,3]),val(positions,[1,4]),val([float(t<len(proposal)+1) for t in range(4)],[1,4]),submitted,inputs['target_stops']]

 emit(dict(event='ready',backend=garnet.backend_info(),profile=profile,active_requests=1,capacity=4096,prefill_chunk=128,prefix_cache=False,multimodal=False,target_graphs=target_graphs,draft_graphs=draft_graphs,memory=json.loads(garnet.memory_stats_json())))
 for message in messages():
  key=message['id'];body=message['body'];limit=body['max_tokens'];rendered=body['rendered_prompt'];user=body['draft_user_content']
  if type(limit) is not int or not 2<=limit<=128 or not isinstance(rendered,str) or not isinstance(user,str) or len(rendered)>65536:
   emit(dict(event='error',id=key,error='Request exceeds diagnostic profile',fatal=False));continue
  ids=list(target_tokenizer.encode(rendered))
  if not ids or len(ids)+limit>4096:
   emit(dict(event='error',id=key,error='Context exceeds diagnostic profile',fatal=False));continue
  tail=draft_tokenizer.decode_ex(draft_tokenizer.encode_ex(user,False)[-192:],False)
  case=dict(prompt_ids=ids,draft_prompt=profile['helper_chat_prefix']+tail+profile['helper_chat_suffix'])
  emit(dict(event='native_admit',id=key,prompt_ids=ids,draft_prompt=case['draft_prompt'],limit=limit))
  identity,epoch,known,ticks=prefilling(case,limit);generated=[];text_ids=[];target_calls=0;accepted=0;proposed=0
  def deliver(token):
   generated.append(token)
   if token not in inputs['target_stops']:text_ids.append(token)
   text=target_tokenizer.decode_ex(text_ids,True).encode('utf8').decode('utf8','replace')
   emit(dict(event='token',id=key,token_id=token,sampled_tokens=len(generated),prompt_tokens=len(ids),cached_tokens=0,text=text))
   return text
  text=deliver(known);slot=len(ids);done=known in inputs['target_stops']
  while not done:
   tick=json.loads(policy.next_json());assert len(tick['batches'])==1;plan=tick['batches'][0];req=plan['requests'][0]
   assert plan['stage']=='decode' and len(plan['requests'])==1 and req['id']==identity and req['epoch']==epoch and req['token_id']==known and req['slot']==slot
   remaining=limit-len(generated);proposal,trace=propose(case,generated,min(3,remaining-1),'resident') if remaining>1 else ([],dict(seconds=0.,reason='clipped_tail',aligned=False))
   args=scheduled_arguments(known,proposal,slot);began=time.monotonic();native=json.loads(policy.verify_decode_json(*args));elapsed=time.monotonic()-began;q=native['requests'][0]
   assert len(native['requests'])==1 and q['id']==identity and q['epoch']==epoch and q['remaining_before']==remaining and q['known']==known
   emit(dict(event='native_dense_verification',id=key,committed_before=len(generated),slot=slot,proposal=proposal,submitted_ids=args[4],draft=trace,native_schedule=plan,native=native,target_seconds=elapsed))
   target_calls+=1;accepted+=q['consumed']-1;proposed+=len(proposal)
   for token in q['emitted']:text=deliver(token)
   known=generated[-1];slot+=q['consumed'];done=q['finished']
  if draft_lease is not None:assert draft.session.release_request_row(0,draft_lease[1]);draft_lease=None
  assert len(generated)<=limit and not decoder.session.release_request_row(0,epoch)
  emit(dict(event='completed',id=key,generated_ids=generated,prompt_ids=ids,text=text,cached_tokens=0,finish_reason='stop' if known in inputs['target_stops'] else 'length',generation_mode='native_scheduled_dense_speculation_B1',profile_id=profile['profile_id'],native_request=identity,speculative_metrics=dict(target_calls=target_calls,accepted=accepted,proposed=proposed),native_status=json.loads(policy.status_json()),prefill_ticks=ticks,memory=json.loads(garnet.memory_stats_json())))
finally:
 if policy is not None:policy.close()
 if draft is not None:draft.close()
 if decoder is not None:decoder.close()
 garnet.shutdown();memory=json.loads(garnet.memory_stats_json())
 assert memory['metal_allocated_bytes']<=4194304
 for k in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):assert memory[k]==0
 emit(dict(event='closed',memory=memory))
