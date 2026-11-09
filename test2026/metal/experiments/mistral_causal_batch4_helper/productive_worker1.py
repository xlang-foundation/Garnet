"""Untimed native C4 scheduler with real batched helper drafts and exact target verification."""
import sys,json,os,time
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet,tensor
from xModel.mistral.text_7b.metal_decoder import MistralMetalDecoder
from xModel.qwen.metal_dense.decoder import DenseMetalDecoder
with open(sys.argv[3]) as f:inputs=json.load(f)
output=sys.argv[4];mode=sys.argv[5];assert mode in ('ordinary','productive') and not os.path.exists(output)
def val(a,shape):return tensor.tensor(a,shape=shape,dtype=tensor.float32)
decoder=None;draft=None;policy=None;jobs={};all_jobs={};ticks=[];snapshots=[];prepared=None;cleanup=None;helper_leases={};helper_calls=[];helper_rounds=[]
target_tokenizer=garnet.tokenizer();target_tokenizer.open(inputs['target_model'])
draft_tokenizer=garnet.tokenizer();draft_tokenizer.open(inputs['draft_model'])
def save(done):
 with open(output,'w') as f:json.dump(dict(completed=done,mode=mode,inputs=inputs,backend=garnet.backend_info(),library=garnet.native_library_path(),prepared=prepared,jobs=list(all_jobs.values()),ticks=ticks,snapshots=snapshots,helper_calls=helper_calls,helper_rounds=helper_rounds,cleanup=cleanup,scope=__doc__),f,indent=2)
def capture(stage):
 states=[]
 for layer in range(32):
  pair=decoder.session.snapshot(layer);states.append([[decoder.io.verification_f32_sha256(decoder.io.rows(part,row,1)) for part in pair] for row in range(4)])
 bank=decoder.session.diagnostic_final_hidden();hidden=[decoder.io.verification_f32_sha256(decoder.io.rows(bank,row,1)) for row in range(4)]
 owners=[dict(id=j['native_request'],label=j['label'],row=j['row'],epoch=j['epoch'],generated_ids=list(j['generated']),prefilled=j['prefilled'],finished=j['finished']) for j in all_jobs.values() if 'row' in j]
 snapshots.append(dict(stage=stage,states=states,hidden=hidden,owners=owners,status=json.loads(policy.status_json())))
def helper_state():
 states=[]
 for layer in range(24):
  pair=draft.session.snapshot(layer);states.append([[draft.io.verification_f32_sha256(draft.io.rows(part,row,1)) for part in pair] for row in range(4)])
 return states
def helper_epochs(labels):
 epochs=[0]*4
 for label in labels:epochs[helper_leases[label][0]]=helper_leases[label][1]
 return epochs
def propose(requests):
 traces={};queries=[];before=helper_state()
 for q in requests:
  job=jobs[q['id']];label=job['label'];generated=list(job['generated']);budget=job['limit']-len(generated);limit=min(3,budget-1)
  trace=dict(label=label,target_request=q['id'],target_row=q['row'],target_epoch=q['epoch'],generated_before=generated,remaining=budget,target_proposal_limit=limit,helper_row=helper_leases[label][0],helper_epoch=helper_leases[label][1],draft_ids=[],executed=[],proposal=[],translated_ids=[],prompt_ids=None,reconcile=None,reason=None)
  traces[label]=trace
  if limit==0:trace['reason']='clipped_tail';continue
  prefix=target_tokenizer.decode_ex(generated,False)
  if target_tokenizer.encode_ex(prefix,False)!=generated:trace['reason']='noncanonical_prefix';continue
  prompt=draft_tokenizer.encode_ex(job['draft_prompt']+prefix,False)
  if len(prompt)+8>512:trace['reason']='draft_capacity';continue
  row,epoch=helper_leases[label];kept=draft.session.reconcile_dense_tokens(row,epoch,prompt);stats=json.loads(draft.session.dense_reconcile_stats_json())
  assert stats['retained_tokens']==kept and stats['host_snapshot_bytes']==0
  trace.update(prompt_ids=prompt,reconcile=stats,prefix=prefix,reason='draft_budget');queries.append(dict(label=label,trace=trace,prompt=prompt,cursor=kept))
 after_reconcile=helper_state();call_first=len(helper_calls)
 while any(q['cursor']<len(q['prompt']) for q in queries):
  byrow={helper_leases[q['label']][0]:q for q in queries};ids=[];slots=[];active=[];selected=[];lengths={}
  for row in range(4):
   q=byrow.get(row);length=min(16,len(q['prompt'])-q['cursor']) if q else 0
   if length:selected.append(q['label']);lengths[q['label']]=length
   for t in range(16):
    on=t<length;position=q['cursor']+t if on else 0
    ids.append(q['prompt'][position] if on else 0);slots.append(float(position));active.append(float(on))
  ep=helper_epochs(selected);hidden=draft.io.stack_tokens([draft.io.embedding([ids[row*16+t] for row in range(4)]) for t in range(16)])
  draft.session.run_tokens(draft_graphs,hidden,val([s for s in slots for a in range(3)],[4,16,3]),val(slots,[4,16]),val(active,[4,16]),-1,ep,ids)
  helper_calls.append(dict(stage='prefill',labels=selected,tokens=ids,slots=slots,active=active,epochs=ep))
  for q in queries:q['cursor']+=lengths.get(q['label'],0)
 predicted=draft.session.predict_rows(helper_epochs([q['label'] for q in queries])) if queries else None
 pending={q['label']:q for q in queries}
 for step in range(8):
  advancing=[];ids=[0]*4;slots=[0.]*4;active=[0.]*4
  for label,q in list(pending.items()):
   trace=q['trace'];row=helper_leases[label][0];token=predicted[row]
   if token in inputs['draft_stops']:trace['reason']='draft_eos';del pending[label];continue
   trace['draft_ids'].append(token);text=draft_tokenizer.decode_ex(trace['draft_ids'],False);translated=target_tokenizer.encode_ex(trace['prefix']+text,False);trace['translated_ids']=translated
   trace['proposal']=translated[len(trace['generated_before']):len(trace['generated_before'])+trace['target_proposal_limit']] if translated[:len(trace['generated_before'])]==trace['generated_before'] else []
   if len(trace['proposal'])==trace['target_proposal_limit']:trace['reason']='target_limit';del pending[label];continue
   if step==7:del pending[label];continue
   ids[row]=token;slots[row]=float(len(q['prompt'])+len(trace['draft_ids'])-1);active[row]=1.;advancing.append(label);trace['executed'].append(token)
  if not advancing:break
  ep=helper_epochs(advancing);predicted=draft.session.run_tokens(draft_base,draft.io.embedding(ids),val([s for s in slots for a in range(3)],[4,3]),val(slots,[4]),val(active,[4]),0,ep,ids)
  helper_calls.append(dict(stage='decode',labels=advancing,tokens=ids,slots=slots,active=active,epochs=ep))
 after=helper_state();helper_rounds.append(dict(traces=list(traces.values()),before=before,after_reconcile=after_reconcile,after=after,call_first=call_first,call_last=len(helper_calls)))
 return {q['id']:traces[jobs[q['id']]['label']]['proposal'] for q in requests}
try:
 with open(inputs['target_model']+'/config.json') as f:config=json.load(f)
 decoder=MistralMetalDecoder(inputs['target_model']+'/model.safetensors',config,4,4096);decoder.prepare_prefill(128);decoder.prepare_compact_prefill(512);decoder.prepare_verification(4);decoder.prefill_compile_ms=0
 base=[l['graph'] for l in decoder.layers]
 assert decoder.session.configure_token_tracking(['d7182d886f93a9efa73d6d556ee33ffe839a4e09e15965eeef46f295ab5b503d',decoder.io.checkpoint_format,inputs['target_tokenizer_sha256'],'text_zero_based_3axis_v1','scheduled-B4-C4096-prefill-up512-verify4-fp32-v1','text'])
 policy=garnet.continuous_scheduler();assert policy.initialize(decoder.session,16) and policy.set_prefill_chunk(512)
 if mode=='productive':
  with open(inputs['draft_model']+'/config.json') as f:dconfig=json.load(f)
  draft=DenseMetalDecoder(inputs['draft_model']+'/model.safetensors',dconfig,4,512);draft.enable_prefill(16);draft.prefill_compile_ms=0;draft_graphs=draft._resident_prefill_graphs(16);draft.prefill_compile_ms=0;draft_base=[l['graph'] for l in draft.layers]
  assert draft.session.configure_token_tracking([inputs['draft_revision'],draft.io.checkpoint_format,inputs['draft_tokenizer_sha256'],'text_zero_based_3axis_v1','helper-B4-D512-T16-profile-v1','text'])
  for number,item in enumerate(inputs['jobs']):
   lease=draft.session.claim_request_row(number+1);assert lease[0]==number;helper_leases[item['label']]=lease
 prepared=dict(memory=json.loads(garnet.memory_stats_json()),target_graphs=len(decoder.prefill_graphs),helper_graphs=len(draft.prefill_graphs) if draft else 0)
 assert prepared['memory']['metal_allocated_bytes']<=prepared['memory']['metal_recommended_working_set_bytes']-536870912
 print('productive4_ready',mode,prepared['memory']['metal_allocated_bytes'],flush=True)
 for item in inputs['jobs']:
  assert target_tokenizer.encode(item['rendered'])==item['prompt_ids'];identity=policy.submit_tokens(item['prompt_ids'],item['limit'],0);assert identity>0
  job=dict(item,native_request=identity,generated=[],prefilled=0,finished=False);jobs[identity]=job;all_jobs[identity]=job
 while jobs:
  tick=json.loads(policy.next_json());assert tick['batches'];record=dict(plan=tick,executions=[]);ticks.append(record)
  for plan in tick['batches']:
   for q in plan['requests']:jobs[q['id']].update(row=q['row'],epoch=q['epoch'])
   if plan['stage']=='prefill':
    width=plan.get('prefill_chunk',128);byrow={q['row']:q for q in plan['requests']};ids=[];slots=[];active=[];final=[0]*4
    for row in range(4):
     q=byrow.get(row)
     if q is not None and q['prefill_offset']+q['prefill_tokens']==len(jobs[q['id']]['prompt_ids']):final[row]=q['epoch']
     for t in range(width):
      on=q is not None and t<q['prefill_tokens'];position=q['prefill_offset']+t if on else 0
      ids.append(jobs[q['id']]['prompt_ids'][position] if on else 0);slots.append(float(position));active.append(float(on))
    hidden=decoder.io.stack_tokens([decoder.io.embedding([ids[row*width+t] for row in range(4)]) for t in range(width)])
    decoder.session.run_tokens(decoder.prefill_profile_graphs(width),hidden,val([s for s in slots for a in range(3)],[4,width,3]),val(slots,[4,width]),val(active,[4,width]),-1,plan['epochs'],ids)
    predicted=decoder.session.predict_rows(final) if any(final) else None;completed=[]
    for q in plan['requests']:
     job=jobs[q['id']];job['prefilled']=q['prefill_offset']+q['prefill_tokens']
     if final[q['row']]:
      token=predicted[q['row']];job['generated'].append(token);job['finished']=token in inputs['stops'] or len(job['generated'])==job['limit'];assert policy.complete_prefill(q['id'],token,job['finished'])==(3 if job['finished'] else 2);completed.append(dict(id=q['id'],token=token,finished=job['finished']))
      if job['finished']:del jobs[q['id']]
     else:assert policy.complete_prefill_chunk(q['id'],q['prefill_tokens'])
    record['executions'].append(dict(stage='prefill',completed=completed));capture('prefill')
   else:
    assert plan['stage']=='decode'
    if mode=='ordinary':
     ids=[0]*4;slots=[0.]*4
     for q in plan['requests']:ids[q['row']]=q['token_id'];slots[q['row']]=float(q['slot']);assert ids[q['row']]==jobs[q['id']]['generated'][-1]
     predicted=decoder.session.run_tokens(base,decoder.io.embedding(ids),val([s for s in slots for a in range(3)],[4,3]),val(slots,[4]),val([float(x) for x in plan['active_rows']],[4]),0,plan['epochs'],ids);completed=[]
     for q in plan['requests']:
      job=jobs[q['id']];token=predicted[q['row']];job['generated'].append(token);job['finished']=token in inputs['stops'] or len(job['generated'])==job['limit'];assert policy.complete_decode(q['id'],token,job['finished'])==(3 if job['finished'] else 2);completed.append(dict(id=q['id'],row=q['row'],epoch=q['epoch'],emitted=[token],finished=job['finished']))
     record['executions'].append(dict(stage='decode',completed=completed,native=None))
    else:
     proposals=propose(plan['requests']);byrow={q['row']:q for q in plan['requests']};ids=[];slots=[];active=[]
     for row in range(4):
      q=byrow.get(row);tokens=[q['token_id']]+proposals[q['id']] if q else []
      for t in range(4):
       on=t<len(tokens);ids.append(tokens[t] if on else 0);slots.append(float(q['slot']+t) if on else 0.);active.append(float(on))
     native=json.loads(policy.verify_decode_json(decoder.io.stack_tokens([decoder.io.embedding([ids[row*4+t] for row in range(4)]) for t in range(4)]),val([s for s in slots for a in range(3)],[4,4,3]),val(slots,[4,4]),val(active,[4,4]),ids,inputs['stops']));completed=native['requests']
     for q in completed:
      job=jobs[q['id']];assert q['row']==job['row'] and q['epoch']==job['epoch'] and q['remaining_before']==job['limit']-len(job['generated']);job['generated'].extend(q['emitted']);job['finished']=q['finished'];assert job['finished']==(q['eos'] or len(job['generated'])==job['limit'])
     record['executions'].append(dict(stage='decode',completed=completed,native=native,submitted_ids=ids,submitted_slots=slots,submitted_active=active,proposals=proposals))
    capture('decode')
    for q in completed:
     if q['finished']:
      job=jobs.pop(q['id'])
      if draft:
       lease=helper_leases.pop(job['label']);assert draft.session.release_request_row(lease[0],lease[1])
    print('productive4_decode',mode,[len(j['generated']) for j in all_jobs.values()],flush=True)
  save(False);assert len(ticks)<100
 assert decoder.prefill_compile_ms==0 and (not draft or draft.prefill_compile_ms==0)
 assert json.loads(policy.status_json())['active']==json.loads(policy.status_json())['tracked']==0 and not helper_leases
finally:
 if policy is not None:policy.close()
 if draft is not None:draft.close()
 if decoder is not None:decoder.close()
 garnet.shutdown();cleanup=json.loads(garnet.memory_stats_json())
 assert all(cleanup[k]==0 for k in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets')) and cleanup['metal_allocated_bytes']<=4194304
 save(len(all_jobs)==4 and all(j['finished'] for j in all_jobs.values()))
