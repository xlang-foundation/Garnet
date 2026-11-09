"""Untimed native prefix reuse; zero proposals and empty stops are correctness fixtures."""
import sys,json,os
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet,tensor
from xModel.mistral.text_7b.metal_decoder import MistralMetalDecoder
with open(sys.argv[3]) as f:inputs=json.load(f)
output=sys.argv[4];mode=sys.argv[5];assert mode in ('ordinary-off','ordinary-cache','verify-cache','reverse-cache') and not os.path.exists(output)
retained=mode!='ordinary-off';verify=mode in ('verify-cache','reverse-cache')
reference={}
if retained:
 with open(inputs['reference_path']) as f:control=json.load(f)
 assert control['completed'] and control['mode']=='ordinary-off'
 for frame in control['frames']:
  if frame['kind'] in ('prefill','before_terminal'):
   for row in frame['rows']:reference[frame['kind'],row['label']]=row
with open(inputs['target_model']+'/config.json') as f:config=json.load(f)
decoder=None;policy=None;jobs={};all_jobs={};frames=[];ticks=[];admissions=[];next_followup=0;pending=None;twins_admitted=False;prepared=None;cleanup=None;failure=None
def val(v,s):return tensor.tensor(v,shape=s,dtype=tensor.float32)
def stats():return json.loads(decoder.session.token_prefix_stats_json())
def save(done):
 with open(output,'w') as f:json.dump(dict(completed=done,mode=mode,scope=__doc__,inputs=inputs,library=garnet.native_library_path(),backend=garnet.backend_info(),prepared=prepared,jobs=list(all_jobs.values()),ticks=ticks,frames=frames,admissions=admissions,cache=stats() if decoder is not None and decoder.session is not None else last_cache,cleanup=cleanup,failure=failure),f,indent=2)
def snapshot(kind,owners):
 selected=[dict(label=j['label'],generated=len(j['generated']),row=j['row'],epoch=j['epoch'],states=[]) for j in owners]
 for layer in range(32):
  pair=decoder.session.snapshot(layer)
  for row in selected:row['states'].append([decoder.io.verification_f32_sha256(decoder.io.rows(p,row['row'],1)) for p in pair])
 hidden=decoder.session.diagnostic_final_hidden()
 for row in selected:row['hidden']=decoder.io.verification_f32_sha256(decoder.io.rows(hidden,row['row'],1))
 frame=dict(kind=kind,rows=selected);frames.append(frame);save(False);print('prefix_frame',mode,kind,[r['label'] for r in selected],flush=True)
 if retained and kind in ('prefill','before_terminal'):
  for row in selected:
   expected=reference[kind,row['label']]
   assert row['generated']==expected['generated']
   assert row['states']==expected['states'] and row['hidden']==expected['hidden'],'Retained prefix state differs from fresh '+kind+' '+row['label']
 return frame
def submit(item):
 identity=policy.submit_tokens(item['prompt_ids'],item['limit'],0);assert identity>0
 j=dict(item,id=identity,generated=[],prefilled=0,status='queued');jobs[identity]=j;all_jobs[identity]=j;return identity
last_cache=None
try:
 assert 'native_wide_verify_preparation_lifetime_v1' in garnet.backend_info()
 decoder=MistralMetalDecoder(inputs['target_model']+'/model.safetensors',config,4,4096);decoder.prepare_prefill(128)
 if mode=='reverse-cache':decoder.prepare_verification(4);decoder.prepare_compact_prefill(512)
 else:decoder.prepare_compact_prefill(512);decoder.prepare_verification(4)
 decoder._resident_prefill_graphs(128);decoder.prefill_compile_ms=0;base=[l['graph'] for l in decoder.layers]
 identity=['d7182d886f93a9efa73d6d556ee33ffe839a4e09e15965eeef46f295ab5b503d',decoder.io.checkpoint_format,inputs['target_tokenizer_sha256'],'text_zero_based_3axis_v1',inputs['profile_identity'],'text']
 if retained:assert decoder.session.configure_token_prefix(identity,inputs['host_entries'],inputs['host_bytes'])
 else:assert decoder.session.configure_token_tracking(identity)
 policy=garnet.continuous_scheduler();assert policy.initialize(decoder.session,16);assert policy.set_prefill_chunk(512)
 prepared=dict(memory=json.loads(garnet.memory_stats_json()),graphs=len(decoder.prefill_graphs),compact_families=decoder.session.compact_prefill_family_count,retention=retained,host_bytes=inputs['host_bytes'] if retained else 0)
 assert prepared['graphs']==224 and prepared['compact_families']==17
 for item in inputs['jobs']:submit(item)
 while jobs:
  survivors=[j for j in jobs.values() if j['label'] in ('B','C','D')]
  before=None
  if pending is not None:
   assert len(survivors)==3 and all(j['status']=='decode' for j in survivors)
   before=snapshot('before_admission',survivors)
  tick=json.loads(policy.next_json());assert tick['batches'];record=dict(plan=tick,executions=[]);ticks.append(record)
  if pending is not None:
   after=snapshot('after_admission',survivors);assert before['rows']==after['rows'],'Prefix restore altered surviving requests'
   q=next(q for p in tick['batches'] for q in p['requests'] if q['id']==pending)
   j=jobs[pending];j.update(row=q['row'],epoch=q['epoch']);assert q['row']==0
   admissions.append(dict(label=j['label'],row=q['row'],epoch=q['epoch'],cached_tokens=q['cached_tokens'],prompt_tokens=len(j['prompt_ids']),survivors=[s['label'] for s in survivors],before_frame=len(frames)-2,after_frame=len(frames)-1,cache=stats()));pending=None
  for plan in tick['batches']:
   for q in plan['requests']:jobs[q['id']].update(row=q['row'],epoch=q['epoch'])
   if plan['stage']=='prefill':
    width=plan['prefill_chunk'];byrow={q['row']:q for q in plan['requests']};ids=[];slots=[];flags=[];final=[0]*4
    for row in range(4):
     q=byrow.get(row)
     if q is not None:
      assert 0<=q['cached_tokens']<=q['prefill_offset'] and q['prefill_offset']+q['prefill_tokens']<=len(jobs[q['id']]['prompt_ids'])
      if q['prefill_tokens']==0:assert retained and q['cached_tokens']==q['prefill_offset']==len(jobs[q['id']]['prompt_ids'])
      if q['prefill_offset']+q['prefill_tokens']==len(jobs[q['id']]['prompt_ids']):final[row]=q['epoch']
     for t in range(width):
      on=q is not None and t<q['prefill_tokens'];position=q['prefill_offset']+t if on else 0
      ids.append(jobs[q['id']]['prompt_ids'][position] if on else 0);slots.append(float(position));flags.append(float(on))
    if any(flags):
     hidden=decoder.io.stack_tokens([decoder.io.embedding([ids[row*width+t] for row in range(4)]) for t in range(width)])
     decoder.session.run_tokens(decoder.prefill_profile_graphs(width),hidden,val([s for s in slots for axis in range(3)],[4,width,3]),val(slots,[4,width]),val(flags,[4,width]),-1,plan['epochs'],ids)
    predicted=decoder.session.predict_rows(final) if any(final) else None;complete=[];witness=[]
    for q in plan['requests']:
     j=jobs[q['id']];j['prefilled']=q['prefill_offset']+q['prefill_tokens']
     if final[q['row']]:
      token=predicted[q['row']];j['generated'].append(token);assert policy.complete_prefill(q['id'],token,False)==2;j['status']='decode';complete.append(dict(id=q['id'],token=token));witness.append(j)
     else:assert policy.complete_prefill_chunk(q['id'],q['prefill_tokens'])
    record['executions'].append(dict(stage='prefill',width=width,did_run=any(flags),tokens=int(sum(flags)),completed=complete,cache=stats()))
    if witness:snapshot('prefill',witness)
   else:
    assert plan['stage']=='decode';ids=[0]*4;slots=[0.]*4
    for q in plan['requests']:ids[q['row']]=q['token_id'];slots[q['row']]=float(q['slot']);assert ids[q['row']]==jobs[q['id']]['generated'][-1]
    if verify:
     byrow={q['row']:q for q in plan['requests']};flat=[];positions=[];flags=[]
     for row in range(4):
      q=byrow.get(row);depth=min(4,jobs[q['id']]['limit']-len(jobs[q['id']]['generated'])) if q is not None else 0
      for t in range(4):on=t<depth;flat.append(ids[row] if t==0 else 0);positions.append(slots[row]+t if on else 0.);flags.append(float(on))
     hidden=decoder.io.stack_tokens([decoder.io.embedding([flat[row*4+t] for row in range(4)]) for t in range(4)])
     native=json.loads(policy.verify_decode_json(hidden,val([s for s in positions for axis in range(3)],[4,4,3]),val(positions,[4,4]),val(flags,[4,4]),flat,[]));completed=native['requests']
     assert native['verification']['target_indexed_decode_dispatches']==225 and native['verification']['target_affine_misses']==0 and native['verification']['target_mpp_affine_dispatches']==0
     for q in completed:assert q['consumed']==1; jobs[q['id']]['generated'].extend(q['emitted'])
    else:
     predicted=decoder.session.run_tokens(base,decoder.io.embedding(ids),val([s for s in slots for axis in range(3)],[4,3]),val(slots,[4]),val([float(x) for x in plan['active_rows']],[4]),0,plan['epochs'],ids);native=None;completed=[]
     for q in plan['requests']:
      j=jobs[q['id']];token=predicted[q['row']];j['generated'].append(token);done=len(j['generated'])==j['limit'];assert policy.complete_decode(q['id'],token,done)==(3 if done else 2);completed.append(dict(id=q['id'],row=q['row'],epoch=q['epoch'],emitted=[token],finished=done))
    witness=[]
    for q in completed:
     j=jobs[q['id']]
     if len(j['generated'])==j['limit']-1:witness.append(j)
     if q['finished']:j['status']='finished';assert not decoder.session.release_request_row(j['row'],j['epoch']);del jobs[q['id']]
    record['executions'].append(dict(stage='decode',native=native,completed=completed,cache=stats()))
    if witness:snapshot('before_terminal',witness)
  predecessor='A' if next_followup==0 else inputs['followups'][next_followup-1]['label']
  previous=next((j for j in all_jobs.values() if j['label']==predecessor),None)
  if next_followup<len(inputs['followups']) and previous is not None and previous['status']=='finished':
   pending=submit(inputs['followups'][next_followup]);next_followup+=1
  if not jobs and not twins_admitted:
   for item in inputs['twins']:submit(item)
   twins_admitted=True
  save(False);assert len(ticks)<200
 assert len(all_jobs)==11 and sum(len(j['generated']) for j in all_jobs.values())==224
 assert decoder.prefill_compile_ms==0 and len(decoder.prefill_graphs)==224 and json.loads(policy.status_json())['tracked']==0
 if retained:
  expected={j['label']:j for j in control['jobs']}
  for j in all_jobs.values():assert j['generated']==expected[j['label']]['generated'],'Output IDs differ for '+j['label']
except Exception as error:
 failure=str(error);raise
finally:
 if decoder is not None and decoder.session is not None:last_cache=stats()
 if policy is not None:policy.close()
 if decoder is not None:decoder.close()
 garnet.shutdown();cleanup=json.loads(garnet.memory_stats_json())
 for k in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):assert cleanup[k]==0
 assert cleanup['metal_allocated_bytes']<=4194304
 save(failure is None and len(all_jobs)==11 and all(j['status']=='finished' for j in all_jobs.values()))
