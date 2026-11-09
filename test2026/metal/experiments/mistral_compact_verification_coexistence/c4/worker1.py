"""Untimed C4 real-model ownership reference/verification; no proposal oracle."""
import sys,json,os,time
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet,tensor
from xModel.mistral.text_7b.metal_decoder import MistralMetalDecoder
with open(sys.argv[3]) as f:inputs=json.load(f)
output=sys.argv[4];mode=sys.argv[5];assert mode in ('ordinary','verify') and not os.path.exists(output)
assert 'native_scheduled_dense_verify_v1' in garnet.backend_info()
with open(inputs['target_model']+'/config.json') as f:config=json.load(f)
def val(a,shape):return tensor.tensor(a,shape=shape,dtype=tensor.float32)
decoder=None;policy=None;jobs={};all_jobs={};ticks=[];witnesses=[];rejections=[];prepared=None;cleanup=None;marks=set();cancelled=False

def save(done):
 with open(output,'w') as f:json.dump(dict(completed=done,mode=mode,inputs=inputs,backend=garnet.backend_info(),library=garnet.native_library_path(),prepared=prepared,jobs=list(all_jobs.values()),ticks=ticks,witnesses=witnesses,rejections=rejections,cleanup=cleanup,scope=__doc__),f,indent=2)
def witness(label):
 if label in marks:return
 marks.add(label);states=[]
 for layer in range(32):
  pair=decoder.session.snapshot(layer)
  states.append([[decoder.io.verification_f32_sha256(decoder.io.rows(part,row,1)) for part in pair] for row in range(4)])
 bank=decoder.session.diagnostic_final_hidden();hidden=[decoder.io.verification_f32_sha256(decoder.io.rows(bank,row,1)) for row in range(4)]
 owners=[dict(id=j['native_request'],label=j['label'],row=j.get('row'),epoch=j.get('epoch'),generated=len(j['generated']),prefilled=j['prefilled']) for j in jobs.values()]
 witnesses.append(dict(label=label,states=states,hidden=hidden,owners=owners,status=json.loads(policy.status_json())));save(False);print('c4_witness',mode,label,'live',len(jobs),flush=True)
def reject(label,args):
 before=policy.status_json();failed=False
 try:policy.verify_decode_json(*args)
 except Exception:failed=True
 assert failed and policy.status_json()==before;rejections.append(label)
try:
 decoder=MistralMetalDecoder(inputs['target_model']+'/model.safetensors',config,4,4096);decoder.prepare_prefill(128);decoder.prepare_verification(1)
 decoder.prefill_compile_ms=0;base=[layer['graph'] for layer in decoder.layers];graphs=decoder._resident_prefill_graphs(128)
 assert decoder.session.configure_token_tracking(['d7182d886f93a9efa73d6d556ee33ffe839a4e09e15965eeef46f295ab5b503d',decoder.io.checkpoint_format,inputs['target_tokenizer_sha256'],'text_zero_based_3axis_v1','scheduled-B4-C4096-T128-verify1','text'])
 policy=garnet.continuous_scheduler();assert policy.initialize(decoder.session,16);assert policy.set_prefill_chunk(128)
 prepared=dict(memory=json.loads(garnet.memory_stats_json()),target_graphs=len(decoder.prefill_graphs));assert prepared['memory']['metal_allocated_bytes']<=prepared['memory']['metal_recommended_working_set_bytes']-536870912
 print('c4_prepared',mode,prepared['memory']['metal_allocated_bytes'],flush=True)
 for item in inputs['jobs']:
  identity=policy.submit_tokens(item['prompt_ids'],item['limit'],0);assert identity>0
  job=dict(item,native_request=identity,generated=[],prefilled=0,status='queued');jobs[identity]=job;all_jobs[identity]=job
 while jobs:
  tick=json.loads(policy.next_json());assert tick['batches'];ticks.append(dict(plan=tick,executions=[]));record=ticks[-1]
  for plan in tick['batches']:
   for q in plan['requests']:
    job=jobs[q['id']];job['row']=q['row'];job['epoch']=q['epoch'];job['status']=plan['stage']
   if plan['stage']=='prefill':
    if any(jobs[q['id']]['label']=='E' and q['prefill_offset']==0 for q in plan['requests']):witness('claim_E')
    if any(jobs[q['id']]['label']=='F' and q['prefill_offset']==0 for q in plan['requests']):witness('claim_F')
    width=plan.get('prefill_chunk',128);assert width==128;byrow={q['row']:q for q in plan['requests']};ids=[];slots=[];flags=[];final_epochs=[0]*4
    for row in range(4):
     q=byrow.get(row)
     if q is not None and q['prefill_offset']+q['prefill_tokens']==len(jobs[q['id']]['prompt_ids']):final_epochs[row]=q['epoch']
     for t in range(width):
      on=q is not None and t<q['prefill_tokens'];position=q['prefill_offset']+t if on else 0
      ids.append(jobs[q['id']]['prompt_ids'][position] if on else 0);slots.append(float(position));flags.append(float(on))
    hidden=decoder.io.stack_tokens([decoder.io.embedding([ids[row*width+t] for row in range(4)]) for t in range(width)])
    decoder.session.run_tokens(graphs,hidden,val([s for s in slots for axis in range(3)],[4,width,3]),val(slots,[4,width]),val(flags,[4,width]),-1,plan['epochs'],ids)
    predicted=decoder.session.predict_rows(final_epochs) if any(final_epochs) else None
    completed=[]
    for q in plan['requests']:
     job=jobs[q['id']];job['prefilled']=q['prefill_offset']+q['prefill_tokens'];assert decoder.session.diagnostic_tracked_token_count(q['row'],q['epoch'])==job['prefilled']
     if final_epochs[q['row']]:
      token=predicted[q['row']];job['generated'].append(token);finished=token in inputs['stops'];assert policy.complete_prefill(q['id'],token,finished)==(3 if finished else 2);job['status']='finished' if finished else 'decode';completed.append(dict(id=q['id'],token=token,finished=finished))
      if finished:del jobs[q['id']]
     else:assert policy.complete_prefill_chunk(q['id'],q['prefill_tokens'])
    record['executions'].append(dict(stage='prefill',completed=completed))
    e=next(j for j in all_jobs.values() if j['label']=='E')
    if e['prefilled']>=1024 and not cancelled:
     witness('mid_E');c=next(j for j in jobs.values() if j['label']=='C');assert policy.cancel(c['native_request']);c['status']='cancelled';del jobs[c['native_request']];cancelled=True;witness('cancel_C')
   else:
    assert plan['stage']=='decode';ids=[0]*4;slots=[0.]*4
    for q in plan['requests']:ids[q['row']]=q['token_id'];slots[q['row']]=float(q['slot']);assert ids[q['row']]==jobs[q['id']]['generated'][-1]
    if mode=='ordinary':
     predicted=decoder.session.run_tokens(base,decoder.io.embedding(ids),val([s for s in slots for axis in range(3)],[4,3]),val(slots,[4]),val([float(x) for x in plan['active_rows']],[4]),0,plan['epochs'],ids);native=None;completed=[]
     for q in plan['requests']:
      job=jobs[q['id']];token=predicted[q['row']];job['generated'].append(token);finished=token in inputs['stops'] or len(job['generated'])==job['limit'];assert policy.complete_decode(q['id'],token,finished)==(3 if finished else 2);completed.append(dict(id=q['id'],row=q['row'],epoch=q['epoch'],emitted=[token],finished=finished))
    else:
     args=[decoder.io.stack_tokens([decoder.io.embedding(ids)]),val([s for s in slots for axis in range(3)],[4,1,3]),val(slots,[4,1]),val([float(x) for x in plan['active_rows']],[4,1]),ids,inputs['stops']]
     if 'mixed_unscheduled_row' not in rejections and 0 in plan['active_rows']:
      witness('before_invalid_mixed');free=plan['active_rows'].index(0);bad=list(args);flags=[float(x) for x in plan['active_rows']];flags[free]=1.;bad[3]=val(flags,[4,1]);reject('mixed_unscheduled_row',bad)
      bad=list(args);known=list(ids);active_row=plan['requests'][0]['row'];known[active_row]=(known[active_row]+1)%32768;bad[4]=known;reject('wrong_known_mixed',bad);witness('after_invalid_mixed')
     native=json.loads(policy.verify_decode_json(*args));completed=native['requests'];assert len(completed)==len(plan['requests'])
     for q in completed:
      job=jobs[q['id']];assert q['row']==job['row'] and q['epoch']==job['epoch'] and q['remaining_before']==job['limit']-len(job['generated']) and q['consumed']==q['submitted']==1
      job['generated'].extend(q['emitted']);assert q['finished']==(q['eos'] or len(job['generated'])==job['limit'])
     if 'stale_tick_replay' not in rejections:reject('stale_tick_replay',args)
    record['executions'].append(dict(stage='decode',native=native,completed=completed))
    finished_labels=[]
    for q in completed:
     if q['finished']:
      job=jobs[q['id']];job['status']='finished';finished_labels.append(job['label']);assert not decoder.session.release_request_row(job['row'],job['epoch']);del jobs[q['id']]
    for label in finished_labels:
     if label in ('A','E','B'):witness('finish_'+label)
  if all(next(j for j in all_jobs.values() if j['label']==x)['prefilled']==len(next(j for j in all_jobs.values() if j['label']==x)['prompt_ids']) for x in ['A','B','C','D']):witness('initial_prefilled')
  assert len(ticks)<100
 witness('all_done');assert cancelled and all(j['status'] in ('finished','cancelled') for j in all_jobs.values())
 assert json.loads(policy.status_json())['active']==json.loads(policy.status_json())['tracked']==0
finally:
 if policy is not None:policy.close()
 if decoder is not None:decoder.close()
 garnet.shutdown();cleanup=json.loads(garnet.memory_stats_json())
 for k in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):assert cleanup[k]==0
 assert cleanup['metal_allocated_bytes']<=4194304
 save(all(j['status'] in ('finished','cancelled') for j in all_jobs.values()) and len(all_jobs)==6)
