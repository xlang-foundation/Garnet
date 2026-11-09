"""Untimed C4/T4 ownership and rollback using zero proposals; no learned draft or speed claim."""
import sys,json,os,time
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet,tensor
from xModel.mistral.text_7b.metal_decoder import MistralMetalDecoder
with open(sys.argv[3]) as f:inputs=json.load(f)
output=sys.argv[4];mode=sys.argv[5];assert mode in ('ordinary','verify','reverse') and not os.path.exists(output)
assert 'native_scheduled_dense_verify_v1' in garnet.backend_info()
with open(inputs['target_model']+'/config.json') as f:config=json.load(f)
def val(a,shape):return tensor.tensor(a,shape=shape,dtype=tensor.float32)
decoder=None;policy=None;jobs={};all_jobs={};ticks=[];witnesses=[];rejections=[];prepared=None;cleanup=None;marks=set();cancelled=False;late_admitted=True

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
def reject(label,args,session=False):
 before=policy.status_json();failed=False
 try:(decoder.session.verify_greedy if session else policy.verify_decode_json)(*args)
 except Exception:failed=True
 assert failed and policy.status_json()==before;rejections.append(label)
try:
 decoder=MistralMetalDecoder(inputs['target_model']+'/model.safetensors',config,4,4096);decoder.prepare_prefill(128)
 if mode=='reverse':decoder.prepare_verification(4);decoder.prepare_compact_prefill(512)
 else:decoder.prepare_compact_prefill(512);decoder.prepare_verification(4)
 assert 'native_wide_verify_preparation_lifetime_v1' in garnet.backend_info() and decoder.session.wide_verification_preparation_open
 assert decoder.session.compact_prefill_family_count==17 and decoder.session.compact_prefill_max_chunk==512
 decoder.prefill_compile_ms=0;base=[layer['graph'] for layer in decoder.layers];graphs=decoder._resident_prefill_graphs(128)
 assert decoder.session.configure_token_tracking(['d7182d886f93a9efa73d6d556ee33ffe839a4e09e15965eeef46f295ab5b503d',decoder.io.checkpoint_format,inputs['target_tokenizer_sha256'],'text_zero_based_3axis_v1','scheduled-B4-C4096-prefill-up512-verify4-fp32-v1','text'])
 policy=garnet.continuous_scheduler();assert policy.initialize(decoder.session,16);assert policy.set_prefill_chunk(512)
 prepared=dict(memory=json.loads(garnet.memory_stats_json()),target_graphs=len(decoder.prefill_graphs),compact_families=decoder.session.compact_prefill_family_count,maximum_compact_chunk=decoder.session.compact_prefill_max_chunk);assert prepared['memory']['metal_allocated_bytes']<=prepared['memory']['metal_recommended_working_set_bytes']-536870912
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
    width=plan.get('prefill_chunk',128);assert width in (2,4,8,16,32,64,128,256,512);byrow={q['row']:q for q in plan['requests']};ids=[];slots=[];flags=[];final_epochs=[0]*4
    for row in range(4):
     q=byrow.get(row)
     if q is not None and q['prefill_offset']+q['prefill_tokens']==len(jobs[q['id']]['prompt_ids']):final_epochs[row]=q['epoch']
     for t in range(width):
      on=q is not None and t<q['prefill_tokens'];position=q['prefill_offset']+t if on else 0
      ids.append(jobs[q['id']]['prompt_ids'][position] if on else 0);slots.append(float(position));flags.append(float(on))
    hidden=decoder.io.stack_tokens([decoder.io.embedding([ids[row*width+t] for row in range(4)]) for t in range(width)])
    decoder.session.run_tokens(decoder.prefill_profile_graphs(width),hidden,val([s for s in slots for axis in range(3)],[4,width,3]),val(slots,[4,width]),val(flags,[4,width]),-1,plan['epochs'],ids)
    predicted=decoder.session.predict_rows(final_epochs) if any(final_epochs) else None
    completed=[]
    for q in plan['requests']:
     job=jobs[q['id']];job['prefilled']=q['prefill_offset']+q['prefill_tokens'];assert decoder.session.diagnostic_tracked_token_count(q['row'],q['epoch'])==job['prefilled']
     if final_epochs[q['row']]:
      token=predicted[q['row']];job['generated'].append(token);finished=token in inputs['stops'];assert policy.complete_prefill(q['id'],token,finished)==(3 if finished else 2);job['status']='finished' if finished else 'decode';completed.append(dict(id=q['id'],token=token,finished=finished))
      if finished:del jobs[q['id']]
     else:assert policy.complete_prefill_chunk(q['id'],q['prefill_tokens'])
    record['executions'].append(dict(stage='prefill',completed=completed,physical_chunk=width,compact_rows=decoder.session.compact_prefill_last_rows,compact_first=decoder.session.compact_prefill_last_first))
    if decoder.session.compact_prefill_last_rows==2:witness('first_B2_prefill')
   else:
    assert plan['stage']=='decode';ids=[0]*4;slots=[0.]*4
    for q in plan['requests']:ids[q['row']]=q['token_id'];slots[q['row']]=float(q['slot']);assert ids[q['row']]==jobs[q['id']]['generated'][-1]
    if mode=='ordinary':
     predicted=decoder.session.run_tokens(base,decoder.io.embedding(ids),val([s for s in slots for axis in range(3)],[4,3]),val(slots,[4]),val([float(x) for x in plan['active_rows']],[4]),0,plan['epochs'],ids);native=None;completed=[]
     for q in plan['requests']:
      job=jobs[q['id']];token=predicted[q['row']];job['generated'].append(token);finished=token in inputs['stops'] or len(job['generated'])==job['limit'];assert policy.complete_decode(q['id'],token,finished)==(3 if finished else 2);completed.append(dict(id=q['id'],row=q['row'],epoch=q['epoch'],emitted=[token],finished=finished))
    else:
     byrow={q['row']:q for q in plan['requests']};flat=[];positions=[];flags=[];budgets=[0]*4;depths=[0]*4
     for row in range(4):
      q=byrow.get(row)
      if q is not None:budgets[row]=jobs[q['id']]['limit']-len(jobs[q['id']]['generated']);depths[row]=min(4,budgets[row])
      for t in range(4):
       on=t<depths[row];flat.append(ids[row] if t==0 else 0);positions.append(slots[row]+float(t) if on else 0.);flags.append(float(on))
     args=[decoder.io.stack_tokens([decoder.io.embedding([flat[row*4+t] for row in range(4)]) for t in range(4)]),val([s for s in positions for axis in range(3)],[4,4,3]),val(positions,[4,4]),val(flags,[4,4]),flat,inputs['stops']]
     if 'mixed_unscheduled_row' not in rejections and 0 in plan['active_rows']:
      witness('before_invalid_mixed');free=plan['active_rows'].index(0);bad=list(args);bad_flags=list(flags);bad_flags[free*4]=1.;bad[3]=val(bad_flags,[4,4]);reject('mixed_unscheduled_row',bad)
      bad=list(args);known=list(flat);active_row=plan['requests'][0]['row'];known[active_row*4]=(known[active_row*4]+1)%32768;bad[4]=known;reject('wrong_known_mixed',bad)
      bad=list(args);bad[2]=val(positions,[16]);reject('wrong_slots_shape',bad)
      epochs=list(plan['epochs']);epochs[active_row]+=1;reject('wrong_native_epoch',list(args)+[budgets,epochs],True)
      remaining=list(budgets);remaining[active_row]=-1;reject('negative_native_budget',list(args)+[remaining,plan['epochs']],True)
      witness('after_invalid_mixed')
     if 'submitted_over_budget' not in rejections and any(0<b<4 for b in budgets):
      witness('before_invalid_budget');row=next(r for r in range(4) if 0<budgets[r]<4);bad=list(args);bad_flags=list(flags);bad_positions=list(positions)
      for t in range(4):bad_flags[row*4+t]=1.;bad_positions[row*4+t]=slots[row]+float(t)
      bad[1]=val([s for s in bad_positions for axis in range(3)],[4,4,3]);bad[2]=val(bad_positions,[4,4]);bad[3]=val(bad_flags,[4,4]);reject('submitted_over_budget',bad);witness('after_invalid_budget')
     native=json.loads(policy.verify_decode_json(*args));completed=native['requests'];assert len(completed)==len(plan['requests'])
     assert native['verification']['target_affine_misses']==0 and native['verification']['target_indexed_decode_dispatches']==225 and native['verification']['target_mpp_affine_dispatches']==0 and native['verification']['target_indexed_prefill_dispatches']==0
     for q in completed:
      job=jobs[q['id']];assert q['row']==job['row'] and q['epoch']==job['epoch'] and q['remaining_before']==job['limit']-len(job['generated']) and q['consumed']==1 and q['submitted']==min(4,q['remaining_before'])
      job['generated'].extend(q['emitted']);assert q['finished']==(q['eos'] or len(job['generated'])==job['limit'])
     if 'stale_tick_replay' not in rejections:reject('stale_tick_replay',args)
    assert decoder.session.compact_prefill_last_rows==0
    record['executions'].append(dict(stage='decode',native=native,completed=completed,compact_rows=0,submitted_tokens=list(flat) if mode!='ordinary' else None,submitted_positions=list(positions) if mode!='ordinary' else None,submitted_active=list(flags) if mode!='ordinary' else None))
    if 'first_B2_prefill' in marks:witness('after_B2_verification')
    finished_labels=[]
    for q in completed:
     if q['finished']:
      job=jobs[q['id']];job['status']='finished';finished_labels.append(job['label']);assert not decoder.session.release_request_row(job['row'],job['epoch']);del jobs[q['id']]
    for label in finished_labels:
     if label in ('A','E','B'):witness('finish_'+label)
  if not jobs and not late_admitted:
   witness('before_late_G');item=inputs['late_probe'];identity=policy.submit_tokens(item['prompt_ids'],item['limit'],0);assert identity>0
   job=dict(item,native_request=identity,generated=[],prefilled=0,status='queued');jobs[identity]=job;all_jobs[identity]=job;late_admitted=True
  assert len(ticks)<100
 assert decoder.prefill_compile_ms==0 and len(decoder.prefill_graphs)==224 and decoder.session.compact_prefill_family_count==17
 witness('all_done');assert all(j['status'] in ('finished','cancelled') for j in all_jobs.values())
 assert json.loads(policy.status_json())['active']==json.loads(policy.status_json())['tracked']==0
finally:
 if policy is not None:policy.close()
 if decoder is not None:decoder.close()
 garnet.shutdown();cleanup=json.loads(garnet.memory_stats_json())
 for k in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):assert cleanup[k]==0
 assert cleanup['metal_allocated_bytes']<=4194304
 save(all(j['status'] in ('finished','cancelled') for j in all_jobs.values()) and len(all_jobs)==3)
