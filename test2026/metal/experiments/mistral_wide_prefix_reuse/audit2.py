"""Exact native retained-prefix lineage, head-only admission and survivor audit."""
from pathlib import Path
import copy,hashlib,json
R=Path(__file__).resolve().parent;B=R.parent/'mistral-wide-lifecycle1'
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def need(ok,why):
 if not ok:raise ValueError(why)
plan1=read(R/'plan1.json');plan=read(R/'plan2.json');build=read(B/'build-audit1.json')
for version in [1,2]:
 pre=read(R/f'preflight{version}.json');need(pre['passed'] and read(R/f'postflight{version}.json')['passed'],'Incomplete collection')
 for n,h in pre['files_sha256'].items():need(sha(Path(n))==h,'Admitted input/source changed '+n)
last=0;pids=set()
for version in [1,2]:
 for mode in read(R/f'sequence{version}.json'):
  state=read(R/(mode+f'-status{version}.json'));need(state['status']=='terminal' and state['returncode']==0 and state['finished_ns']>state['started_ns']>=last and state['pid'] not in pids,'Nonserial/fresh/terminal model execution');last=state['finished_ns'];pids.add(state['pid'])
control=read(R/'ordinary-off-result1.json')
def shape(row):
 need(len(row['states'])==32 and all(len(pair)==2 for pair in row['states']),'Full KV coverage')
 for h in [v for pair in row['states'] for v in pair]+[row['hidden']]:need(isinstance(h,str) and len(h)==64 and all(c in '0123456789abcdef' for c in h),'Invalid full-state digest')
def normalized(d):
 result={}
 for frame in d['frames']:
  if frame['kind'] in ('prefill','before_terminal'):
   for row in frame['rows']:
    key=frame['kind'],row['label'];need(key not in result,'Duplicate owner checkpoint');shape(row);result[key]=row
 return result
expected_rows=normalized(control);need(len(expected_rows)==22,'Every job needs prefill and live before-terminal witness')
def lcp(a,b):
 n=0
 for x,y in zip(a,b):
  if x!=y:break
  n+=1
 return n
def check(d,version,require_hits):
 mode=d['mode'];need(d['completed'] and d['failure'] is None and mode in ('ordinary-off','ordinary-cache','verify-cache','reverse-cache'),'Incomplete arm')
 need(sha(Path(d['library']))==build['library_sha256']['on'] and 'native_wide_verify_preparation_lifetime_v1' in d['backend'],'Wrong native engine')
 inputs=read(R/f'input{version}.json');need(d['inputs']==inputs,'Wrong frozen inputs')
 a=dict(inputs);b=dict(control['inputs']);a.pop('host_bytes');b.pop('host_bytes');need(a==b,'Fresh reference differs beyond inactive host budget')
 need(len(d['jobs'])==11 and sum(len(j['generated']) for j in d['jobs'])==224,'Output coverage')
 need(d['prepared']['graphs']==224 and d['prepared']['compact_families']==17,'Prepared profiles changed')
 retained=mode!='ordinary-off';need(d['prepared']['retention']==retained and d['prepared']['host_bytes']==(inputs['host_bytes'] if retained else 0),'Wrong cache capacity profile')
 if retained:
  memory=d['prepared']['memory'];used=max(memory['metal_allocated_bytes'],memory['task_physical_footprint_bytes']);need(memory['task_info_status']==0 and used+2*inputs['host_bytes']<=memory['metal_recommended_working_set_bytes']-256*1024**2,'Native cache replacement-reserve envelope')
 need(d['cleanup']['metal_allocated_bytes']<=4194304 and all(d['cleanup'][k]==0 for k in ['garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets']),'Native owners retained')
 by={j['id']:j for j in d['jobs']};prior={j['label']:j for j in control['jobs']};need(set(j['label'] for j in d['jobs'])==set(prior),'Request coverage')
 for j in d['jobs']:
  p=prior[j['label']];need(all(j[k]==p[k] for k in ['prompt_ids','limit','generated','row','epoch']) and j['status']=='finished' and j['prefilled']==len(j['prompt_ids']) and len(j['generated'])==j['limit'],'Output/prompt/budget/placement differs from fresh control')
 rows=normalized(d);need(rows==expected_rows,'Full KV/hidden differs from fresh recomputation')
 cached={};offsets={};decode_progress={j['id']:1 for j in d['jobs']};heads=set();cache_samples=[d['cache']];native_calls=0;prefilled=0
 for tick in d['ticks']:
  need(len(tick['plan']['batches'])==len(tick['executions']),'Tick coverage')
  for p,e in zip(tick['plan']['batches'],tick['executions']):
   need(p['stage']==e['stage'],'Stage mismatch');cache_samples.append(e['cache'])
   if p['stage']=='prefill':
    total=0
    for q in p['requests']:
     j=by[q['id']];n=len(j['prompt_ids']);c=q['cached_tokens'];offset=q['prefill_offset'];count=q['prefill_tokens']
     if q['id'] not in cached:cached[q['id']]=c;offsets[q['id']]=c
     need(c==cached[q['id']] and 0<=c<=offset==offsets[q['id']] and 0<=count<=e['width'] and offset+count<=n and q['context_length']==offset+count,'Native cached-prefix/prefill ledger')
     if count==0:need(retained and c==offset==n,'Zero-token prefill without a full hit');heads.add(j['label'])
     offsets[q['id']]+=count;total+=count;prefilled+=count
    need(e['tokens']==total and e['did_run']==(total>0),'Head-only hit ran dummy prefill or skipped real input')
   else:
    need((e['native'] is not None)==(mode in ('verify-cache','reverse-cache')),'Verifier/control route membership')
    if e['native'] is not None:
     native_calls+=1;m=e['native']['verification'];need(m['target_indexed_decode_dispatches']==m['target_affine_dispatches']==225 and m['target_affine_misses']==m['target_mpp_affine_dispatches']==m['target_indexed_prefill_dispatches']==0,'Wrong target projection route')
     qs=e['native']['requests'];need(qs==e['completed'],'Native publication mismatch');need(m['target_kv_backups']==64 and m['commit_kv_restores']==(64 if any(q['submitted']>1 for q in qs) else 0),'Rejected suffix rollback')
    for q in e['completed']:
     j=by[q['id']];before=decode_progress[q['id']];need(q['row']==j['row'] and q['epoch']==j['epoch'] and q['emitted']==j['generated'][before:before+1] and q['finished']==(before+1==j['limit']),'Native target token/lease/budget')
     if e['native'] is not None:need(q['known']==j['generated'][before-1] and q['remaining_before']==j['limit']-before and q['submitted']==min(4,j['limit']-before) and q['consumed']==1 and not q['eos'],'Native verification remaining budget/known token')
     decode_progress[q['id']]+=1
 need(set(cached)==set(by) and all(offsets[i]==len(j['prompt_ids']) and decode_progress[i]==j['limit'] for i,j in by.items()),'Incomplete native token coverage')
 need(len(d['admissions'])==5 and [a['label'] for a in d['admissions']]==[a['label'] for a in inputs['followups']],'Returning query coverage')
 unchanged=0;hits={};prior_prompts=[j['prompt_ids'] for j in inputs['jobs']]
 for admission,item in zip(d['admissions'],inputs['followups']):
  need(admission['label']==item['label'] and admission['cached_tokens']==cached[next(j['id'] for j in d['jobs'] if j['label']==item['label'])] and admission['survivors']==['B','C','D'],'Admission/restore metadata')
  maximum=max(lcp(item['prompt_ids'],p) for p in prior_prompts);need(0<=admission['cached_tokens']<=maximum,'Prefix crossed an actual token divergence');prior_prompts.append(item['prompt_ids'])
  before=d['frames'][admission['before_frame']];after=d['frames'][admission['after_frame']];need(before['kind']=='before_admission' and after['kind']=='after_admission' and before['rows']==after['rows'] and [r['label'] for r in before['rows']]==['B','C','D'],'Restore altered another active owner')
  for r in before['rows']:shape(r)
  unchanged+=195;hits[item['label']]=admission['cached_tokens'];cache_samples.append(admission['cache'])
 twins={j['label']:cached[j['id']] for j in d['jobs'] if j['label'].startswith('twin_')}
 for c in cache_samples:
  need(c['retention_enabled']==retained and c['entries']<=inputs['host_entries'] and 0<=c['accounted_host_bytes']<=d['prepared']['host_bytes'],'Host cache exceeds profile')
  need(c['accounted_host_bytes']==c['private_host_bytes']+c['shared_resource_bytes'],'Host resource accounting')
 if retained:
  need(d['cache']['hits']==sum(v>0 for v in cached.values()) and d['cache']['misses']==sum(v==0 for v in cached.values()),'Native hit/miss counters differ from actual admissions')
  need(d['cache']['dense_capture_reused_blocks']>0 and d['cache']['dense_capture_copied_bytes']>0 and d['cache']['prefill_capture_skipped_mixed']>=11,'No retained shared blocks or decode-domain guard')
 else:need(not heads and all(c==0 for c in cached.values()) and d['cache']['accounted_host_bytes']==0,'Fresh control reused state')
 if require_hits:
  for label,n in plan['required_full_hits'].items():need(hits[label]==n and label in heads,'Full returning prefix was not retained')
  for label,n in plan['required_partial_hits'].items():need(hits[label]==n and label not in heads,'Append/edited suffix did not reuse expected prefix')
  need(plan['early_hit_bounds'][0]<=hits['A_early']<=plan['early_hit_bounds'][1] and 'A_early' not in heads,'Earlier divergence did not get a valid retained checkpoint')
  need(all(v==2052 for v in twins.values()) and set(twins)<=heads,'One-row to two-row appended-prompt full reuse missing')
 return dict(mode=mode,host_bytes=d['prepared']['host_bytes'],output_ids=224,state_digest_pairs_vs_fresh=1430 if retained else 0,survivor_unchanged_digest_pairs=unchanged,admitted_cached_tokens=hits,twin_cached_tokens=twins,head_only_requests=sorted(heads),actual_prefill_tokens=prefilled,native_verification_calls=native_calls,final_cache=d['cache'],maximum_sampled_cache_bytes=max(c['accounted_host_bytes'] for c in cache_samples))
check(control,1,False)
small=[read(R/(mode+'-result1.json')) for mode in ['ordinary-cache','verify-cache','reverse-cache']]
large=[read(R/(mode+'-result2.json')) for mode in ['ordinary-cache','verify-cache','reverse-cache']]
small_results=[check(d,1,False) for d in small];large_results=[check(d,2,True) for d in large]
need(all(d['admitted_cached_tokens']['A_exact']==0 for d in small_results),'Small capacity limitation not reproduced')
negative=[]
def reject(name,mutate):
 d=copy.deepcopy(large[1]);mutate(d)
 try:check(d,2,True)
 except (ValueError,KeyError,IndexError,TypeError):negative.append(name);return
 raise AssertionError('Accepted corruption '+name)
reject('wrong_cache_hit',lambda d:d['admissions'][0].update(cached_tokens=2047))
reject('wrong_restored_hidden',lambda d:next(f for f in d['frames'] if f['kind']=='prefill' and f['rows'][0]['label']=='A_exact')['rows'][0].update(hidden='0'*64))
reject('wrong_restored_kv',lambda d:next(f for f in d['frames'] if f['kind']=='prefill' and f['rows'][0]['label']=='B_exact')['rows'][0]['states'][0].__setitem__(0,'0'*64))
reject('wrong_survivor_cache',lambda d:d['frames'][d['admissions'][0]['after_frame']]['rows'][0]['states'][0].__setitem__(1,'0'*64))
def head(d):return next(e for t in d['ticks'] for e in t['executions'] if e['stage']=='prefill' and e['tokens']==0)
reject('dummy_prefill_on_full_hit',lambda d:head(d).update(did_run=True))
reject('fake_host_budget',lambda d:d['cache'].update(accounted_host_bytes=4*1024**3))
reject('missing_shared_blocks',lambda d:d['cache'].update(dense_capture_reused_blocks=0))
reject('decoded_snapshot_domain',lambda d:d['cache'].update(prefill_capture_skipped_mixed=0))
def decode(d):return next(e for t in d['ticks'] for e in t['executions'] if e['stage']=='decode')
reject('fake_native_remaining',lambda d:decode(d)['native']['requests'][0].update(remaining_before=999))
reject('missing_rollback',lambda d:decode(d)['native']['verification'].update(commit_kv_restores=63))
reject('wrong_target_id',lambda d:d['jobs'][0]['generated'].__setitem__(1,999))
reject('candidate_skipped_verifier',lambda d:decode(d).update(native=None))
reject('fake_memory_envelope',lambda d:d['prepared']['memory'].update(metal_allocated_bytes=d['prepared']['memory']['metal_recommended_working_set_bytes']))
reject('fake_cache_hit_counter',lambda d:d['cache'].update(hits=0))
result=dict(passed=True,scope=__doc__,small_profile=dict(host_bytes=2*1024**3,all_states_outputs_exact=True,full_A_return_qualified=False,reason='Host budget eviction, not a matching/state error',arms=small_results),qualified_profile=dict(host_bytes=3*1024**3,arms=large_results,state_digest_pairs_vs_fresh=4290,survivor_unchanged_digest_pairs=2925,full_exact_append_suffix_early_and_twin_reuse=True),fresh_control_reused_without_inference_rerun=True,fresh_control_retention_disabled=True,source_changed=False,serial_fresh_engines=7,negative_controls_rejected=negative,prefix_retention_qualified=True,wide_cancel_refill_qualification_unchanged=True,learned_helper_batching_qualified=False,new_serving_speed_metrics=False,selected_full_serving_comparison_promoted=False,no_comparator_execution=True,goal_complete=False)
result['files_sha256']={str(p):sha(p) for p in [R/'audit2.py',R/'worker1.py',R/'input1.json',R/'input2.json',R/'plan1.json',R/'plan2.json',R/'preflight1.json',R/'preflight2.json',R/'postflight1.json',R/'postflight2.json',R/'ordinary-off-result1.json']+[R/(mode+f'-result{v}.json') for v in [1,2] for mode in ['ordinary-cache','verify-cache','reverse-cache']]}
(R/'audit2.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='files_sha256'},indent=2))
