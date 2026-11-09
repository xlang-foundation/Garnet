"""Audit engine-lifetime guards, true T4 rollback, C4 ownership and B2 overlap."""
from pathlib import Path
import copy,hashlib,json
from audit_owner2 import check as owner_check
from audit_probe_core1 import check as probe_check
R=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def need(ok,why):
 if not ok:raise ValueError(why)
plan=read(R/'plan2.json');pre=read(R/'run2-preflight.json');build=read(R/'build-audit1.json')
need(pre['passed'] and build['passed'] and read(R/'run2-postflight.json')['passed'],'Incomplete native collection')
for n,h in pre['files_sha256'].items():need(sha(Path(n))==h,'Admitted source/model changed '+n)
last=0;pids=set()
for label,worker,lib,fixture,mode,version in [list(x)+[1] for x in read(R/'sequence1.json')]+[list(x)+[2] for x in read(R/'sequence2.json')]:
 state=read(R/(label+'-status'+str(version)+'.json'));need(state['status']=='terminal' and state['returncode']==0 and state['finished_ns']>state['started_ns']>=last and state['pid'] not in pids,'Nonserial/fresh/terminal process '+label);last=state['finished_ns'];pids.add(state['pid'])
def guard_check(reports):
 expected={
 'off':['wide_adapter_default_off'],
 'on-claim':['short_layer_list','duplicate_layer_owner','base_head_owner','late_native_after_claim','late_native_downgrade','late_native_after_release','late_native_after_reset'],
 'on-run':['short_layer_list','duplicate_layer_owner','base_head_owner','late_native_after_unmanaged_run','late_native_after_unmanaged_reset'],
 'on-adapter-late':['late_adapter_before_compile'],
 'on-compact':['short_layer_list','duplicate_layer_owner','base_head_owner','late_native_after_claim','late_native_after_release','late_native_after_reset','late_compact_after_reset'],
 'on-capacity':['unsupported_capacity'],'on-batch':['unsupported_batch']}
 pairs=0;rejected=0
 need(set(reports)==set(expected),'Guard mode coverage')
 for mode,labels in expected.items():
  report=reports[mode];need(report['completed'] and report['mode']==mode,'Incomplete guard')
  need([r['label'] for r in report['rejections']]==labels,'Native guard coverage')
  if mode=='on-run':need(report['reset_existing_profile_replay_exact'],'Reset broke existing profile')
  for item in report['rejections']:
   a,b=item['before'],item['after'];need(a==b and item['error'],'Preparation rejection modified native state/pipelines')
   rows=2 if mode=='on-batch' else 4;need(len(a['kv'])==32 and all(len(layer)==rows and all(len(row)==2 for row in layer) for layer in a['kv']),'Guard full-cache coverage')
   if a['hidden'] is not None:need(len(a['hidden'])==rows,'Guard hidden coverage')
   pairs+=64*rows+(len(a['hidden']) if a['hidden'] is not None else 0);rejected+=1
  need(report['cleanup']['metal_allocated_bytes']<=4194304 and all(report['cleanup'][k]==0 for k in ['garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets']),'Guard native owners retained')
 return dict(fresh_engines=len(expected),native_preparation_rejections=rejected,unchanged_state_digest_pairs=pairs,reset_does_not_reopen_preparation=True,reset_existing_profile_replay_exact=True,late_adapter_compilation_rejected=True)
def spans(report):
 calls=0;speculative_rows=set();depths=set();restored=0;total_positions=0;chunks=set()
 need('native_wide_verify_preparation_lifetime_v1' in report['backend'],'Wrong native capability')
 need('zero proposals' in report['scope'],'Wrong proposal disclosure')
 need(sha(Path(report['library']))==build['library_sha256']['on'],'Wrong native library')
 jobs={j['native_request']:j for j in report['jobs']}
 for tick in report['ticks']:
  for p,e in zip(tick['plan']['batches'],tick['executions']):
   if p['stage']=='prefill':chunks.add(e['physical_chunk']);continue
   if report['mode']=='ordinary':need(e['submitted_tokens'] is None and e['submitted_active'] is None and e['submitted_positions'] is None,'Control submitted guesses');continue
   calls+=1;tokens=e['submitted_tokens'];active=e['submitted_active'];positions=e['submitted_positions'];need(len(tokens)==len(active)==len(positions)==16,'Explicit 16-position controls')
   qby={q['row']:q for q in p['requests']};expected_active=[];expected_positions=[];expected_tokens=[]
   for row in range(4):
    q=qby.get(row);budget=jobs[q['id']]['limit']-q['generated_tokens'] if q is not None else 0;depth=min(4,budget);depths.add(depth) if depth else None
    for t in range(4):expected_active.append(float(t<depth));expected_positions.append(float(q['slot']+t) if t<depth else 0.);expected_tokens.append(q['token_id'] if q is not None and t==0 else 0)
   need(tokens==expected_tokens and active==expected_active and positions==expected_positions,'Submitted zero-proposal slots/mask/known rows differ from native plan')
   speculative_rows.add(int(sum(active)));total_positions+=int(sum(active));native=e['native'];need(native['requests']==e['completed'],'Publication differs from native result')
   m=native['verification'];expected_restores=64 if any(q['submitted']>q['consumed'] for q in native['requests']) else 0
   need(m['target_indexed_decode_dispatches']==m['target_affine_dispatches']==225 and m['target_affine_misses']==m['target_mpp_affine_dispatches']==m['target_indexed_prefill_dispatches']==0,'Target selected a noncanonical/unprepared projection route')
   need(m['target_kv_backups']==64 and m['commit_kv_restores']==expected_restores and m['commit_replay_layers']==0,'Dense layer rollback dispatch differs');restored+=expected_restores
   for q in native['requests']:
    need(q['consumed']==1 and len(q['emitted'])==1,'Zero proposal unexpectedly accepted beyond first input')
    if q['submitted']>1:need(q['emitted'][0]!=0,'Zero first proposal did not actually mismatch')
 return dict(native_calls=calls,submitted_physical_counts=sorted(speculative_rows),submitted_depths=sorted(depths),actual_prefill_chunks=sorted(chunks),submitted_positions=total_positions,kv_restore_dispatches=restored)
guards={mode:read(R/('guard-'+mode+'-result1.json')) for mode in plan['guards']}
a=read(R/'ordinary-result2.json');b=read(R/'verify-result2.json');rev=read(R/'reverse-result2.json');pa=read(R/'probe-ordinary-result1.json');pb=read(R/'probe-verify-result1.json')
def audit(guards,a,b,rev,pa,pb):
 g=guard_check(guards);need(a['mode']=='ordinary' and b['mode']=='verify' and rev['mode']=='reverse','Arm identity');r=copy.deepcopy(rev);r['mode']='verify'
 primary=owner_check(a,b);reverse=owner_check(a,r);probe=probe_check(pa,pb)
 span_results={d['mode']+suffix:spans(d) for d,suffix in [(a,''),(b,''),(rev,''),(pa,'-probe'),(pb,'-probe')]}
 need(16 in span_results['verify']['submitted_physical_counts'] and span_results['verify']['submitted_depths']==[1,2,3,4],'Wide/T4/depth coverage')
 need(16 in span_results['reverse']['submitted_physical_counts'],'Reverse wide geometry')
 need(4 in span_results['verify-probe']['submitted_depths'],'B2 overlap never submitted T4')
 # Ordinary controls are independently rerun, then bound to the earlier T1
 # qualified fixture rather than redefining expected tokens or owner lifetime.
 prior=read(R.parent/'mistral-compact-verification-coexist1/ordinary-result1.json');prior_probe=read(R.parent/'mistral-compact-verification-coexist1/probe-ordinary-result1.json')
 old_control=read(R/'ordinary-result1.json')
 need(old_control['jobs']==prior['jobs'] and old_control['witnesses']==prior['witnesses'],'Original ordinary ownership behavior changed since frozen T1 checkpoint')
 for current,old in zip(a['jobs'],prior['jobs']):
  n=min(len(current['generated']),len(old['generated']));need(current['label']==old['label'] and current['prompt_ids']==old['prompt_ids'] and current['generated'][:n]==old['generated'][:n],'Ordinary output prefix changed')
 need(pa['jobs']==prior_probe['jobs'] and pa['witnesses']==prior_probe['witnesses'],'Ordinary B2 behavior changed since frozen T1 checkpoint')
 return dict(passed=True,preparation=g,primary=primary,reverse=reverse,probe=probe,spans=span_results,full_state_digest_pairs=primary['full_four_row_state_digest_pairs']+reverse['full_four_row_state_digest_pairs']+probe['full_four_row_state_digest_pairs'],invalid_call_state_digest_pairs=primary['rejected_call_state_digest_pairs']+reverse['rejected_call_state_digest_pairs']+probe['rejected_call_state_digest_pairs'],wide_cancel_refill_survivors_qualified=True,b2_compact_interleave_qualified=True,closed_preparation_lifetime_qualified=True,independent_native_budgets_epochs_qualified=True,all_owned_engines_terminal=True,default_option_on=False,prefix_retention_qualified=False,learned_helper_batching_qualified=False,performance_measured=False,selected_full_serving_comparison_promoted=False,no_comparator_execution=True,goal_complete=False)
result=audit(guards,a,b,rev,pa,pb);negative=[]
def reject(name,mutate):
 args=copy.deepcopy([guards,a,b,rev,pa,pb]);mutate(args)
 try:audit(*args)
 except (ValueError,KeyError,IndexError,TypeError):negative.append(name);return
 raise AssertionError('Corruption accepted '+name)
def first(d):return next(e for t in d['ticks'] for e in t['executions'] if e['stage']=='decode')
reject('missing_reset_guard',lambda r:r[0]['on-claim']['rejections'].pop())
reject('reset_pipeline_mutation',lambda r:r[0]['on-run']['rejections'][0]['after'].update(pipelines=999))
reject('wrong_survivor_hidden',lambda r:next(w for w in r[2]['witnesses'] if w['label']=='cancel_C')['hidden'].__setitem__(1,'0'*64))
reject('wrong_reused_row_epoch',lambda r:next(j for j in r[2]['jobs'] if j['label']=='E').update(epoch=1))
reject('wrong_native_budget',lambda r:first(r[2])['native']['requests'][0].update(remaining_before=999))
reject('false_submitted_depth',lambda r:first(r[2])['native']['requests'][0].update(submitted=1))
reject('missing_rollback_bank',lambda r:first(r[2])['native']['verification'].update(commit_kv_restores=63))
reject('hidden_oracle_proposal',lambda r:first(r[2])['submitted_tokens'].__setitem__(1,999))
reject('inactive_submitted_position',lambda r:first(r[2])['submitted_active'].__setitem__(15,.5))
reject('wrong_b2_span',lambda r:next(e for t in r[5]['ticks'] for e in t['executions'] if e['stage']=='prefill' and e['compact_rows']==2).update(compact_first=0))
reject('wrong_b2_survivor_state',lambda r:next(w for w in r[5]['witnesses'] if w['label']=='after_B2_verification')['states'][0][0].__setitem__(0,'0'*64))
reject('missing_native_epoch_rejection',lambda r:r[2]['rejections'].remove('wrong_native_epoch'))
result['negative_controls_rejected']=negative
result['files_sha256']={str(p):sha(p) for p in [R/'audit_all2.py',R/'audit_owner2.py',R/'audit_probe_core1.py',R/'run2-preflight.json',R/'run2-postflight.json',R/'build-audit1.json',R/'plan2.json']+[R/(label+'-result'+str(version)+'.json') for label,worker,lib,fixture,mode,version in [list(x)+[1] for x in read(R/'sequence1.json')]+[list(x)+[2] for x in read(R/'sequence2.json')]]}
(R/'audit2.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='files_sha256'},indent=2))
