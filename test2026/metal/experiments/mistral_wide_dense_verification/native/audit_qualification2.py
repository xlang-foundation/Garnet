"""Full-model B4/T4 exact accepted-span replay; deliberately no speed claim."""
from pathlib import Path
import copy,hashlib,json
R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def need(ok,why):
 if not ok:raise ValueError(why)
inputs=read(R/'input2.json');plan=read(R/'qualification-plan2.json');build=read(R/'build-audit2.json')
need(build['passed'] and not build['default_option_on'],'Build gate')
need(read(R/'qualification2-preflight.json')['passed'] and read(R/'qualification2-postflight.json')['passed'],'Source/model bindings')
for label in ['ordinary','verify','reverse']:need(read(R/(label+'-status2.json'))['returncode']==0,'Process failed')
reports=[read(R/(label+'-result2.json')) for label in ['ordinary','verify','reverse']]
def audit(reports):
 controls={};pairs=0;invalidpairs=0;totals=[]
 control=reports[0];need(control['mode']=='ordinary','Control arm')
 for report,mode in zip(reports,['ordinary','verify','reverse']):
  need(report['completed'] and report['mode']==mode,'Incomplete arm')
  need(report['inputs']==inputs and 'ORACLES' in report['scope'],'Fixture or oracle disclosure')
  need('native_dense_verify_b4t4_optional_v1' in report['backend'],'Wide native capability')
  need(sha(Path(report['library']))==build['library_sha256']['on'],'Native library')
  need(report['prepared']['verification_tokens']==4 and report['prepared']['target_graphs']==224 and report['prepared']['compact_families']==17,'Prepared profiles')
  need([j['label'] for j in report['jobs']]==['A','B','C','D'],'Request coverage')
  need(all(j['status']=='finished' and len(j['generated'])==j['limit'] and j['prefilled']==2048 for j in report['jobs']),'Budgets or terminal state')
  need(sum(len(j['generated']) for j in report['jobs'])==47,'Output count')
  need(report['cleanup']['metal_allocated_bytes']<=4194304,'Resources retained')
  for key in ['garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets']:need(report['cleanup'][key]==0,'Native owner retained')
  for record in report['records']:
   for row in record['rows']:
    need(len(row['states'])==32 and all(len(layer)==2 for layer in row['states']),'Full state coverage')
    for h in [v for layer in row['states'] for v in layer]+[row['hidden']]:need(len(h)==64 and all(c in '0123456789abcdef' for c in h),'State digest format')
  if mode=='ordinary':
   for record in report['records']:
    for row in record['rows']:
     key=row['label'],row['generated'];need(key not in controls,'Duplicate ordinary checkpoint');controls[key]=row
   need(len(controls)==47,'Ordinary every-token coverage');continue
  need(report['rejections']==plan['rejections'],'Native rejection coverage')
  for job,ordinary in zip(report['jobs'],control['jobs']):need(job['generated']==ordinary['generated'],'Output token mismatch')
  checked=0
  for record in report['records']:
   for row in record['rows']:
    expected=controls[row['label'],row['generated']]
    need(all(row[k]==expected[k] for k in ['row','epoch','status','states','hidden']),'Per-owner KV/hidden differs from ordinary at identical logical prefix')
    checked+=65;pairs+=65
  for before,after in [('before_invalid','after_invalid'),('before_stale','after_stale')]:
   a=next(x for x in report['records'] if x['label']==before);b=next(x for x in report['records'] if x['label']==after);need(a['rows']==b['rows'] and a['status']==b['status'],'Rejected API changed state');invalidpairs+=65*len(a['rows'])
  outputs={j['label']:j['generated'] for j in control['jobs']};owners={j['id']:j for j in report['jobs']};first=None;consumed=set();submitted=set();active=set();calls=0;zero_misses=True
  for tick in report['ticks']:
   for execution in tick['executions']:
    if execution['stage']!='decode':continue
    native=execution['native'];stats=native['verification'];need(stats['target_affine_misses']==0 and stats['target_indexed_decode_dispatches']==225,'FP32 indexed body/head route')
    expected_restores=64 if any(q['consumed']<q['submitted'] for q in native['requests']) else 0
    need(stats['target_kv_backups']==64 and stats['commit_kv_restores']==expected_restores,'Every rejected layer KV rollback; full acceptance needs no restore')
    need(stats['target_mpp_affine_dispatches']==stats['target_indexed_prefill_dispatches']==0,'FP16 prefill route selected during target verification')
    requests=native['requests'];need(requests==execution['completed'],'Native completion publication');need(len(requests)==len(execution['submitted']),'Submission coverage');calls+=1;active.add(len(requests));expected_first=[]
    byid={s['id']:s for s in execution['submitted']}
    for q in requests:
     job=owners[q['id']];s=byid[q['id']];before=s['generated_before'];ids=outputs[job['label']];tokens=s['tokens'];depth=s['depth'];need(1<=depth<=4 and len(tokens)==depth,'Depth')
     need(tokens[0]==q['known']==s['known']==ids[before-1],'Known token')
     expected=depth
     for t in range(1,depth):
      if tokens[t]!=ids[before+t-1]:expected=t;break
     need(q['submitted']==depth and q['consumed']==expected and q['emitted']==ids[before:before+expected],'Accepted prefix or corrected bonus')
     need(q['slot_before']==2048+before-1 and q['remaining_before']==job['limit']-before,'Native logical slot/budget')
     need(q['row']==job['row'] and q['epoch']==job['epoch'] and not q['eos'] and q['finished']==(before+expected==job['limit']),'Owner or terminal lease')
     consumed.add(expected);submitted.add(depth);expected_first.append((q['row'],expected))
    if first is None:first=[v for _,v in sorted(expected_first)]
  need(first==[4,1,2,3],'Asymmetric first B4/T4 acceptance')
  need(consumed==submitted=={1,2,3,4},'Accepted/submitted span coverage')
  need(active=={1,2,3,4},'Mixed occupancy and independent completion')
  totals.append(dict(mode=mode,target_calls=calls,state_digest_pairs=checked,accepted_lengths=sorted(consumed),submitted_lengths=sorted(submitted),active_requests=sorted(active)))
 return dict(passed=True,output_ids_per_arm=47,ordinary_every_token_checkpoints=47,state_digest_pairs_vs_ordinary=pairs,invalid_call_unchanged_digest_pairs=invalidpairs,arms=totals,full_kv_planes_and_hidden_exact=True,prepared_b4t4_core_qualified=True,compact_preparation_both_orders=True,oracle_proposals_for_correctness_only=True,default_option_on=False,cancel_refill_survivors_wide_qualified=False,prefix_retention_qualified=False,learned_helper_batching_qualified=False,performance_measured=False,selected_full_serving_comparison_promoted=False,goal_complete=False)
result=audit(reports);negative=[]
def reject(name,mutate):
 bad=copy.deepcopy(reports);mutate(bad)
 try:audit(bad)
 except (ValueError,KeyError,IndexError,TypeError):negative.append(name);return
 raise AssertionError('Accepted corruption '+name)
reject('wrong_output_id',lambda r:r[1]['jobs'][0]['generated'].__setitem__(2,999))
reject('wrong_kv_digest',lambda r:r[1]['records'][0]['rows'][0]['states'][0].__setitem__(0,'0'*64))
reject('wrong_hidden_digest',lambda r:r[2]['records'][0]['rows'][0].update(hidden='0'*64))
reject('wrong_epoch',lambda r:r[1]['records'][0]['rows'][0].update(epoch=999))
def first(r):return next(e for t in r[1]['ticks'] for e in t['executions'] if e['stage']=='decode')
reject('false_accepted_length',lambda r:first(r)['native']['requests'][0].update(consumed=3))
reject('false_budget',lambda r:first(r)['native']['requests'][0].update(remaining_before=999))
reject('missing_fp32_head',lambda r:first(r)['native']['verification'].update(target_indexed_decode_dispatches=224))
reject('missing_layer_rollback',lambda r:first(r)['native']['verification'].update(commit_kv_restores=63))
reject('missing_native_rejection',lambda r:r[1]['rejections'].pop())
result['negative_controls_rejected']=negative
result['files_sha256']={str(p):sha(p) for p in [R/'input2.json',R/'qualification-plan2.json',R/'worker1.py',R/'run_qualification2.py',R/'build-audit2.json']+[R/(label+'-result2.json') for label in ['ordinary','verify','reverse']]}
(R/'qualification-audit2.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
