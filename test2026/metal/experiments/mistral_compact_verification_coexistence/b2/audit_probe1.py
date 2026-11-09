"""Audit B2 compact spans interleaved with an independent native decoder row."""
from pathlib import Path
import copy,hashlib,json
R=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
def need(v,m):
 if not v:raise ValueError(m)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def check(a,b):
 for d,mode in [(a,'ordinary'),(b,'verify')]:
  need(d['completed'] and d['mode']==mode and d['inputs']==read(R/'probe-input1.json') and d['inputs']['stops']==[],'wrong probe mode/inputs')
  need(len(d['jobs'])==3 and d['prepared']['compact_families']==17 and d['prepared']['maximum_compact_chunk']==512,'probe family/owner count')
  by={j['label']:j for j in d['jobs']};need(set(by)=={'S','P','Q'} and [by[x]['row'] for x in ['S','P','Q']]==[0,1,2],'probe placement')
  for name,n,limit in [('S',64,8),('P',2048,2),('Q',2048,2)]:need(len(by[name]['prompt_ids'])==by[name]['prefilled']==n and by[name]['status']=='finished' and len(by[name]['generated'])==by[name]['limit']==limit,'probe budget/progress')
  end=d['cleanup'];need(end['metal_allocated_bytes']<=4194304 and all(end[k]==0 for k in ['garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets']),'probe owner cleanup')
 need(a['jobs']==b['jobs'] and len(a['ticks'])==len(b['ticks']),'probe output/tick parity')
 routes=set();mixed=0;calls=0
 for x,y in zip(a['ticks'],b['ticks']):
  need(x['plan']==y['plan'] and len(x['executions'])==len(y['executions'])==len(x['plan']['batches']),'probe scheduler differs')
  for p,r,s in zip(x['plan']['batches'],x['executions'],y['executions']):
   need(r['stage']==s['stage']==p['stage'],'stage mismatch')
   if p['stage']=='prefill':
    need(r==s,'compact profile route differs');route=(r['physical_chunk'],r['compact_rows'],r['compact_first']);routes.add(route)
    if r['compact_rows']==2:need([q['row'] for q in p['requests']]==[1,2] and r['compact_first']==1,'wrong B2 span')
    if r['compact_rows']==2 and any(z['stage']=='decode' for z in x['plan']['batches']):mixed+=1
    continue
   calls+=1;need(r['compact_rows']==s['compact_rows']==0 and r['native'] is None,'decode entered compact prefill');native=s['native'];need(native is not None and len(native['requests'])==len(p['requests'])==len(r['completed']),'native verification row count')
   m=native['verification'];need(m['target_affine_dispatches']==225 and m['target_affine_misses']==0 and m['target_attention_dispatches']==32 and m['target_kv_backups']==64 and m['commit_kv_restores']==0,'probe FP32 route')
   byid={j['native_request']:j for j in b['jobs']}
   for planned,old,q in zip(p['requests'],r['completed'],native['requests']):
    j=byid[q['id']];need(q['id']==old['id']==planned['id'] and q['row']==old['row']==planned['row'] and q['epoch']==old['epoch']==planned['epoch'],'probe lease identity')
    need(q['known']==planned['token_id'] and q['slot_before']==planned['slot'] and q['remaining_before']==j['limit']-planned['generated_tokens'] and q['submitted']==q['consumed']==1,'probe native budget/slot')
    need(q['emitted']==old['emitted']==j['generated'][planned['generated_tokens']:planned['generated_tokens']+1] and q['finished']==old['finished'] and q['eos'] is False,'probe emission/terminal state')
 need({(128,2,1),(256,2,1)}<=routes and mixed>=6,'B2 mixed/long profile missing')
 aw={w['label']:w for w in a['witnesses']};bw={w['label']:w for w in b['witnesses']};need(set(aw)=={'first_B2_prefill','after_B2_verification','all_done'} and set(bw)==set(aw)|{'before_invalid_mixed','after_invalid_mixed'},'probe witness coverage')
 for key in aw:need(aw[key]==bw[key],'complete B2/survivor state differs '+key)
 one=copy.deepcopy(bw['before_invalid_mixed']);two=copy.deepcopy(bw['after_invalid_mixed']);one.pop('label');two.pop('label');need(one==two,'invalid mixed call altered state')
 need(set(b['rejections'])=={'mixed_unscheduled_row','wrong_known_mixed','stale_tick_replay'} and not a['rejections'],'probe rejected call coverage')
 need(all(aw['all_done']['status'][k]==0 for k in ['active','queued','tracked','inflight']),'probe retained native requests')
 return dict(output_ids=12,full_four_row_state_digest_pairs=780,rejected_call_state_digest_pairs=260,native_verification_calls=calls,b2_mixed_ticks=mixed,actual_routes=[list(x) for x in sorted(routes)])
def main():
 need(not (R/'probe-audit1.json').exists(),'audit overwrite');pre=read(R/'probe1-preflight.json');need(pre['passed'] and read(R/'probe1-postflight.json')['passed'],'probe incomplete')
 for n,h in pre['files_sha256'].items():need(sha(Path(n))==h,'admitted file changed '+n)
 last=read(R/'reverse-status1.json')['finished_ns'];pids=set()
 for mode in ['ordinary','verify']:
  s=read(R/('probe-'+mode+'-status1.json'));need(s['status']=='terminal' and s['returncode']==0 and s['pid'] not in pids and s['finished_ns']>s['started_ns']>=last,'probe nonserial/fresh/terminal');last=s['finished_ns'];pids.add(s['pid'])
 a,b=(read(R/('probe-'+m+'-result1.json')) for m in ['ordinary','verify']);summary=check(a,b);rejected=[]
 for name,fn in [
  ('survivor_hidden',lambda d:next(w for w in d['witnesses'] if w['label']=='after_B2_verification')['hidden'].__setitem__(0,'0'*64)),
  ('b2_cache',lambda d:next(w for w in d['witnesses'] if w['label']=='first_B2_prefill')['states'][0][2].__setitem__(1,'0'*64)),
  ('wrong_native_budget',lambda d:next(e for t in d['ticks'] for e in t['executions'] if e['stage']=='decode')['native']['requests'][0].update(remaining_before=999)),
  ('wrong_compact_span',lambda d:next(e for t in d['ticks'] for e in t['executions'] if e['stage']=='prefill' and e['compact_rows']==2).update(compact_first=0)),
  ('wrong_target_id',lambda d:d['jobs'][0]['generated'].__setitem__(0,-1))]:
  bad=copy.deepcopy(b);fn(bad)
  try:check(a,bad)
  except (ValueError,KeyError,TypeError,IndexError):rejected.append(name)
  else:raise ValueError('corruption admitted '+name)
 report=dict(passed=True,scope=__doc__,summary=summary,all_output_ids_full_kv_hidden_exact_current_ordinary=True,b2_compact_interleave_qualified=True,stop_policy='Empty stop list only in this geometry probe; output budgets govern termination. Primary qualification uses EOS2.',negative_controls_rejected=rejected,files_sha256={str(p):sha(p) for p in [R/'audit_probe1.py',R/'probe-ordinary-result1.json',R/'probe-verify-result1.json',R/'probe1-preflight.json']},performance_measured=False,no_comparator_execution=True,c4t4_qualified=False,goal_complete=False)
 (R/'probe-audit1.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='files_sha256'},indent=2))
if __name__=='__main__':main()
