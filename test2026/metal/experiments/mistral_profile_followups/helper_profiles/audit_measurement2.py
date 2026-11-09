"""Audit opposite-order native helper-profile sweeps; all committed target work stays fixed."""
from pathlib import Path
import copy,hashlib,json,statistics as st
from audit_qualification1 import check as qualified,need
R=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
CAPS=[16,32,64,128,256];QUAL={cap:read(R/(f'cap{cap}-result1.json')) for cap in CAPS}
def check(d,cap):
 need(d['experiment']=='bounded' and len(d['records'])==1,'bounded native measurement mode');row=d['records'][0];wrapped=copy.deepcopy(d);wrapped['experiment']='qualify';wrapped['records']=[QUAL[cap]['records'][0],row];qualified(wrapped,cap,require_samples=False);control=QUAL[cap]['records'][1];need(len(row['rounds'])==len(control['rounds']),'same-profile target call count')
 for x,y in zip(row['rounds'],control['rounds']):
  for k in ['committed_before','slot','proposal','submitted_ids','emitted_ids','consumed','accepted_draft']:need(x[k]==y[k],'same-profile target trace '+k)
  for k in ['reason','prompt_ids','generated_ids','executed_draft_ids','prefill_chunks','cached_tokens']:need(x['draft'].get(k)==y['draft'].get(k),'same-profile helper trace '+k)
  need(not x['draft'].get('prefill_samples'),'qualification-only diagnostics inside timed generation')
  if x['metrics']:
   m=x['metrics'];need(m['target_attention_gqa16_score_dispatches']==m['target_attention_gqa16_value_dispatches']==32 and m['target_attention_vector4_dispatches']==m['target_attention_split_score_dispatches']==0,'qualified shared-KV target route')
 return dict(decode_tps=row['decode_tokens_per_second'],decode_seconds=row['decode_seconds'],target_prefill_seconds=row['prefill_seconds'],draft_seconds=sum(q['draft']['seconds'] for q in row['rounds']),target_seconds=sum(q['target_seconds'] for q in row['rounds']),accepted=sum(q['accepted_draft'] for q in row['rounds']),proposed=sum(len(q['proposal']) for q in row['rounds']),target_calls=len(row['rounds']),helper_graphs=d['prepared']['draft_graphs'],prepared_metal_bytes=d['prepared']['memory']['metal_allocated_bytes'])
def main():
 need(read(R/'measurement2-postflight.json')['passed'],'terminal collection gate');pre=read(R/'measurement2-preflight.json');need(pre['passed'] and not pre['hashes_during_collection'],'preflight')
 for n,h in pre['files_sha256'].items():need(sha(Path(n))==h,'changed bound input '+n)
 rows=[];data={};last=0;files=[]
 for order,caps in [('forward',CAPS),('reverse',list(reversed(CAPS)))]:
  for cap in caps:
   name=f'clean2-{order}-cap{cap}';status=read(R/(name+'-status1.json'));need(status['status']=='terminal' and status['returncode']==0 and status['started_ns']>=last,'fresh serial engines');last=status['finished_ns'];d=read(R/(name+'-result1.json'));data[order,cap]=d;rows.append(dict(cohort=order,cap=cap,**check(d,cap)));files += [R/(name+'-result1.json'),R/(name+'-status1.json'),R/(name+'-command1.json')]
 ratios=[]
 for order in ['forward','reverse']:
  base=next(r for r in rows if r['cohort']==order and r['cap']==16)
  for cap in CAPS:
   r=next(r for r in rows if r['cohort']==order and r['cap']==cap);ratios.append(dict(cohort=order,cap=cap,decode_ratio_over_cap16=base['decode_seconds']/r['decode_seconds']))
 negative=[]
 def reject(name,mutate):
  d=copy.deepcopy(data['forward',256]);mutate(d)
  try:check(d,256)
  except (ValueError,KeyError,TypeError,IndexError,StopIteration):negative.append(name);return
  raise AssertionError('accepted corruption '+name)
 reject('false token speed',lambda d:d['records'][0].__setitem__('decode_tokens_per_second',999.))
 reject('wrong target KV',lambda d:d['records'][0]['state_sha256'][0].__setitem__(0,'0'*64))
 reject('wrong helper profile',lambda d:d.__setitem__('helper_prefill_maximum',16))
 reject('future helper prefix',lambda d:next(q['draft'] for q in d['records'][0]['rounds'] if q['draft'].get('prompt_ids'))['prompt_ids'].append(1))
 reject('wrong prepared family',lambda d:d['prepared'].__setitem__('draft_graphs',96))
 reject('false shared-key route',lambda d:next(q['metrics'] for q in d['records'][0]['rounds'] if q['metrics']).__setitem__('target_attention_gqa16_score_dispatches',0))
 reject('wrong accepted token count',lambda d:d['records'][0]['rounds'][0].__setitem__('accepted_draft',99))
 medians={str(c):st.median(r['decode_tps'] for r in rows if r['cap']==c) for c in CAPS};best=max(CAPS,key=lambda c:medians[str(c)])
 report=dict(passed=True,collection_classification='Confirmation after the reported external archive terminated; same worker/library/fixtures as the exploratory first sweep.',stages=rows,paired_ratios=ratios,median_native_decode_tps=medians,fastest_observed_cap=best,target_ids_exact_qualified=1280,full_target_state_and_per_profile_proposals_exact=True,qualification_diagnostics_disabled_inside_timed_generation=True,negative_controls_rejected=negative,no_http_ttft_throughput_or_comparator_claim=True,no_comparator_execution=True,goal_complete=False)
 files += [R/n for n in ['audit_measurement2.py','measurement-plan2.json','measurement2-preflight.json','measurement2-postflight.json','qualification-audit2.json']];report['files_sha256']={str(p):sha(p) for p in files};p=R/'measurement-audit2.json';need(not p.exists(),'report exists');p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='files_sha256'},indent=2))
if __name__=='__main__':main()
