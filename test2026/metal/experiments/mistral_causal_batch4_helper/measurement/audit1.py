"""Audit serial native ABBA burst timing; do not substitute for matched HTTP results."""
from pathlib import Path
import copy,hashlib,json,math,statistics,sys
from tokenizers import Tokenizer
R=Path(__file__).resolve().parent;Q=R.parent/'mistral-causal-batch4-helper1'
sys.path.insert(0,str(R));from audit_trace1 import audit,target_run
read=lambda p:json.loads(p.read_text())
def need(ok,why):
 if not ok:raise ValueError(why)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def envelope(engine,wave):
 return dict(completed=engine['completed'],mode=engine['mode'],inputs=engine['inputs'],backend=engine['backend'],library=engine['library'],prepared=engine['prepared'],cleanup=engine['cleanup'],**wave)
def timings(d):
 rows=[]
 for w in d['waves']:
  need(w['phase']==('cold' if w['wave']==0 else 'warm'),'wave phase')
  start=w['began_monotonic'];end=w['ended_monotonic'];need(math.isfinite(start) and math.isfinite(end) and end>start,'burst timer')
  need(sum(len(j['generated']) for j in w['jobs'])==512,'burst output count')
  aggregate=512/(end-start);need(w['aggregate_tokens_per_second']==aggregate,'aggregate metric')
  latency=[];speeds=[]
  for j,m in zip(w['jobs'],w['per_request']):
   first=j['first_token_monotonic'];last=j['finished_monotonic'];need(start<first<last<=end and len(j['generated'])==128,'request timer bounds')
   need(m['label']==j['label'] and m['native_ttft_seconds']==first-start and m['decode_tokens_per_second']==127/(last-first),'request metrics')
   latency.append(first-start);speeds.append(127/(last-first))
  rows.append(dict(wave=w['wave'],phase=w['phase'],native_first_prediction_seconds=statistics.median(latency),decode_tokens_per_second_per_request=statistics.median(speeds),native_aggregate_tokens_per_second=aggregate))
 return rows
def main():
 plan=read(R/'plan1.json');inputs=read(R/'input1.json');need(read(R/'postflight1.json')['passed'],'collection postflight');pre=read(R/'preflight1.json')
 for n,h in pre['files_sha256'].items():need(sha(Path(n))==h,'preflight binding '+n)
 need(read(Q/'productive-audit1.json')['passed'],'full-state qualification')
 engines=[];paths=[];previous_finished=0
 for i,mode in enumerate(plan['sequence']):
  label=f'engine{i+1}-{mode}';p=R/(label+'-result1.json');d=read(p);status=read(R/(label+'-status1.json'));need(status['status']=='terminal' and status['returncode']==0 and status['started_ns']>previous_finished,'serial process order');previous_finished=status['finished_ns']
  need(d['completed'] and d['mode']==mode and d['inputs']==inputs and [w['wave'] for w in d['waves']]==[0,1,2],'engine identity/waves');engines.append(d);paths.extend([p,R/(label+'-status1.json'),R/(label+'-command1.json')])
 ttok=Tokenizer.from_file(inputs['target_model']+'/tokenizer.json');dtok=Tokenizer.from_file(inputs['draft_model']+'/tokenizer.json')
 control=engines[0];control_hashes=[]
 for d in engines:
  for w in d['waves']:
   actual=target_run(envelope(d,w),inputs,d['mode'])
   ref=target_run(envelope(control,control['waves'][w['wave']]),inputs,'ordinary');need(actual['histories']==ref['histories'],'fresh ordinary token equivalence')
   need(all(j['generated'][:32]==next(x for x in read(Q/'ordinary-target-result1.json')['jobs'] if x['label']==j['label'])['generated'] for j in w['jobs']),'qualified prefix equivalence')
   control_hashes.append(hashlib.sha256(json.dumps(actual['histories'],sort_keys=True).encode()).hexdigest())
 need(len(set(control_hashes))==1,'all burst outputs equal')
 audits=[]
 for index in [1,2]:
  offset=0;d=engines[index]
  for w in d['waves']:
   report=audit(envelope(control,control['waves'][w['wave']]),envelope(d,w),inputs,ttok,dtok,offset);offset+=report['helper_reconciliation_calls'];audits.append(dict(engine=index+1,wave=w['wave'],**report))
 rows=[dict(engine=i+1,mode=d['mode'],waves=timings(d)) for i,d in enumerate(engines)]
 metrics=['native_first_prediction_seconds','decode_tokens_per_second_per_request','native_aggregate_tokens_per_second'];warm={mode:{k:statistics.median(row[k] for engine in rows if engine['mode']==mode for row in engine['waves'] if row['phase']=='warm') for k in metrics} for mode in ['ordinary','productive']}
 ratios=[]
 for c,p in [(rows[0],rows[1]),(rows[3],rows[2])]:
  cm={k:statistics.median(w[k] for w in c['waves'] if w['phase']=='warm') for k in metrics};pm={k:statistics.median(w[k] for w in p['waves'] if w['phase']=='warm') for k in metrics};ratios.append(dict(control_engine=c['engine'],productive_engine=p['engine'],decode=pm[metrics[1]]/cm[metrics[1]],aggregate=pm[metrics[2]]/cm[metrics[2]],first_prediction_control_over_productive=cm[metrics[0]]/pm[metrics[0]]))
 negative=[];o=envelope(control,control['waves'][0]);c=envelope(engines[1],engines[1]['waves'][0])
 def reject(name,fn):
  bad=copy.deepcopy(c);fn(bad)
  try:audit(o,bad,inputs,ttok,dtok)
  except (ValueError,KeyError,IndexError,TypeError,StopIteration):negative.append(name);return
  raise AssertionError('accepted corruption '+name)
 reject('published target output',lambda d:d['jobs'][0]['generated'].__setitem__(0,1))
 reject('future helper prefix',lambda d:d['helper_rounds'][0]['traces'][0]['generated_before'].append(1))
 reject('helper epoch',lambda d:d['helper_rounds'][0]['traces'][0].__setitem__('helper_epoch',0))
 reject('helper prompt token',lambda d:d['helper_rounds'][0]['traces'][0]['prompt_ids'].__setitem__(0,0))
 reject('helper snapshot copy',lambda d:d['helper_rounds'][0]['traces'][0]['reconcile'].__setitem__('host_snapshot_bytes',1))
 reject('helper call omission',lambda d:d['helper_calls'].pop())
 reject('helper call active mask',lambda d:d['helper_calls'][0]['active'].__setitem__(0,0.))
 reject('foreign runtime',lambda d:d.__setitem__('library','/tmp/foreign.dylib'))
 # Keep timer validation independent of token/trace validation.
 bad=copy.deepcopy(engines[1]);bad['waves'][1]['aggregate_tokens_per_second']*=2
 try:timings(bad)
 except ValueError:negative.append('inflated aggregate speed')
 else:raise AssertionError('accepted timer corruption')
 report=dict(passed=True,scope='Native C4 burst; original 2K prompts and128 outputs/request; no HTTP/SSE transport.',rows=rows,warm_medians=warm,paired_ratios=ratios,causal_trace_audits=audits,total_exact_target_ids=12*512,all_burst_output_sha256=control_hashes[0],negative_controls_rejected=negative,full_state_evidence=str(Q/'productive-audit1.json'),full_state_capture_in_timed_collection=False,matched_http_comparison_admitted=False,vllm_sglang_results_unchanged=True,no_comparator_execution=True,quality_suite_passed=False,thermal_os_conditions='Uncontrolled observational measurements.',goal_complete=False)
 paths.extend([R/n for n in ['audit1.py','audit_trace1.py','prepare_auditor1.py','trace-audit-adaptation1.patch','trace-audit-preparation1.json','preflight1.json','postflight1.json','plan1.json','preparation1.json']]);report['files_sha256']={str(p):sha(p) for p in paths}
 p=R/'audit1.json';need(not p.exists(),'fresh audit destination');p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['passed','warm_medians','paired_ratios','total_exact_target_ids','negative_controls_rejected','matched_http_comparison_admitted']},indent=2))
if __name__=='__main__':main()
