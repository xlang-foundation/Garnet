"""Audit actual causal-helper stage rates; no HTTP or comparator speed admission."""
from pathlib import Path
import copy,hashlib,json,statistics
from audit_qualification1 import b1_check,need
R=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for buf in iter(lambda:f.read(8*1024*1024),b''):h.update(buf)
 return h.hexdigest()
QUAL={mode:read(R/f'b1-{mode}-result1.json') for mode in ['on','off']}
KEYS=['case_id','prompt_ids','generated_ids','slots','state_sha256','hidden_sha256']
def check(data,mode):
 need(data['experiment']=='bounded' and len(data['records'])==1,'Wrong measurement mode')
 row=data['records'][0];wrapped=copy.deepcopy(data);wrapped['experiment']='qualify';wrapped['records']=[QUAL[mode]['records'][0],row]
 b1_check(wrapped,mode)
 control=QUAL[mode]['records'][1]
 for k in KEYS:need(row[k]==control[k],'Qualified target state changed '+k)
 need(len(row['rounds'])==len(control['rounds']),'Target call count changed')
 for actual,expected in zip(row['rounds'],control['rounds']):
  for k in ['committed_before','slot','proposal','submitted_ids','emitted_ids','consumed','accepted_draft']:need(actual[k]==expected[k],'Target history changed '+k)
  for k in ['reason','prompt_ids','generated_ids','executed_draft_ids','prefill_chunks','cached_tokens']:need(actual['draft'].get(k)==expected['draft'].get(k),'Causal helper trace changed '+k)
 return dict(prefill_seconds=row['prefill_seconds'],decode_seconds=row['decode_seconds'],decode_tps=row['decode_tokens_per_second'],target_seconds=sum(q['target_seconds'] for q in row['rounds']),draft_seconds=sum(q['draft']['seconds'] for q in row['rounds']),accepted=sum(q['accepted_draft'] for q in row['rounds']),proposed=sum(len(q['proposal']) for q in row['rounds']),target_calls=len(row['rounds']))
def main():
 need(not (R/'measurement-audit1.json').exists(),'Audit overwrite')
 pre=read(R/'measurement1-preflight.json');need(pre['passed'] and read(R/'measurement1-postflight.json')['passed'] and not pre['hashes_during_collection'],'Incomplete collection')
 for name,h in pre['files_sha256'].items():need(sha(Path(name))==h,'Changed bound input '+name)
 plan=read(R/'measurement-plan1.json');last=0;pids=set();summaries=[];data={}
 for cohort,mode in plan['sequence']:
  label=f'cohort{cohort}-{mode}';status=read(R/(label+'-status1.json'));need(status['status']=='terminal' and status['returncode']==0 and status['started_ns']>=last and status['finished_ns']>status['started_ns'] and status['pid'] not in pids,'Nonserial/fresh/terminal measurement');last=status['finished_ns'];pids.add(status['pid'])
  d=read(R/(label+'-result1.json'));summary=check(d,mode);data[cohort,mode]=d;summaries.append(dict(cohort=cohort,mode=mode,**summary))
 ratios=[]
 for cohort in [1,2]:
  off=next(s for s in summaries if s['cohort']==cohort and s['mode']=='off');on=next(s for s in summaries if s['cohort']==cohort and s['mode']=='on')
  ratios.append(dict(cohort=cohort,decode_ratio=off['decode_seconds']/on['decode_seconds'],off_decode_tps=off['decode_tps'],on_decode_tps=on['decode_tps'],off_prefill_seconds=off['prefill_seconds'],on_prefill_seconds=on['prefill_seconds']))
 negative=[]
 def reject(name,fn):
  bad=copy.deepcopy(data[1,'on']);fn(bad)
  try:check(bad,'on')
  except (ValueError,KeyError,TypeError,IndexError):negative.append(name);return
  raise AssertionError('Corruption admitted: '+name)
 reject('false_decode_rate',lambda d:d['records'][0].update(decode_tokens_per_second=999.))
 reject('wrong_full_kv',lambda d:d['records'][0]['state_sha256'][0].__setitem__(0,'0'*64))
 reject('false_gqa_value_route',lambda d:next(q['metrics'] for q in d['records'][0]['rounds'] if q['metrics']).update(target_attention_gqa16_value_dispatches=0))
 reject('future_helper_prefix',lambda d:next(q['draft'] for q in d['records'][0]['rounds'] if q['draft'].get('prompt_ids'))['prompt_ids'].append(1))
 reject('changed_accepted_count',lambda d:d['records'][0]['rounds'][0].update(consumed=99))
 result=dict(passed=True,scope=__doc__,sequence=plan['sequence'],stages=summaries,paired_ratios=ratios,median_decode_tps={mode:statistics.median(s['decode_tps'] for s in summaries if s['mode']==mode) for mode in ['on','off']},native_stage_gate_passed=all(r['decode_ratio']>=plan['minimum_native_decode_ratio'] for r in ratios),negative_controls_rejected=negative,output_ids_exact_qualified_current_ordinary=True,full_kv_hidden_exact_qualified_current_ordinary=True,proposals_and_helper_trace_exact=True,serving_ttft_measured=False,matched_maximum_concurrency_profile=False,full_serving_comparison_promoted=False,coding_quality_qualified=False,no_comparator_execution=True,goal_complete=False,files_sha256={str(p):sha(p) for p in [R/'audit_measurement1.py',R/'measurement-plan1.json',R/'measurement1-preflight.json',R/'measurement1-postflight.json']+[R/f'cohort{c}-{mode}-result1.json' for c,mode in plan['sequence']]})
 (R/'measurement-audit1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='files_sha256'},indent=2))
if __name__=='__main__':main()
