"""Audit serial ABBA native causal-helper decoding for the existing fused/shared-KV combination."""
from pathlib import Path
import copy,hashlib,json,statistics as st
from audit_qualification1 import check as qualified,trace_equal,need
R=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
QUAL={mode:read(R/('b1-'+mode+'-result1.json')) for mode in ['off','on']}
def check(d,mode):
 need(d['experiment']=='bounded' and len(d['records'])==1,'measurement mode');row=d['records'][0];wrapped=copy.deepcopy(d);wrapped['experiment']='qualify';wrapped['records']=[QUAL[mode]['records'][0],row];qualified(wrapped,mode);trace_equal(row,QUAL[mode]['records'][1])
 return dict(decode_tps=row['decode_tokens_per_second'],decode_seconds=row['decode_seconds'],prefill_seconds=row['prefill_seconds'],draft_seconds=sum(q['draft']['seconds'] for q in row['rounds']),target_seconds=sum(q['target_seconds'] for q in row['rounds']),accepted=sum(q['accepted_draft'] for q in row['rounds']),proposed=sum(len(q['proposal']) for q in row['rounds']),target_calls=len(row['rounds']),actual_fused_blocks=sum(q['metrics'].get('target_fused_q4_decode_blocks',0) for q in row['rounds'] if q['metrics']))
def main():
 need(read(R/'measurement1-postflight.json')['passed'],'terminal gate');pre=read(R/'measurement1-preflight.json');need(pre['passed'] and not pre['hashes_during_collection'],'preflight binding')
 for n,h in pre['files_sha256'].items():need(sha(Path(n))==h,'changed input '+n)
 sequence=[(1,'off'),(1,'on'),(2,'on'),(2,'off')];rows=[];data={};last=0;files=[]
 for cohort,mode in sequence:
  name=f'cohort{cohort}-{mode}';status=read(R/(name+'-status1.json'));need(status['status']=='terminal' and status['returncode']==0 and status['started_ns']>=last,'fresh serial terminal engines');last=status['finished_ns'];d=read(R/(name+'-result1.json'));data[cohort,mode]=d;rows.append(dict(cohort=cohort,mode=mode,**check(d,mode)));files += [R/(name+'-result1.json'),R/(name+'-status1.json'),R/(name+'-command1.json')]
 ratios=[]
 for cohort in [1,2]:
  off=next(r for r in rows if r['cohort']==cohort and r['mode']=='off');on=next(r for r in rows if r['cohort']==cohort and r['mode']=='on');ratios.append(dict(cohort=cohort,decode_ratio=off['decode_seconds']/on['decode_seconds']))
 negative=[]
 def reject(name,mutate):
  d=copy.deepcopy(data[1,'on']);mutate(d)
  try:check(d,'on')
  except (ValueError,KeyError,TypeError,IndexError,StopIteration):negative.append(name);return
  raise AssertionError('accepted corruption '+name)
 reject('wrong rate',lambda d:d['records'][0].__setitem__('decode_tokens_per_second',999.))
 reject('wrong target KV',lambda d:d['records'][0]['state_sha256'][0].__setitem__(0,'0'*64))
 reject('missing fused route',lambda d:next(q['metrics'] for q in d['records'][0]['rounds'] if len(q['proposal'])==3).__setitem__('target_fused_q4_decode_blocks',0))
 reject('future helper prefix',lambda d:next(q['draft'] for q in d['records'][0]['rounds'] if q['draft'].get('prompt_ids'))['prompt_ids'].append(1))
 reject('false accepted count',lambda d:d['records'][0]['rounds'][0].__setitem__('accepted_draft',99))
 files += [R/n for n in ['audit_measurement1.py','measurement1-preflight.json','measurement1-postflight.json','qualification-audit1.json','measurement-plan1.json']]
 report=dict(passed=True,stages=rows,paired_ratios=ratios,median_decode_tps={m:st.median(r['decode_tps'] for r in rows if r['mode']==m) for m in ['off','on']},native_stage_gate_passed=all(r['decode_ratio']>=1.03 for r in ratios),target_output_ids=512,full_target_state_and_proposals_exact_qualified=True,negative_controls_rejected=negative,no_http_ttft_throughput_or_comparator_claim=True,no_comparator_execution=True,goal_complete=False,files_sha256={str(p):sha(p) for p in files});p=R/'measurement-audit1.json';need(not p.exists(),'report exists');p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='files_sha256'},indent=2))
if __name__=='__main__':main()
