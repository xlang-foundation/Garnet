"""Audit exact target states and causal drafts for the compiled fused/shared-KV combination."""
from pathlib import Path
import copy,hashlib,importlib.util,json,sys
R=Path(__file__).resolve().parent;M=R.parent;N=M/'mistral-gqa16-value-native1';Q=M/'mistral-causal-batch4-helper1';sys.path.insert(0,str(R))
spec=importlib.util.spec_from_file_location('fused_bounded_qualification',R/'audit_bounded1.py');bounded=importlib.util.module_from_spec(spec);spec.loader.exec_module(bounded);bounded.ROOT=R;bounded.M=M
read=lambda p:json.loads(p.read_text())
def need(v,why):
 if not v:raise ValueError(why)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
KEYS=['case_id','prompt_ids','generated_ids','slots','state_sha256','hidden_sha256'];REF=read(N/'b1-on-result1.json')['records'][0]
def check(d,mode):
 need(d['inputs']==read(R/'input1.json'),'fixture change');need(d['fused_decode_enabled']==(mode=='on') and ('native_fused_q4_decode_rows4_default_v1' in d['backend'])==(mode=='on'),'compiled fused capability')
 need(d['gqa_scores_enabled'] is True and d['gqa_values_enabled'] is True and d['split_scores_enabled'] is True,'qualified attention capability changed')
 lib=str((R if mode=='on' else Q)/'build-on1/libgarnet.dylib');need(d['library']==lib,'native library binding');summary=bounded.check(d);selected=0;fallback=0
 for row in d['records']:
  for k in KEYS:need(row[k]==REF[k],'frozen ordinary output/full state '+k)
 for step in d['records'][1]['rounds']:
  if not step['metrics']:continue
  m=step['metrics'];expected=32 if mode=='on' and len(step['proposal'])==3 else 0
  need(m.get('target_fused_q4_decode_blocks',0)==expected,'actual fused route');selected+=bool(expected);fallback+=not bool(expected)
  need(m['target_attention_gqa16_score_dispatches']==m['target_attention_gqa16_value_dispatches']==32 and m['target_attention_vector4_dispatches']==m['target_attention_split_score_dispatches']==0,'shared-KV attention route')
 need(mode=='off' or selected>0,'fusion never executed')
 return dict(summary=summary,selected_verifier_frames=selected,fallback_verifier_frames=fallback,full_state_digest_pairs=130,output_ids=256)
def trace_equal(a,b):
 need(len(a['rounds'])==len(b['rounds']),'causal round count')
 for x,y in zip(a['rounds'],b['rounds']):
  for k in ['committed_before','slot','proposal','submitted_ids','emitted_ids','consumed','accepted_draft']:need(x[k]==y[k],'target trace changed '+k)
  for k in ['reason','prompt_ids','generated_ids','executed_draft_ids','prefill_chunks','cached_tokens']:need(x['draft'].get(k)==y['draft'].get(k),'helper trace changed '+k)
def main():
 need(read(R/'qualification1-postflight.json')['passed'],'terminal collection gate');pre=read(R/'qualification1-preflight.json');need(pre['passed'],'preflight')
 for n,h in pre['files_sha256'].items():need(sha(Path(n))==h,'source/model binding '+n)
 data={};reports={};last=0
 for mode in ['off','on']:
  status=read(R/('b1-'+mode+'-status1.json'));need(status['status']=='terminal' and status['returncode']==0 and status['started_ns']>=last,'serial fresh engines');last=status['finished_ns'];d=read(R/('b1-'+mode+'-result1.json'));reports[mode]=check(d,mode);data[mode]=d
 trace_equal(data['on']['records'][1],data['off']['records'][1]);negative=[]
 def reject(name,mutate):
  d=copy.deepcopy(data['on']);mutate(d)
  try:check(d,'on')
  except (ValueError,KeyError,TypeError,IndexError,StopIteration):negative.append(name);return
  raise AssertionError('accepted corruption '+name)
 reject('wrong output',lambda d:d['records'][1]['generated_ids'].__setitem__(0,-1))
 reject('wrong full KV',lambda d:d['records'][1]['state_sha256'][0].__setitem__(0,'0'*64))
 reject('wrong final hidden',lambda d:d['records'][1].__setitem__('hidden_sha256','0'*64))
 reject('false compiled fusion',lambda d:d.__setitem__('fused_decode_enabled',False))
 reject('missing actual fusion',lambda d:next(q['metrics'] for q in d['records'][1]['rounds'] if len(q['proposal'])==3).__setitem__('target_fused_q4_decode_blocks',0))
 reject('false shared value route',lambda d:next(q['metrics'] for q in d['records'][1]['rounds'] if q['metrics']).__setitem__('target_attention_gqa16_value_dispatches',0))
 reject('future helper prefix',lambda d:next(q['draft'] for q in d['records'][1]['rounds'] if q['draft'].get('prompt_ids'))['prompt_ids'].append(1))
 reject('resident host snapshot',lambda d:next(q['draft']['reconcile'] for q in d['records'][1]['rounds'] if q['draft'].get('reconcile')).__setitem__('host_snapshot_bytes',1))
 reject('wrong acceptance',lambda d:d['records'][1]['rounds'][0].__setitem__('consumed',0))
 reject('missing KV rollback',lambda d:next(q['metrics'] for q in d['records'][1]['rounds'] if q['metrics'] and q['consumed']<len(q['proposal'])+1).__setitem__('commit_kv_restores',0))
 reject('false token speed',lambda d:d['records'][1].__setitem__('decode_tokens_per_second',999.))
 reject('live GPU owner',lambda d:d['cleanup'].__setitem__('garnet_live_matrix_owners',1))
 files=[R/n for n in ['audit_qualification1.py','audit_bounded1.py','audit_common1.py','input1.json','plan1.json','build-audit1.json','qualification1-preflight.json','qualification1-postflight.json','b1-off-result1.json','b1-on-result1.json']]
 report=dict(passed=True,fresh_engines=2,output_ids=512,full_state_digest_pairs=260,reports=reports,causal_draft_proposals_and_acceptance_exact=True,negative_controls_rejected=negative,performance_admitted=False,http_or_quality_admitted=False,no_comparator_execution=True,goal_complete=False,files_sha256={str(p):sha(p) for p in files})
 p=R/'qualification-audit1.json';need(not p.exists(),'report exists');p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='files_sha256'},indent=2))
if __name__=='__main__':main()
