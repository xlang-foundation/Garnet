"""Audit native split verification in a full2K target against current ordinary."""
from pathlib import Path
import copy,hashlib,json,sys
R=Path(__file__).resolve().parent;M=R.parent;B=M/'mistral-bounded-draft-long1'
sys.path.insert(0,str(B));from audit_qualification1 import check as old_check,need
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
PLAN=read(R/'stage-plan1.json');REFERENCE=read(B/'qualification1-result.json')['records'][0]
def check(data,mode):
 need(data['split_scores_enabled']==(mode=='on'),'compiled split route flag differs')
 need(data['library']==PLAN['libraries'][mode]+'/libgarnet.dylib','wrong native library')
 need(('native_mistral_verification_split_scores128_default_v1' in data['backend'])==(mode=='on'),'wrong native feature tag')
 summary=old_check(data)
 for row in data['records']:
  for k in ['case_id','prompt_ids','generated_ids','slots','state_sha256','hidden_sha256']:need(row[k]==REFERENCE[k],'current ordinary state mismatch '+k)
 for q in data['records'][1]['rounds']:
  if q['metrics']:need(q['metrics']['target_attention_split_score_dispatches']==(32 if mode=='on' else 0),'split verification did not run exactly32 layers')
 return summary

def main():
 need(not (R/'qualification-audit1.json').exists(),'audit overwrite')
 pre=read(R/'qualification1-preflight.json');need(pre['passed'] and read(R/'qualification1-postflight.json')['passed'],'unfinished collection')
 for n,h in pre['files_sha256'].items():need(sha(Path(n))==h,'changed admitted input '+n)
 summaries=[];data_by={};last=0;pids=set()
 for mode in PLAN['qualification_modes']:
  status=read(R/f'qualification-{mode}1-status.json');need(status['status']=='terminal' and status['returncode']==0 and status['started_ns']>=last and status['finished_ns']>status['started_ns'] and status['pid'] not in pids,'nonserial/fresh/terminal qualification');last=status['finished_ns'];pids.add(status['pid'])
  data=read(R/f'qualification-{mode}1-result.json');data_by[mode]=data;summaries.append(dict(mode=mode,summary=check(data,mode)))
 left=data_by['on']['records'][1]['rounds'];right=data_by['off']['records'][1]['rounds'];need(len(left)==len(right),'different target call count')
 for a,b in zip(left,right):
  for k in ['committed_before','slot','proposal','submitted_ids','emitted_ids','consumed','accepted_draft']:need(a[k]==b[k],'ON/OFF target history differs '+k)
  for k in ['reason','prompt_ids','generated_ids','executed_draft_ids','prefill_chunks','cached_tokens']:need(a['draft'].get(k)==b['draft'].get(k),'ON/OFF helper history differs '+k)
 rejected=[]
 mutations={
  'wrong_output':lambda d:d['records'][1]['generated_ids'].__setitem__(0,-1),
  'wrong_kv':lambda d:d['records'][1]['state_sha256'][0].__setitem__(0,'0'*64),
  'wrong_hidden':lambda d:d['records'][1].update(hidden_sha256='0'*64),
  'false_split_dispatch':lambda d:next(q['metrics'] for q in d['records'][1]['rounds'] if q['metrics']).update(target_attention_split_score_dispatches=0),
  'false_compiled_feature':lambda d:d.update(split_scores_enabled=False),
  'wrong_library':lambda d:d.update(library=PLAN['libraries']['off']+'/libgarnet.dylib'),
  'future_prefix':lambda d:next(q['draft'] for q in d['records'][1]['rounds'] if q['draft'].get('prompt_ids'))['prompt_ids'].append(1),
  'missing_rollback':lambda d:next(q['metrics'] for q in d['records'][1]['rounds'] if q['metrics'] and q['consumed']<len(q['proposal'])+1).update(commit_kv_restores=0),
  'false_resident_lcp':lambda d:next(q['draft']['reconcile'] for q in d['records'][1]['rounds'] if q['draft'].get('reconcile')).update(common_tokens=9999),
  'live_owner':lambda d:d['cleanup'].update(garnet_live_matrix_owners=1),
 }
 for name,fn in mutations.items():
  bad=copy.deepcopy(data_by['on']);fn(bad)
  try:check(bad,'on')
  except (ValueError,KeyError,TypeError,IndexError):rejected.append(name)
  else:raise ValueError('negative control admitted '+name)
 splitframes=sum(q['metrics'] is not None for q in left)
 result=dict(passed=True,scope=__doc__,target_prompt_tokens=2048,output_tokens_per_mode=128,records=4,full_current_ordinary_state_digest_pairs=260,on_split_verification_frames=splitframes,on_split_layer_dispatches=splitframes*32,off_split_layer_dispatches=0,all_output_ids_full_kv_hidden_exact_current_ordinary=True,proposals_commits_and_helper_history_exact_on_off=True,summary=summaries,negative_controls_rejected=rejected,files_sha256={str(p):sha(p) for p in [R/'audit_qualification1.py',R/'qualification-on1-result.json',R/'qualification-off1-result.json',R/'qualification1-preflight.json',R/'stage-plan1.json']},qualification_order_not_performance_admission=True,matched_http_speedup_admitted=False,coding_quality_qualified=False,goal_complete=False)
 (R/'qualification-audit1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='files_sha256'},indent=2))
if __name__=='__main__':main()
