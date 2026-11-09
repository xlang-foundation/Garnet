"""Verify the corrected cache declaration, unchanged mathematical code, and a fresh native model trace."""
from pathlib import Path
import ast,copy,hashlib,json
from audit_qualification1 import check,need
R=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def functions(text):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
def audit(d):
 need(d['helper_prefix_profile_id']=='B1D512T64','cache profile declaration differs from compiled bound');check(d,64);old=read(R/'cap64-result1.json')
 for left,right in zip(d['records'],old['records']):
  for k in ['case_id','mode','prompt_ids','generated_ids','slots','state_sha256','hidden_sha256']:need(left[k]==right[k],'identity change altered target '+k)
 a=d['records'][1]['rounds'];b=old['records'][1]['rounds'];need(len(a)==len(b),'target call count')
 for x,y in zip(a,b):
  for k in ['committed_before','slot','proposal','submitted_ids','emitted_ids','consumed','accepted_draft']:need(x[k]==y[k],'target proposal/commit changed')
  for k in ['reason','prompt_ids','generated_ids','executed_draft_ids','prefill_chunks','cached_tokens']:need(x['draft'].get(k)==y['draft'].get(k),'helper causal history changed')
 return True
def main():
 need(read(R/'identity3-postflight.json')['passed'] and read(R/'identity64-status3.json')['returncode']==0,'terminal correctness trial')
 for p,h in read(R/'identity3-preflight.json')['files_sha256'].items():need(sha(Path(p))==h,'bound source/model changed')
 old=(R/'b1_worker1.py').read_text();new=(R/'b1_worker2.py').read_text();a=functions(old);b=functions(new);names=set(a)-{'save'};need(set(a)==set(b) and all(a[n]==b[n] for n in names),'mathematical helper/target functions changed')
 need("'B1D'+str(inputs.get('draft_capacity',inputs['capacity']))+'T16'" in old and "'T'+str(helper_prefill_maximum)" in new,'identity source transform')
 d=read(R/'identity64-result3.json');audit(d);bad=copy.deepcopy(d);bad['helper_prefix_profile_id']='B1D512T16'
 try:audit(bad)
 except ValueError:pass
 else:raise AssertionError('incorrect legacy identity admitted')
 report=dict(passed=True,profile_id='B1D512T64',old_historical_label='B1D512T16',historical_cache_entries_zero=True,historical_mathematical_work_preserved=True,mathematical_functions_ast_identical=sorted(names),fresh_target_output_ids=256,fresh_full_target_state_digest_pairs=130,full_causal_proposals_and_acceptance_match_old_cap64=True,wrong_identity_negative_control_rejected=True,performance_collection_not_rerun_or_relabelled=True,no_comparator_execution=True,goal_complete=False,files_sha256={str(R/n):sha(R/n) for n in ['b1_worker1.py','b1_worker2.py','prepare_identity3.py','identity-preparation3.json','identity-change3.patch','identity64-result3.json','identity3-preflight.json','identity3-postflight.json','audit_identity3.py']})
 p=R/'identity-audit3.json';need(not p.exists(),'report exists');p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='files_sha256'},indent=2))
if __name__=='__main__':main()
