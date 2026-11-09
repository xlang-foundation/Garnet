# SPDX-License-Identifier: Apache-2.0
"""Audit two-case raw/shared feasibility; no serving or speed qualification."""
import sys,json,hashlib,copy,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_mistral_medusa_generation import validate,require
R=Path(sys.argv[1]);output=Path(sys.argv[2]) if len(sys.argv)>2 else R/'generation-audit1.json'
manifest=json.loads((R/'generation-manifest1.json').read_text())
for p,h in manifest['files'].items():require(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,'changed frozen source/library')
fixture=json.loads((R/'fixture2cases.json').read_text());data=json.loads((R/'generation1/status.json').read_text());controls={c['id']:json.loads((R.parent/'mistral-native-4k-coding1'/c['id']/'request-0.result.json').read_text()) for c in fixture['cases']}
require(json.loads((R/'generation-status1.json').read_text())['returncode']==0,'collector failed')

def check(d):
 require(len(d['records'])==6 and 'medusa_residual_features_bf16' in d['backend'],'missing shared extension/records')
 expected=[]
 for i,c in enumerate(fixture['cases']):
  modes=['ordinary','raw','shared'] if i%2==0 else ['shared','raw','ordinary']
  expected.extend([(c['id'],mode) for mode in modes])
 require([(r['id'],r['mode']) for r in d['records']]==expected,'changed mode order/coverage')
 comparisons=[]
 for mode in ('raw','shared'):
  subset=copy.deepcopy(d);subset['records']=[r for r in subset['records'] if r['mode'] in ('ordinary',mode)]
  for r in subset['records']:
   if r['mode']==mode:r['mode']='medusa'
  result=validate(subset,fixture,controls)
  for r in result:r['mode']=mode
  comparisons.extend(result)
 for r in d['records']:
  for step in r['rounds']:
   for key in ('draft_seconds','target_seconds'):require(type(step[key]) in (int,float) and math.isfinite(step[key]) and step[key]>=0,'invalid component time')
   require(step['metrics']['target_attention_layout_dispatches']==32 and step['metrics']['target_attention_vector4_dispatches']==32,'changed target verification route')
 return comparisons
comparisons=check(data);mutations={
 'incomplete':lambda d:d.update(completed=False),
 'missing_mode':lambda d:d['records'].pop(),
 'changed_output':lambda d:d['records'][0]['generated_ids'].__setitem__(10,0),
 'wrong_state':lambda d:d['records'][1]['state_sha256'][0].__setitem__(0,'0'*64),
 'wrong_hidden':lambda d:d['records'][2].update(hidden_sha256='0'*64),
 'future_cursor':lambda d:d['records'][2]['rounds'][0].update(slot=99),
 'wrong_acceptance':lambda d:d['records'][2]['rounds'][0].update(accepted_proposal_inputs=99),
 'fake_rate':lambda d:d['records'][2].update(decode_tokens_per_second=999),
 'vector_fallback':lambda d:d['records'][2]['rounds'][0]['metrics'].update(target_attention_vector4_dispatches=0),
 'missing_rollback':lambda d:d['records'][2]['rounds'][0]['metrics'].update(commit_kv_restores=0),
 'unbounded_memory':lambda d:d['prepared'].update(metal_allocated_bytes=10**15),
 'live_owner':lambda d:d['cleanup'].update(garnet_live_matrix_owners=1),
}
# The existing auditor checks per-record allocations; also bind prepared allocation.
def checked(d):
 require(d['prepared']['metal_allocated_bytes']<=d['prepared']['metal_recommended_working_set_bytes']-256*1024*1024,'prepared memory')
 return check(d)
checked(data);rejected=[]
for label,mutate in mutations.items():
 changed=copy.deepcopy(data);mutate(changed)
 try:checked(changed)
 except (ValueError,KeyError,IndexError,TypeError):rejected.append(label)
 else:raise ValueError('accepted corrupt evidence '+label)
summary={}
for mode in ('ordinary','raw','shared'):
 rows=[r for r in data['records'] if r['mode']==mode];n=sum(len(r['generated_ids'])-1 for r in rows);seconds=sum(r['decode_seconds'] for r in rows);rounds=[s for r in rows for s in r['rounds']]
 summary[mode]=dict(tokens=n,seconds=seconds,tps=n/seconds,calls=len(rounds),accepted=sum(s['accepted_proposal_inputs'] for s in rounds),proposed=sum(s['active_tokens']-1 for s in rounds),draft_seconds=sum(s['draft_seconds'] for s in rounds),target_seconds=sum(s['target_seconds'] for s in rounds))
with output.open('x') as f:json.dump(dict(passed=True,comparisons=comparisons,summary=summary,rejected=rejected,result_sha256=hashlib.sha256((R/'generation1/status.json').read_bytes()).hexdigest(),scope=__doc__),f,indent=2)
print(json.dumps(summary,indent=2));print('two-case causal/state audit PASS corruptions',len(rejected))
