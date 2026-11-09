"""Audit current native coding collection and immutable-completion comparisons.
Collection integrity passing is distinct from the failing code-quality gate.
"""
from pathlib import Path
import json,hashlib,copy,sys
r=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
owners=['garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets']
def load():
 events=[]
 for line in (r/'server1/worker-stdout.jsonl').read_text().splitlines():
  d=json.loads(line)
  try:e=json.loads(d['line'])
  except ValueError:continue
  events.append(e)
 clients={}
 for task in json.loads((r/'fixture1.json').read_text())['requests']:
  p=r/'responses1'/task['id']/'request-0';sse=[json.loads(v)['data'] for v in p.with_suffix('.events.jsonl').read_text().splitlines()]
  clients[task['id']]=dict(result=json.loads(p.with_suffix('.result.json').read_text()),request=json.loads(p.with_suffix('.request.json').read_text()),events=[v if v=='[DONE]' else json.loads(v) for v in sse])
 return dict(native=events,clients=clients,strict=json.loads((r/'strict-quality1.json').read_text()),diagnosis=json.loads((r/'block-diagnosis1.json').read_text()),comparison=json.loads((r/'saved-comparison2.json').read_text()),worker_exit=json.loads((r/'server1/worker-exit.json').read_text()))
def validate(d):
 fixture=json.loads((r/'fixture1.json').read_text());suite=json.loads((r/'tools1/suite.json').read_text());assert hashlib.sha256((r/'tools1/suite.json').read_bytes()).hexdigest()=='675ea667c357f361ce39e3863b61193dc6b32f4a229eb7d29a17803b75408786'
 es=d['native'];assert not any(e.get('event')=='error' for e in es);assert d['worker_exit']==dict(returncode=0,expected_close=True)
 ready=[e for e in es if e.get('event')=='ready'];closed=[e for e in es if e.get('event')=='closed'];assert len(ready)==len(closed)==1
 assert ready[0]['profile']==json.loads((r/'profile1.json').read_text()) and ready[0]['prefix_cache'] is False and ready[0]['active_requests']==4
 assert all(closed[0]['memory'][k]==0 for k in owners) and closed[0]['memory']['metal_allocated_bytes']<=4*1024*1024
 complete={e['id']:e for e in es if e.get('event')=='completed'};assert len(complete)==8;allids={};total=0
 for task in fixture['requests']:
  name=task['id'];c=d['clients'][name];assert c['request']==task['body'];a=c['result'];assert a['status']=='completed' and a['finish_reason']=='stop' and a['cached_prompt_tokens']==0
  sse=c['events'];assert sse[-1]=='[DONE]' and sse.count('[DONE]')==1;final=[e for e in sse[:-1] if 'response_token_ids' in e];assert len(final)==1;final=final[0];ids=[e['token_id'] for e in sse[:-1] if 'token_id' in e];assert ids==final['response_token_ids'] and ids[-1]==2 and len(ids)==a['generated_tokens']<=1024
  assert final['prompt_token_ids']==task['workload']['token_ids'] and a['prompt_tokens']==len(task['workload']['token_ids'])
  n=complete[final['id']];assert n['generated_ids']==ids and n['prompt_ids']==final['prompt_token_ids'] and n['text']==a['text'] and n['cached_tokens']==0;allids[name]=ids;total+=len(ids)
  old=json.loads((r.parent/'mistral-native-4k-coding1'/name/'request-0.result.json').read_text());assert ids==old['generated_ids']
 assert len({e['native_request'] for e in complete.values()})==8
 live={};maximum=0
 for e in es:
  if e.get('event')=='native_batch':
   assert len(e['active_rows'])==4 and sum(e['active_rows'])==len(e['requests'])
   for q in e['requests']:
    assert live.get(q['row']) in (None,(q['id'],q['epoch']));live[q['row']]=(q['id'],q['epoch'])
   maximum=max(maximum,len(live));assert maximum<=4
  elif e.get('event')=='completed':assert live.pop(e['native_row'])==(e['native_request'],e['native_epoch'])
  elif e.get('event')=='native_affine_profile':assert e['misses']==0 and 0<e['pipeline_count']<=512
 assert maximum==4 and not live and total==3052
 q=d['strict'];assert q['sandbox_probe']['status']=='passed' and q['passed']==0 and q['total']==8 and q['suite_sha256']==fixture['suite_sha256'];assert q['evaluator_sha256']=='a73809aab308180338ab1df644afdd8c4970bc136d6b0bc93c049c8862ebcc7a';assert len(q['results'])==8
 for row,task in zip(q['results'],suite['tasks']):assert row['task']==task['id'] and row['status']=='invalid_code' and row['finish_reason']=='stop' and row['response_sha256']==hashlib.sha256((r/'responses1'/task['id']/'request-0.result.json').read_bytes()).hexdigest()
 diag=d['diagnosis'];assert diag['diagnostic_only'] is True and diag['strict_passed']==0 and diag['extracted_contract_passed']==1 and diag['strict_total']==8;assert [row['task'] for row in diag['results']]==[t['id'] for t in suite['tasks']]
 comparison=d['comparison'];assert comparison['passed'] is True and comparison['old_native_exact_ids']==8 and comparison['mlx_exact_ids']==5 and comparison['current_output_tokens']==3052 and comparison['vllm_exact_text']==comparison['sglang_exact_text']==0 and comparison['vllm_leading_space_only']==6 and comparison['sglang_leading_space_only']==5
 for row in comparison['rows']:
  name=row['task'];assert row['current_tokens']==len(allids[name]);assert row['current_ids_sha256']==hashlib.sha256(json.dumps(allids[name],separators=(',',':')).encode()).hexdigest()
  for label,old in row['comparisons'].items():
   raw=json.loads(Path(old['response_path']).read_text());assert old['response_sha256']==hashlib.sha256(Path(old['response_path']).read_bytes()).hexdigest();text=d['clients'][name]['result']['text'];assert old['text_exact'] is (text==raw['text']);assert old['single_leading_space_only'] is (raw['text']==' '+text)
   if 'generated_ids' in raw:assert old['ids_exact'] is (allids[name]==raw['generated_ids'])
   else:assert old['ids_exact'] is None
 return dict(completed_requests=8,output_ids=3052,strict_passed=0,strict_total=8,diagnostic_extracted_passed=1,old_native_exact_ids=8,mlx_exact_ids=5,maximum_live_rows=maximum,quality_gate_passed=False)
def main():
 d=load();out=validate(d)
 def corrupt_output(v):
  name=next(iter(v['clients']));final=next(e for e in v['clients'][name]['events'] if isinstance(e,dict) and 'response_token_ids' in e);final['response_token_ids'][0]=-1
 mutations=[('false_quality_pass',lambda v:v['strict'].__setitem__('passed',1)),('false_sandbox_pass',lambda v:v['strict']['sandbox_probe'].__setitem__('status','failed')),('truncated_answer',lambda v:next(iter(v['clients'].values()))['result'].__setitem__('finish_reason','length')),('changed_output_id',corrupt_output),('cached_invalid_control',lambda v:next(iter(v['clients'].values()))['result'].__setitem__('cached_prompt_tokens',128)),('retained_owner',lambda v:next(e for e in v['native'] if e.get('event')=='closed')['memory'].__setitem__('garnet_live_matrix_owners',1)),('false_reference_id_parity',lambda v:next(iter(v['comparison']['rows'][0]['comparisons'].values())).__setitem__('ids_exact',False)),('false_leading_space_claim',lambda v:v['comparison'].__setitem__('vllm_leading_space_only',8))]
 rejected=[]
 for name,mutate in mutations:
  v=copy.deepcopy(d);mutate(v);assert v!=d
  try:validate(v)
  except (AssertionError,KeyError,ValueError):rejected.append(name)
  else:raise AssertionError('corruption admitted '+name)
 out.update(passed=True,scope=__doc__,rejected_corruptions=rejected,no_reference_inference_rerun=True,no_new_performance_claim=True)
 (r/'collection-audit1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
