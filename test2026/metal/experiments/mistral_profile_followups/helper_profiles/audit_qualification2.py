"""Audit helper upper bounds without requiring helper arithmetic to match across different profiles."""
from pathlib import Path
import copy,hashlib,importlib.util,json,math,sys
R=Path(__file__).resolve().parent;M=R.parent;V=M/'mistral-fused-verify-gqa-profile1';sys.path.insert(0,str(R))
spec=importlib.util.spec_from_file_location('helper_profile_bounded',R/'audit_bounded1.py');bounded=importlib.util.module_from_spec(spec);spec.loader.exec_module(bounded);bounded.ROOT=R
read=lambda p:json.loads(p.read_text())
def need(v,why):
 if not v:raise ValueError(why)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
REFERENCE=read(V/'b1-on-result1.json')['records'][0];KEYS=['case_id','prompt_ids','generated_ids','slots','state_sha256','hidden_sha256']
def check(d,cap,require_samples=True):
 plan=read(R/'plan1.json');need(d['helper_prefill_maximum']==cap and d['library']==plan['library']+'/libgarnet.dylib','profile/library binding');need(d['inputs']==read(R/'input1.json'),'frozen target/helper fixture')
 need(d['fused_decode_enabled'] and d['gqa_scores_enabled'] and d['gqa_values_enabled'],'target compiled profile changed');summary=bounded.check(d)
 for row in d['records']:
  for k in KEYS:need(row[k]==REFERENCE[k],'target full state/output '+k)
 traces=[q['draft'] for q in d['records'][1]['rounds'] if q['draft'].get('prompt_ids')]
 if require_samples:
  for t in traces:need(len(t['prefill_samples'])==len(t['prefill_chunks']),'missing prefill actual-token witness')
 initial=traces[0];need(len(initial['prompt_ids'])==222 and initial['missing_tokens']==222 and initial['cached_tokens']==0,'changed initial helper prompt')
 if require_samples:need(sum(s['actual'] for s in initial['prefill_samples'])==222 and initial['prefill_samples'][-1]['tracked']==222,'padded tokens entered native helper history')
 return dict(summary=summary,initial_prefill_chunks=initial['prefill_chunks'],actual_initial_helper_tokens=222,initial_helper_prefill_gpu_ms=sum(s['gpu_ms'] for s in initial['prefill_samples']),initial_helper_prefill_wall_seconds=sum(s['wall_seconds'] for s in initial['prefill_samples']),helper_graphs=d['prepared']['draft_graphs'],prepared_metal_bytes=d['prepared']['memory']['metal_allocated_bytes'],full_target_digest_pairs=130,output_ids=256)
def main():
 need(read(R/'qualification1-postflight.json')['passed'] and read(R/'sweep2-postflight.json')['passed'],'terminal collection gates');pre=read(R/'sweep2-preflight.json');need(pre['passed'],'preflight')
 for n,h in pre['files_sha256'].items():need(sha(Path(n))==h,'source/model binding '+n)
 data={};reports={};last=0
 for cap in [16,256,32,64,128]:
  s=read(R/(f'cap{cap}-status1.json'));need(s['status']=='terminal' and s['returncode']==0 and s['started_ns']>=last,'serial fresh native engines');last=s['finished_ns'];d=read(R/(f'cap{cap}-result1.json'));data[cap]=d;reports[str(cap)]=check(d,cap)
 negative=[]
 def reject(name,mutate):
  d=copy.deepcopy(data[256]);mutate(d)
  try:check(d,256)
  except (ValueError,KeyError,IndexError,TypeError,StopIteration):negative.append(name);return
  raise AssertionError('accepted corruption '+name)
 reject('wrong bound',lambda d:d.__setitem__('helper_prefill_maximum',16))
 reject('wrong graph family',lambda d:d['prepared'].__setitem__('draft_graphs',96))
 reject('wrong target output',lambda d:d['records'][1]['generated_ids'].__setitem__(0,-1))
 reject('wrong target KV',lambda d:d['records'][1]['state_sha256'][0].__setitem__(0,'0'*64))
 reject('future helper prefix',lambda d:next(q['draft'] for q in d['records'][1]['rounds'] if q['draft'].get('prompt_ids'))['prompt_ids'].append(1))
 reject('fake acceptance',lambda d:d['records'][1]['rounds'][0].__setitem__('consumed',0))
 reject('missing native token witness',lambda d:next(q['draft'] for q in d['records'][1]['rounds'] if q['draft'].get('prefill_samples'))['prefill_samples'].pop())
 reject('padded tokens tracked',lambda d:next(q['draft'] for q in d['records'][1]['rounds'] if q['draft'].get('prefill_samples'))['prefill_samples'][0].__setitem__('tracked',256))
 reject('false active tokens',lambda d:next(q['draft'] for q in d['records'][1]['rounds'] if q['draft'].get('prefill_samples'))['prefill_samples'][0].__setitem__('actual',256))
 reject('unprepared chunk',lambda d:next(q['draft'] for q in d['records'][1]['rounds'] if q['draft'].get('prefill_chunks'))['prefill_chunks'].__setitem__(0,512))
 reject('host KV copy',lambda d:next(q['draft']['reconcile'] for q in d['records'][1]['rounds'] if q['draft'].get('reconcile')).__setitem__('host_snapshot_bytes',1))
 reject('false decode speed',lambda d:d['records'][1].__setitem__('decode_tokens_per_second',999.))
 files=[R/n for n in ['audit_qualification2.py','audit_qualification1.py','audit_common1.py','audit_bounded1.py','plan1.json','input1.json','qualification1-preflight.json','qualification1-postflight.json','cap16-result1.json','cap256-result1.json','cap32-result1.json','cap64-result1.json','cap128-result1.json','sweep2-preflight.json','sweep2-postflight.json']]
 report=dict(passed=True,profiles=reports,full_target_output_ids=1280,full_target_state_digest_pairs=650,all_target_commits_exact_ordinary=True,helper_state_bits_equal_across_profiles_required=False,helper_proposals_independently_audited=True,all_graph_families_precompiled=True,native_prefix_retention_without_host_snapshots=True,negative_controls_rejected=negative,performance_admitted=False,http_or_quality_admitted=False,no_comparator_execution=True,goal_complete=False,files_sha256={str(p):sha(p) for p in files});p=R/'qualification-audit2.json';need(not p.exists(),'report exists');p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='files_sha256'},indent=2))
if __name__=='__main__':main()
