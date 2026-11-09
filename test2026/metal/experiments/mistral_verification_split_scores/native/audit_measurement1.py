"""Audit current split scores ON/OFF native timing after full qualification."""
from pathlib import Path
import copy,hashlib,json,math,sys
R=Path(__file__).resolve().parent;B=R.parent/'mistral-bounded-draft-long1'
sys.path.insert(0,str(B));from audit_bounded_common1 import validate
from audit_qualification1 import check_input,need
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
PLAN=read(R/'stage-plan1.json');REFERENCE=read(B/'qualification1-result.json')['records'][0]
HISTORY=read(R/'qualification-off1-result.json')['records'][1]['rounds']
def check(data,mode):
 need(data['experiment']=='bounded' and len(data['records'])==1 and not data['fused_decode_enabled'],'wrong bounded native stage')
 need(data['split_scores_enabled']==(mode=='on') and data['library']==PLAN['libraries'][mode]+'/libgarnet.dylib','changed native build')
 need(('native_mistral_verification_split_scores128_default_v1' in data['backend'])==(mode=='on'),'wrong feature tag')
 check_input(data['inputs']);row=data['records'][0]
 summary=validate(data,dict(records=[dict(REFERENCE,mode='control')]),draft_family='qwen2')[0]
 for k in ['case_id','prompt_ids','generated_ids','slots','state_sha256','hidden_sha256']:need(row[k]==REFERENCE[k],'current ordinary anchor mismatch '+k)
 need(math.isfinite(row['native_request_seconds']) and abs(row['native_request_seconds']-row['prefill_seconds']-row['decode_seconds'])<1e-8,'native request sum')
 need(len(row['rounds'])==len(HISTORY),'target call count changed')
 for q,expected in zip(row['rounds'],HISTORY):
  need(q['draft']['reason']!='draft_capacity','helper capacity fallback')
  for k in ['committed_before','slot','proposal','submitted_ids','emitted_ids','consumed','accepted_draft']:need(q[k]==expected[k],'target proposal/commit differs '+k)
  for k in ['reason','prompt_ids','generated_ids','executed_draft_ids','prefill_chunks','cached_tokens']:need(q['draft'].get(k)==expected['draft'].get(k),'draft history differs '+k)
  if q['metrics']:need(q['metrics']['target_attention_split_score_dispatches']==(32 if mode=='on' else 0),'split layer dispatch mismatch')
 return row,summary

def main():
 need(not (R/'measurement-audit1.json').exists(),'audit overwrite')
 pre=read(R/'measurement1-preflight.json');need(pre['passed'] and read(R/'measurement1-postflight.json')['passed'] and read(R/'qualification-audit1.json')['passed'],'unfinished admission')
 for n,h in pre['files_sha256'].items():need(sha(Path(n))==h,'changed admitted input '+n)
 rows=[];data_by={};last=0;pids=set()
 for cohort,mode in PLAN['sequence']:
  name=f'cohort{cohort}-{mode}';s=read(R/(name+'-status1.json'));need(s['status']=='terminal' and s['returncode']==0 and s['started_ns']>=last and s['finished_ns']>s['started_ns'] and s['pid'] not in pids,'nonserial/fresh/terminal measurement');last=s['finished_ns'];pids.add(s['pid'])
  data=read(R/(name+'-result1.json'));data_by[cohort,mode]=data;row,summary=check(data,mode)
  rows.append(dict(cohort=cohort,mode=mode,prompt_tokens=2048,output_tokens=128,native_prefill_seconds=row['prefill_seconds'],decode_seconds=row['decode_seconds'],decode_tokens_per_second=row['decode_tokens_per_second'],native_request_seconds=row['native_request_seconds'],native_request_output_tokens_per_second=128/row['native_request_seconds'],accepted_draft=summary['accepted'],proposed=summary['proposed'],target_calls=summary['target_calls'],draft_seconds=summary['draft_seconds'],target_seconds=summary['target_seconds'],target_gpu_ms=sum(q['metrics']['target_gpu_ms'] for q in row['rounds'] if q['metrics']),split_score_layer_dispatches=sum(q['metrics']['target_attention_split_score_dispatches'] for q in row['rounds'] if q['metrics']),prepared_metal_bytes=data['prepared']['memory']['metal_allocated_bytes']))
 comparisons=[]
 for cohort in [1,2]:
  a=next(x for x in rows if x['cohort']==cohort and x['mode']=='off');b=next(x for x in rows if x['cohort']==cohort and x['mode']=='on')
  comparisons.append(dict(cohort=cohort,decode_rate_ratio=a['decode_seconds']/b['decode_seconds'],native_request_rate_ratio=a['native_request_seconds']/b['native_request_seconds'],declared_native_stage_gate_passed=a['decode_seconds']/b['decode_seconds']>=PLAN['minimum_native_decode_ratio']))
 rejected=[]
 mutations={
  'wrong_output':lambda d:d['records'][0]['generated_ids'].__setitem__(0,-1),
  'wrong_kv':lambda d:d['records'][0]['state_sha256'][0].__setitem__(0,'0'*64),
  'wrong_hidden':lambda d:d['records'][0].update(hidden_sha256='0'*64),
  'fake_split_counter':lambda d:next(q['metrics'] for q in d['records'][0]['rounds'] if q['metrics']).update(target_attention_split_score_dispatches=0),
  'fake_rate':lambda d:d['records'][0].update(decode_tokens_per_second=999.),
  'future_prefix':lambda d:next(q['draft'] for q in d['records'][0]['rounds'] if q['draft'].get('prompt_ids'))['prompt_ids'].append(1),
  'false_resident_lcp':lambda d:next(q['draft']['reconcile'] for q in d['records'][0]['rounds'] if q['draft'].get('reconcile')).update(common_tokens=9999),
  'missing_rollback':lambda d:next(q['metrics'] for q in d['records'][0]['rounds'] if q['metrics'] and q['consumed']<len(q['proposal'])+1).update(commit_kv_restores=0),
  'live_owner':lambda d:d['cleanup'].update(garnet_live_matrix_owners=1),
 }
 for name,fn in mutations.items():
  bad=copy.deepcopy(data_by[1,'on']);fn(bad)
  try:check(bad,'on')
  except (ValueError,KeyError,IndexError,TypeError):rejected.append(name)
  else:raise ValueError('corruption admitted '+name)
 result=dict(passed=True,scope=__doc__,rows=rows,comparisons=comparisons,declared_native_stage_gate_passed=all(c['declared_native_stage_gate_passed'] for c in comparisons),all_output_ids_full_kv_hidden_exact_current_ordinary=True,state_digest_pairs=260,proposals_commits_and_helper_history_repeat_exactly=True,negative_controls_rejected=rejected,files_sha256={str(p):sha(p) for p in [R/'audit_measurement1.py',R/'measurement1-preflight.json']+[R/f'cohort{c}-{m}-result1.json' for c,m in PLAN['sequence']]},no_comparator_execution=True,serving_ttft_measured=False,matched_http_speedup_admitted=False,coding_quality_qualified=False,production_runtime_integrated=False,goal_complete=False)
 (R/'measurement-audit1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='files_sha256'},indent=2))
if __name__=='__main__':main()
