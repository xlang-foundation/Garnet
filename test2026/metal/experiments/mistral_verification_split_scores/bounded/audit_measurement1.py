"""Audit fresh2K native stages; no serving TTFT or comparator speed admission."""
from pathlib import Path
import copy,hashlib,json,math,sys
HERE=Path(__file__).resolve().parent;ROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else HERE
DEST=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else ROOT/'measurement-audit1.json'
sys.path.insert(0,str(HERE));from audit_bounded_common1 import validate
from audit_qualification1 import check_input,check_ordinary,need
read=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def check(data,mode,reference):
    need(data['experiment']==mode and len(data['records'])==1,'wrong native stage mode/coverage')
    row=data['records'][0];check_input(data['inputs'])
    need(data['library']==read(ROOT/'measurement-plan1.json')['library']+'/libgarnet.dylib' and data['fused_decode_enabled'] is False,'changed compiled native library')
    for k in ['case_id','prompt_ids','generated_ids','slots','state_sha256','hidden_sha256']:need(row[k]==reference[k],'complete state differs from current ordinary '+k)
    if mode=='ordinary':check_ordinary(data,row);summary=None
    else:
        summary=validate(data,dict(records=[dict(reference,mode='control')]),draft_family='qwen2')[0]
        need(row['mode']=='base' and summary['accepted']>0 and summary['proposed']>0,'draft never helped target')
        need(not any(q['draft']['reason']=='draft_capacity' for q in row['rounds']),'declared bounded input exceeds helper profile')
    need(abs(row['native_request_seconds']-row['prefill_seconds']-row['decode_seconds'])<1e-8,'native request sum differs')
    return row,summary
def main():
    need(not DEST.exists(),'never overwrite audit');plan=read(ROOT/'measurement-plan1.json');pre=read(ROOT/'measurement1-preflight.json');need(pre['passed'] and read(ROOT/'measurement1-postflight.json')['passed'],'collection unfinished')
    for n,h in pre['files_sha256'].items():need(sha(Path(n))==h,'changed admitted input '+n)
    need(read(ROOT/'qualification-audit1.json')['passed'],'missing independent state/causality gate')
    reference=read(ROOT/'qualification1-result.json')['records'][0];rows=[];data_by={};files={};previous=0;pids=set();rejected=[]
    for cohort,mode in plan['sequence']:
        label='cohort'+str(cohort)+'-'+mode;status=read(ROOT/(label+'-status1.json'))
        need(status['status']=='terminal' and status['returncode']==0 and status['started_ns']>=previous and status['pid'] not in pids,'nonserial or reused process');previous=status['finished_ns'];pids.add(status['pid'])
        path=ROOT/(label+'-result1.json');data=read(path);row,summary=check(data,mode,reference);data_by[(cohort,mode)]=data;files[str(path)]=sha(path)
        rows.append(dict(cohort=cohort,mode=mode,prompt_tokens=2048,output_tokens=128,native_prefill_seconds=row['prefill_seconds'],decode_seconds=row['decode_seconds'],decode_tokens_per_second=row['decode_tokens_per_second'],native_request_seconds=row['native_request_seconds'],native_request_output_tokens_per_second=128/row['native_request_seconds'],accepted_draft=summary['accepted'] if summary else 0,proposed=summary['proposed'] if summary else 0,target_calls=summary['target_calls'] if summary else 127,draft_seconds=summary['draft_seconds'] if summary else 0,target_seconds=summary['target_seconds'] if summary else row['decode_seconds'],prepared_metal_bytes=data['prepared']['memory']['metal_allocated_bytes']))
    comparisons=[]
    for cohort in [1,2]:
        a=next(x for x in rows if x['cohort']==cohort and x['mode']=='ordinary');b=next(x for x in rows if x['cohort']==cohort and x['mode']=='bounded')
        comparisons.append(dict(cohort=cohort,decode_rate_ratio=a['decode_seconds']/b['decode_seconds'],native_request_rate_ratio=a['native_request_seconds']/b['native_request_seconds'],declared_native_stage_gate_passed=a['decode_seconds']/b['decode_seconds']>=plan['minimum_native_decode_ratio']))
    left=data_by[(1,'bounded')]['records'][0]['rounds'];right=data_by[(2,'bounded')]['records'][0]['rounds']
    need(len(left)==len(right),'bounded target-call count changed between processes')
    for a,b in zip(left,right):
        for k in ['committed_before','slot','proposal','submitted_ids','emitted_ids','consumed','accepted_draft']:need(a[k]==b[k],'bounded proposal/commit changed '+k)
        for k in ['reason','prompt_ids','generated_ids','executed_draft_ids','prefill_chunks','cached_tokens']:need(a['draft'].get(k)==b['draft'].get(k),'bounded draft history changed '+k)
    for name,mut in [('wrong_output',lambda d:d['records'][0]['generated_ids'].__setitem__(0,-1)),('wrong_state',lambda d:d['records'][0]['state_sha256'][0].__setitem__(0,'0'*64)),('fake_rate',lambda d:d['records'][0].update(decode_tokens_per_second=999.)),('future_prefix',lambda d:next(q['draft'] for q in d['records'][0]['rounds'] if q['draft'].get('prompt_ids'))['prompt_ids'].append(1)),('false_cache_lcp',lambda d:next(q['draft']['reconcile'] for q in d['records'][0]['rounds'] if q['draft'].get('reconcile')).update(retained_tokens=9999)),('fake_acceptance',lambda d:d['records'][0]['rounds'][0].update(consumed=0)),('missing_rollback',lambda d:next(q['metrics'] for q in d['records'][0]['rounds'] if q['metrics'] and q['consumed']<len(q['proposal'])+1).update(commit_kv_restores=0)),('live_owner',lambda d:d['cleanup'].update(garnet_live_matrix_owners=1))]:
        bad=copy.deepcopy(data_by[(1,'bounded')]);mut(bad)
        try:check(bad,'bounded',reference)
        except (ValueError,KeyError,IndexError,TypeError):rejected.append(name)
        else:raise ValueError('corruption admitted '+name)
    result=dict(passed=True,scope=__doc__,rows=rows,comparisons=comparisons,declared_native_stage_gate_passed=all(x['declared_native_stage_gate_passed'] for x in comparisons),bounded_proposals_commits_and_draft_history_repeat_exactly=True,all_output_ids_exact_frozen_native=True,all_full_kv_hidden_exact_current_ordinary=True,state_digest_pairs=4*65,target_context_unchanged=True,helper_base_tokens=221,helper_capacity=512,negative_controls_rejected=rejected,files_sha256=files,no_comparator_execution=True,serving_ttft_measured=False,matched_http_speedup_admitted=False,coding_quality_qualified=False,production_runtime_integrated=False,goal_complete=False)
    DEST.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='files_sha256'},indent=2))
if __name__=='__main__':main()
