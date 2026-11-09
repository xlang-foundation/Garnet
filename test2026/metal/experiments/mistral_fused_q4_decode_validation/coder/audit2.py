"""Audit causal learned-draft on/off fusion stage, complete outputs and state."""
from pathlib import Path
import copy,hashlib,json,math,sys
HERE=Path(__file__).resolve().parent
R=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else HERE
DEST=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else R/'audit2.json'
sys.path.insert(0,str(HERE));from audit_mistral_prediction_confidence import validate
read=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def need(v,m):
    if not v:raise ValueError(m)
def routes(d,enabled):
    need(d['fused_decode_enabled'] is enabled,'wrong on/off engine')
    need(('native_fused_q4_decode_rows4_default_v1' in d['backend'])==enabled,'wrong backend')
    need(d['prepared']['fused_decode_profiles'] in ([1,2] if enabled else [0]),'wrong prepared profiles')
    full=partial=blocks=copies=fallback=0;gpu=0.
    for row in d['records']:
        for q in row['rounds']:
            metrics=q['metrics']
            if metrics is None:fallback+=1;continue
            length=len(q['proposal'])+1;full+=length==4;partial+=length<4
            want=32 if length==4 else 0
            if enabled:
                need(metrics['target_fused_q4_decode_blocks']==want and type(metrics['target_fused_q4_decode_input_copies']) is int and 0<=metrics['target_fused_q4_decode_input_copies']<=want,'wrong fused frame/copy count')
                blocks+=want;copies+=metrics['target_fused_q4_decode_input_copies']
            else:need('target_fused_q4_decode_blocks' not in metrics and 'target_fused_q4_decode_input_copies' not in metrics,'control unexpectedly exposes fusion')
            need(metrics['target_affine_decode_block_dispatches']==(161 if length==4 else 1),'changed logical blocked projection count')
            need(math.isfinite(metrics['target_gpu_ms']) and metrics['target_gpu_ms']>0,'invalid target GPU duration')
            gpu+=metrics['target_gpu_ms']/1000
    need(full>0 and partial>0 and fallback>0,'missing full/partial/fallback coverage')
    return dict(full_frames=full,partial_frames=partial,fallback_frames=fallback,fused_blocks=blocks,input_copies=copies,covered_target_gpu_seconds=gpu)
def main():
    need(not DEST.exists(),'never overwrite audit');plan=read(R/'plan1.json');pre=read(R/'preflight1.json');need(pre['passed'] and read(R/'postflight1.json')['passed'],'unfinished collection')
    for n,h in pre['files_sha256'].items():need(sha(Path(n))==h,'changed bound input '+n)
    reference=read(Path(plan['reference']));inputs=read(Path(plan['input']));results={};summaries=[];negative=[];files={};last=0;pids=set()
    for cohort,role in plan['sequence']:
        label='cohort'+str(cohort)+'-'+role;status=read(R/(label+'-status1.json'))
        need(status['status']=='terminal' and status['returncode']==0 and status['started_ns']>=last and status['pid'] not in pids,'nonserial/incomplete run');last=status['finished_ns'];pids.add(status['pid'])
        p=R/(label+'-result1.json');d=read(p);files[str(p)]=sha(p)
        need(d['inputs']==inputs and d['library']==str(Path(plan['libraries'][role])/'libgarnet.dylib') and d['projection_occupancy']=='occupancy-control','changed native input/library/profile')
        rows=validate(d,reference,draft_family='qwen2');route=routes(d,role=='candidate');tokens=sum(len(x['generated_ids'])-1 for x in d['records']);seconds=sum(x['decode_seconds'] for x in d['records'])
        summaries.append(dict(cohort=cohort,role=role,tokens=tokens,seconds=seconds,tokens_per_second=tokens/seconds,draft_seconds=sum(x['draft_seconds'] for x in rows),target_seconds=sum(x['target_seconds'] for x in rows),target_calls=sum(x['target_calls'] for x in rows),accepted=sum(x['accepted'] for x in rows),proposed=sum(x['proposed'] for x in rows),prepared_metal_bytes=d['prepared']['memory']['metal_allocated_bytes'],route=route));results[(cohort,role)]=d
        for name,mut in [('wrong_output',lambda b:b['records'][0]['generated_ids'].__setitem__(0,-1)),('wrong_kv',lambda b:b['records'][0]['state_sha256'][0].__setitem__(0,'0'*64)),('wrong_hidden',lambda b:b['records'][0].update(hidden_sha256='0'*64)),('future_cursor',lambda b:b['records'][0]['rounds'][0].update(committed_before=2)),('wrong_rate',lambda b:b['records'][0].update(decode_tokens_per_second=0.)),('wrong_engine',lambda b:b.update(fused_decode_enabled=not (role=='candidate')))]:
            bad=copy.deepcopy(d);mut(bad)
            try:validate(bad,reference,draft_family='qwen2');routes(bad,role=='candidate')
            except (ValueError,KeyError,IndexError,TypeError):negative.append(label+':'+name)
            else:raise ValueError('corruption admitted '+label+':'+name)
        if role=='candidate':
            for name,mut in [('missing_fusion',lambda b:next(q['metrics'] for row in b['records'] for q in row['rounds'] if q['metrics'] and len(q['proposal'])==3).update(target_fused_q4_decode_blocks=0)),('fused_partial',lambda b:next(q['metrics'] for row in b['records'] for q in row['rounds'] if q['metrics'] and len(q['proposal'])<3).update(target_fused_q4_decode_blocks=32)),('false_input_copy',lambda b:next(q['metrics'] for row in b['records'] for q in row['rounds'] if q['metrics']).update(target_fused_q4_decode_input_copies=33))]:
                bad=copy.deepcopy(d);mut(bad)
                try:routes(bad,True)
                except (ValueError,KeyError,IndexError,TypeError):negative.append(label+':'+name)
                else:raise ValueError('corruption admitted '+name)
    comparisons=[]
    for cohort in [1,2]:
        a=results[(cohort,'control')];b=results[(cohort,'candidate')];per=[]
        for x,y in zip(a['records'],b['records']):
            for k in ['case_id','prompt_ids','generated_ids','slots','state_sha256','hidden_sha256']:need(x[k]==y[k],'on/off target differs '+k)
            need(len(x['rounds'])==len(y['rounds']),'changed target calls')
            for u,v in zip(x['rounds'],y['rounds']):
                for k in ['committed_before','slot','proposal','submitted_ids','emitted_ids','consumed','accepted_draft']:need(u[k]==v[k],'changed greedy proposal/commit '+k)
                for k in ['reason','prompt_ids','cached_tokens','missing_tokens','prefill_chunks','generated_ids','executed_draft_ids','translated_ids','stop_reason']:
                    need(u['draft'].get(k)==v['draft'].get(k),'changed proposer history '+k)
            per.append(dict(case_id=x['case_id'],ratio=x['decode_seconds']/y['decode_seconds']))
        old=next(x for x in summaries if x['cohort']==cohort and x['role']=='control');new=next(x for x in summaries if x['cohort']==cohort and x['role']=='candidate');ratio=old['seconds']/new['seconds']
        comparisons.append(dict(cohort=cohort,speed_ratio=ratio,per_case=per,stage_gate_passed=ratio>=plan['minimum_stage_gain_ratio'] and min(x['ratio'] for x in per)>=plan['no_individual_case_regression_below_ratio']))
    DEST.write_text(json.dumps(dict(passed=True,scope=__doc__,summaries=summaries,comparisons=comparisons,declared_stage_gate_passed=all(x['stage_gate_passed'] for x in comparisons),all_output_ids_full_kv_hidden_exact_frozen=True,all_proposals_commits_and_draft_history_exact_on_off=True,negative_controls_rejected=negative,raw_files_sha256=files,strict_coding_score_unchanged='0/8',matched_http_speedup_admitted=False,no_comparator_execution=True,production_runtime_integrated=False,goal_complete=False),indent=2)+'\n');print(json.dumps(dict(passed=True,summaries=summaries,comparisons=comparisons),indent=2))
if __name__=='__main__':main()
