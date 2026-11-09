"""Saved native coder-draft cost census; different workload from HTTP baselines."""
from pathlib import Path
import collections,copy,hashlib,json,math,sys
ROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
DEST=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else ROOT/'saved-speculative-cost1.json'
D=ROOT.parent/'mistral-coder-draft-native1';files={}
def need(v,m):
    if not v:raise ValueError(m)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    b=p.read_bytes();files[str(p)]=hashlib.sha256(b).hexdigest();return json.loads(b)
def finite(x):return type(x) in [int,float] and math.isfinite(x) and x>=0
def census(data,expected):
    need(data['completed'],'unfinished native run')
    rounds=[q for rec in data['records'] for q in rec['rounds'] if q.get('draft')]
    need(len(rounds)==expected['target_calls']==1179,'wrong native round count')
    draft_total=target_total=decode_wall=decode_gpu=reconcile_gpu=covered_wall=covered_gpu=0
    decode_steps=missing=chunks=zeroed=covered=0;occupancies=collections.Counter();block_counts=collections.Counter();fallback=0
    for q in rounds:
        d=q['draft'];need(finite(d.get('seconds')) and finite(q['target_seconds']),'invalid stage duration')
        draft_total+=d['seconds'];target_total+=q['target_seconds']
        for sample in d.get('decode_samples',[]):
            wall,gpu=sample['wall_seconds'],sample['gpu_ms']/1000
            need(finite(wall) and finite(gpu) and 0<gpu<=wall,'invalid draft wall/GPU units/order')
            decode_steps+=1;decode_wall+=wall;decode_gpu+=gpu
        if d.get('reconcile'):
            r=d['reconcile'];need(finite(r['gpu_ms']) and type(r['zeroed_bytes']) is int and r['zeroed_bytes']>=0,'invalid reconciliation observation')
            reconcile_gpu+=r['gpu_ms']/1000;zeroed+=r['zeroed_bytes']
        missing+=d.get('missing_tokens',0);chunks+=len(d.get('prefill_chunks',[]))
        if q.get('metrics'):
            metrics=q['metrics'];gpu=metrics['sequential_gpu_duration_sum_ms']/1000
            need(finite(gpu) and 0<gpu<=q['target_seconds'],'invalid target GPU units/order')
            need(len(q['submitted_ids'])==4 and metrics['target_affine_decode_block_dispatches'] in [1,161],'saved verifier input profile/native route changed')
            occupancies[len(q['submitted_ids'])]+=1;block_counts[metrics['target_affine_decode_block_dispatches']]+=1
            covered+=1;covered_gpu+=gpu;covered_wall+=q['target_seconds']
        else:fallback+=1
    need(decode_steps==expected['draft_steps']==1875 and covered==1125 and fallback==54 and block_counts=={161:1121,1:4},'stage coverage differs from independent causal audit')
    need(math.isclose(draft_total,expected['draft_seconds'],rel_tol=2e-12) and math.isclose(target_total,expected['target_seconds'],rel_tol=2e-12),'stage totals disagree with causal audit')
    need(decode_gpu<=decode_wall<=draft_total and covered_gpu<=covered_wall<=target_total,'invalid stage accounting')
    return dict(target_rounds=len(rounds),draft_total_wall_seconds=draft_total,draft_decode_samples=decode_steps,draft_decode_wall_seconds=decode_wall,draft_decode_gpu_seconds=decode_gpu,reconciliation_gpu_seconds=reconcile_gpu,reconciliation_zeroed_bytes=zeroed,missing_draft_tokens=missing,draft_prefill_chunks=chunks,draft_other_wall_seconds=draft_total-decode_wall,target_total_wall_seconds=target_total,target_gpu_covered_rounds=covered,target_gpu_uninstrumented_fallback_rounds=fallback,target_covered_wall_seconds=covered_wall,target_covered_gpu_seconds=covered_gpu,target_covered_gpu_fraction_of_wall=covered_gpu/covered_wall,target_submitted_input_count_histogram=dict(occupancies),native_blocked_dispatch_count_histogram=dict(block_counts))
def main():
    need(not DEST.exists(),'never overwrite census')
    provenance=D/'provenance1.json';need(sha(provenance)=='80a24ab1d8d7e37536c4ca8058c84e92da13aaf0caee8a1e508557370a134c32','changed native coder checkpoint');files[str(provenance)]=sha(provenance)
    audit=read(D/'comparison-audit2.json');need(audit['completed'] and audit['all_target_ids_kv_hidden_match_frozen_native'] and audit['matched_http_speedup_admitted'] is False,'missing qualified saved scope')
    summaries={e['cohort']:e for e in audit['summaries'] if e['role']=='candidate'};reports={};data={}
    for cohort in [1,2]:
        p=D/('cohort'+str(cohort)+'-candidate-result1.json');need(sha(p)==audit['raw_files'][str(p)],'changed audited native trace')
        data[cohort]=read(p);reports[str(cohort)]=census(data[cohort],summaries[cohort])
    rejected=[]
    for name in ['nonfinite_draft_time','wrong_target_units','wrong_verifier_occupancy','wrong_native_block_route','missing_round']:
        bad=copy.deepcopy(data[1]);rec=next(r for r in bad['records'] if any(q.get('draft') and q.get('metrics') for q in r['rounds']));q=next(q for q in rec['rounds'] if q.get('draft') and q.get('metrics'))
        if name=='nonfinite_draft_time':q['draft']['seconds']=float('nan')
        elif name=='wrong_target_units':q['metrics']['sequential_gpu_duration_sum_ms']*=1000
        elif name=='wrong_verifier_occupancy':q['submitted_ids'].pop()
        elif name=='wrong_native_block_route':q['metrics']['target_affine_decode_block_dispatches']=0
        else:rec['rounds'].remove(q)
        try:census(bad,summaries[1])
        except (ValueError,KeyError,TypeError,IndexError):rejected.append(name)
        else:raise ValueError('invalid stage census admitted '+name)
    output=dict(passed=True,scope=__doc__,cohorts=reports,negative_controls_rejected=rejected,files_sha256=files,finding='All1125 instrumented target frames submit a four-input profile;1121 use161 native blocked dispatches and4 use only1. Four-row fusion is relevant to most single-request speculative verification frames, but must guard actual active rows and retain the narrower fallback. Ordinary C1 fusion did not pass its operator gain gate.',limitations=['No isolated prefill timing exists for the draft: other draft wall time includes prefill, retokenization, CPU routing and other work.','Target GPU coverage excludes54 fallback rounds/cohort; no missing interval is counted as zero-cost GPU work.','A four-input submitted profile does not by itself prove four active body rows.','These eight coding cases differ from the frozen2K/128 HTTP baseline; rates cannot establish2x against vLLM/SGLang.'],new_inference_executed=False,serving_speed_claim=False,goal_complete=False)
    DEST.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps({k:v for k,v in output.items() if k!='files_sha256'},indent=2))
if __name__=='__main__':main()
