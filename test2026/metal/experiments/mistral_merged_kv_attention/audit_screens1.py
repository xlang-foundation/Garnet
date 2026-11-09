"""Audit serial merged-KV operator screens against admitted saved GPU fixtures.
Usage: python3 audit_screens1.py ABSOLUTE_STUDY
Does not start GPU work or rerun comparators.
"""
from pathlib import Path
import hashlib
import json
import math
import statistics
import sys

r=Path(sys.argv[1]).resolve();load=lambda n:json.loads((r/n).read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
status=load('screen-status1.json');assert status['completed'] and status['status']=='collected_pending_audit' and status['source_hashing_finished_before_collection']
assert load('correctness-status2.json')['returncode']==0 and load('correctness2/audit1.json')['passed']
assert load('compile-status2.json')['returncode']==load('compile-status3.json')['returncode']==0
for path,h in load('screen-preflight-hashes1.json').items():assert sha(Path(path))==h,path
for n,h in load('correctness2/payload-hashes1.json').items():assert sha(r/'correctness2'/n)==h,n
original=(r/'merged_attention2.mm').read_text();reverse=(r/'merged_attention3.mm').read_text()
assert original.replace('int v=(round+order)%3;','int v=2-(round+order)%3;')==reverse
fixture=load('correctness2/correctness1.json');assert len(status['runs'])==2
rows=[];previous=0;pids=set()
for cohort,run in enumerate(status['runs']):
    assert run['cohort']==cohort and run['status']=='terminal' and run['returncode']==0 and run['pid'] not in pids
    assert previous<=run['started_ns']<run['finished_ns'];previous=run['finished_ns'];pids.add(run['pid'])
    assert run['argv']==[str(r/('merged_attention'+str(cohort+2))),str(r/'combined1.metal'),str(r/'correctness2'),str(r/f'screen{cohort}'),'screen']
    raw=load(f'screen{cohort}/screen1.json');assert raw['collected'] and raw['all_prepared_before_measurement'] and len(raw['cases'])==7
    assert 'Completed cell 6 screen' in (r/f'screen{cohort}.stdout').read_text()
    for index,c in enumerate(raw['cases']):
        for k,v in fixture['cases'][index].items():
            if k!='checked_parent_words':assert c[k]==v,(cohort,index,k)
        assert len(c['samples'])==27
        samples={}
        for i,s in enumerate(c['samples']):
            round_,order=divmod(i,3);variant=(round_+order)%3 if cohort==0 else 2-(round_+order)%3
            assert [s['round'],s['order'],s['variant']]==[round_,order,variant]
            assert math.isfinite(s['gpu_ms']) and math.isfinite(s['wall_ms']) and 0<s['gpu_ms']<=s['wall_ms']+1
            samples[round_,variant]=s['gpu_ms']
        ratios=[samples[i,1]/samples[i,2] for i in range(9)]
        medians=[statistics.median(samples[i,v] for i in range(9)) for v in range(3)]
        rows.append(dict(cohort=cohort,case=index,batch=c['batch'],chunk=c['chunk'],capacity=c['capacity'],extent=c['extent'],counts=c['counts'],gated=c['gated'],
            original_gpu_ms=medians[0],contiguous_control_gpu_ms=medians[1],merged_gpu_ms=medians[2],
            paired_contiguous_over_merged_median=statistics.median(ratios),paired_ratio_range=[min(ratios),max(ratios)],merged_faster_pairs=sum(v>1 for v in ratios),
            temporary_merged_kv_bytes=c['temporary_merged_kv_bytes']))
result=dict(passed=True,scope='Two fresh strictly serial standalone operator processes;9rotating three-variant rounds per7cases, reversed order. All MPS/pipelines warmed before first timing. Pack/materialization included; actual inference, scheduling/TTFT, coding and prefix state are outside scope. Ragged full-row results are not selected native active-span measurements.',
    measured_command_buffers=378,comparisons=rows,
    assessment='Reject broad native integration of this materializing variant. Against the already-tested exact contiguous-softmax4product control, full B4 long-prefix merged-KV is slower in both reversed-order cohorts. Cold-prefix cells favor2products but are noisy and do not establish complete prefill/body gains. The prototype adds128MiB physical KV scratch at B4/C4096. A future no-copy graph-level rewrite would require independent immutable logical-alias/liveness and state-commit proof; do not mutate the generic read-only attention inputs.',
    hypothesis='Fewer matrix products alone are insufficient; staging and changed shapes/dispatches must be measured together. These data do not isolate copy time as the sole cause.',
    selected_for_native_integration=False,new_native_default=False,serving_speed_claim=False,quality_gate_passed=False,goal_complete=False)
(r/'screen-audit1.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
