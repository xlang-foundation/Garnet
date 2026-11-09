"""Audit both retained opposite-order component screens; no GPU work."""
from pathlib import Path
import hashlib
import json
import math
import statistics
import sys

s=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
correct=json.loads((s/'correctness-audit1.json').read_text())
assert correct['passed'] and not correct['native_integration']
cohorts=[]
for cohort in [1,2]:
    state=json.loads((s/f'screen-run-status{cohort}.json').read_text())
    assert state['status']=='complete' and state['returncode']==0
    assert state['paused'] and state['resumed'] and state['paused_state']=='T'
    assert state['source_and_fixture_hashing_finished_before_screen']
    assert 'xctrace export' in state['export_identity']
    manifest=json.loads((s/f'screen-preflight-hashes{cohort}.json').read_text())
    for name, expected in manifest.items():
        h=hashlib.sha256()
        with (s/name).open('rb') as f:
            for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
        assert h.hexdigest()==expected,name
    raw=json.loads((s/f'screen{cohort}/screen{cohort}.json').read_text())
    assert raw['passed'] and raw['warmups_per_variant']==3 and raw['pairs_per_case']==9
    assert not raw['serving_speed_claim'] and not raw['native_integration']
    assert len(raw['cases'])==5
    cells=[]
    for ci,cell in enumerate(raw['cases']):
        samples=cell['samples']
        assert len(samples)==18
        words=correct['cases'][ci]['exact_parent_words']
        pairs=[]
        for round in range(9):
            pair=[x for x in samples if x['round']==round]
            start=(round+cohort-1)%2
            assert [(x['order'],x['variant']) for x in pair]==[(i,'candidate' if (start+i)%2 else 'control') for i in range(2)]
            for x in pair:
                assert math.isfinite(x['gpu_ms']) and x['gpu_ms']>0
                assert x['exact_parent_words']==words and x['readonly_inputs_unchanged']
            by={x['variant']:x['gpu_ms'] for x in pair}
            pairs.append(by['control']/by['candidate'])
        timings={k:[x['gpu_ms'] for x in samples if x['variant']==k] for k in ['control','candidate']}
        cells.append(dict(case=ci,batch=cell['batch'],actual_chunk=cell['chunk'],extent=cell['extent'],first=cell['first'],active_counts=cell['active_counts'],
                          control_median_gpu_ms=statistics.median(timings['control']),candidate_median_gpu_ms=statistics.median(timings['candidate']),
                          median_paired_control_over_candidate=statistics.median(pairs),
                          ratio_of_medians=statistics.median(timings['control'])/statistics.median(timings['candidate']),
                          paired_range=[min(pairs),max(pairs)],raw_pairs=pairs))
    cohorts.append(dict(cohort=cohort,cells=cells))
result=dict(passed=True,scope='Two fresh serial opposite-start-order processes; component GPU command spans only. Each call checks complete output parents against frozen audited control and immutable read-only inputs after its timestamp span. Source/fixture hashing and compilation finish before timing. Original CPU export is held, then resumed.',
            cohorts=cohorts,
            assessment='4K ragged-prefix cell improves consistently about1.57-1.58x across these cohorts. 2K cells have large order/drift variability, including slower candidate pairs; do not promote ratio-of-medians or a2x claim. This motivates native guarded integration and state/lifetime/body measurements, not production selection.',
            clocks_and_external_activity_controlled=False,serving_speed_claim=False,quality_gate_passed=False,native_integration=False,goal_complete=False)
(s/'component-screen-audit1.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
