"""Audit saved component timings only; no model, GPU execution or baseline reruns."""
from pathlib import Path
import sys
import copy
import hashlib
import json
import math
import statistics

p=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
load=lambda q:json.loads(q.read_text())
def sha(q):
    h=hashlib.sha256()
    with q.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def audit_report(r, reverse):
    assert r['collected'] and r['actual_trained_operands']
    assert r['reverse'] is reverse and r['admitted_variants']==[0,1,3] and r['excluded_variants']==[2]
    assert r['all_prepared_before_measurement'] and r['warmups_per_cell_variant']==3 and r['measured_rounds']==9
    assert not r['native_integration'] and not r['serving_speed_claim'] and not r['goal_complete']
    assert r['compensated_half_layout_capacities']==[8,16,16,8,16,16]*32
    assert r['register_layout_status']==[0]*32 and r['relaxed_precision'] is False and r['math_mode']=='Safe'
    c=r['compensated_half_coordinate_parent_words']
    assert c==[0x7fc12345]*16+[1]*2560+[0]+[0x7fc12345]*16
    assert r['probability_scale']==32768 and r['qkv_scale']==16
    assert r['variant_names']==['original4mps','committed2mps','compensated_half3_fp32','compensated_half4_fp32']
    assert len(r['cases'])==3
    result=[]
    for ci,c in enumerate(r['cases']):
        assert [c[k] for k in ['batch','chunk','capacity','extent','gated']]==[4,128,4096,2048,False]
        assert c['first']==[1920]*4 and c['counts']==[128]*4
        assert c['params']==[4,128,32,8,128,4096,4100,1028,1028,4194304,4194304,131,133,4100,4100,0]
        assert c['pitches']==[128,4096,128]
        assert c['checked_parent_words']==10*(c['readonly_parent_words']+c['output_parent_words'])
        samples=c['samples'];assert len(samples)==27
        by={v:[] for v in [0,1,3]};wall={v:[] for v in [0,1,3]};paired={v:[] for v in [0,1]}
        for rnd in range(9):
            values={}
            for order in range(3):
                s=samples[rnd*3+order];slot=(rnd+order)%3
                expected=[0,1,3][2-slot if reverse else slot]
                assert s['round']==rnd and s['order']==order and s['variant']==expected
                assert all(math.isfinite(s[k]) and s[k]>0 for k in ['gpu_ms','wall_ms'])
                by[expected].append(s['gpu_ms']);wall[expected].append(s['wall_ms']);values[expected]=s['gpu_ms']
            for v in [0,1]:paired[v].append(values[v]/values[3])
        result.append(dict(layer=[0,15,31][ci],gpu_median_ms={str(v):statistics.median(x) for v,x in by.items()},
                           wall_median_ms={str(v):statistics.median(x) for v,x in wall.items()},
                           gpu_range_ms={str(v):[min(x),max(x)] for v,x in by.items()},
                           paired_control_over_candidate={str(v):statistics.median(x) for v,x in paired.items()}))
    return result

assert load(p/'bindings-audit2.json')['passed'] and load(p/'trained-screen-postflight2.json')['passed']
assert load(p/'trained-screens-status2.json')==dict(status='terminal',returncode=0,cohorts=2)
pre=load(p/'trained-screen-preflight2.json')
assert pre['passed'] and pre['source_hashing_and_builds_before_timing']
for q,h in pre['sha256'].items():assert sha(Path(q))==h,q
cohorts=[]
for index in [1,2]:
    assert load(p/f'trained-screen{index}-status.json')['returncode']==0
    r=load(p/f'trained-screen{index}/screen1.json')
    cohorts.append(dict(cohort=index,reverse=index==2,cases=audit_report(r,index==2)))

base=load(p/'trained-screen1/screen1.json');negatives={}
def negative(name,change):
    altered=copy.deepcopy(base);change(altered)
    try:audit_report(altered,False)
    except (AssertionError,KeyError,IndexError,TypeError,ValueError):negatives[name]=True
    else:raise AssertionError('Accepted corrupt timing: '+name)
negative('missing_sample',lambda r:r['cases'][0]['samples'].pop())
negative('duplicate_round',lambda r:r['cases'][0]['samples'][3].update(round=0))
negative('rejected_variant_injected',lambda r:r['cases'][0]['samples'][0].update(variant=2))
negative('wrong_order',lambda r:r['cases'][0]['samples'][0].update(order=1))
negative('negative_gpu_time',lambda r:r['cases'][0]['samples'][0].update(gpu_ms=-1))
negative('nan_gpu_time',lambda r:r['cases'][0]['samples'][0].update(gpu_ms=float('nan')))
negative('wrong_capacity',lambda r:r['compensated_half_layout_capacities'].__setitem__(0,9))
negative('changed_coordinates',lambda r:r['compensated_half_coordinate_parent_words'].__setitem__(16,2))
negative('relaxed_precision',lambda r:r.update(relaxed_precision=True))
negative('synthetic_relabel',lambda r:r.update(actual_trained_operands=False))
negative('missing_completion',lambda r:r.update(collected=False))
negative('wrong_context',lambda r:r['cases'][0].update(extent=4096))
negative('skipped_warmups',lambda r:r.update(warmups_per_cell_variant=0))
negative('false_serving_claim',lambda r:r.update(serving_speed_claim=True))
result=dict(passed=True,scope=__doc__,cohorts=cohorts,measured_gpu_commands=162,warmup_attention_commands=54,
            corruptions_rejected=len(negatives),corruption_controls=negatives,
            candidate_numerical_gate_passed=True,candidate_performance_rejected=True,
            native_integration=False,serving_speed_claim=False,goal_complete=False,
            interpretation='Ratios are paired per round; >1 favors candidate. FP32 cache and inputs retained. All layers regress in both cohorts; do not promote.')
assert all(c['paired_control_over_candidate']['1']<1 for cohort in cohorts for c in cohort['cases'])
(p/'trained-screens-audit2.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
