"""Audit GPU encoder timestamp attribution; instrumentation is not a benchmark."""
import sys
from pathlib import Path
from collections import Counter,defaultdict
import copy,hashlib,json,math
root=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
foundation=root.parent/'mistral-half-attention-native1'
load=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
status=load(root/'status1.json');assert status['status']=='terminal' and status['returncode']==0
preflight=load(root/'preflight1.json');assert preflight['passed'] and preflight['diagnostic_only']
for p,h in preflight['files_sha256'].items():assert sha(Path(p))==h,p
assert sha(foundation/'provenance7.json')==preflight['foundation_provenance_sha256']
record=load(root/'diagnostic1.json')
assert record['completed'] and record['diagnostic_only'] and record['counters_enabled']
assert (record['owners'],record['enabled'],record['batch'],record['capacity'],record['chunk'],record['offset'],record['warmups'],record['repeats'])==(4,1,4,4096,128,1920,3,1)
assert len(record['seeds'])==15 and len(record['measured'])==1
measured=record['measured'][0];assert (measured['half_dispatches'],measured['mps_dispatches'],measured['committed_dispatches'])==(32,0,0)
frozen=load(foundation/'provenance7.json')['files_sha256']
control=foundation/'body7/cohort0-owners4-on.json';assert sha(control)==frozen['body7/cohort0-owners4-on.json']
original=load(control)
for key in ['state_sha256','hidden_sha256','hidden_values','predictions']:assert record[key]==original[key],key
for key in ['garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets']:assert record['cleanup'][key]==0,key
assert record['cleanup']['metal_allocated_bytes']<=4194304
assert record['prepared']['metal_allocated_bytes']==record['after']['metal_allocated_bytes']

records=[json.loads(line[len('metal_gpu_counters '):]) for line in (root/'run1.stderr').read_text().splitlines() if line.startswith('metal_gpu_counters ')]
assert len(records)==20
body=[r for r in records if any(row['kind']=='AttentionChunk' for row in r['rows'])]
assert len(body)==19
def validate(r):
    assert r['diagnostic_only'] and r['mps_encoders_observed'] is False
    assert r['cpu1']>r['cpu0'] and r['gpu1']>r['gpu0']
    scale=(r['cpu1']-r['cpu0'])/(r['gpu1']-r['gpu0'])/1e6
    rows=r['rows'];assert rows
    if any(x['kind']=='AttentionChunk' for x in rows):
        assert len(rows)==1026
        assert Counter(x['kind'] for x in rows)==Counter(Layout=288,Norm=64,MatMul=384,NormRope=64,session=66,AttentionChunk=64,Add=64,Silu=32)
        assert sum(x['site']=='EncodeAttentionChunkInternal' for x in rows if x['kind']=='AttentionChunk')==64
    for row in rows:
        assert r['gpu0']<=row['start']<=row['end']<=r['gpu1']
        assert row['start']>0 and row['end']>0
        assert math.isfinite(row['ms']) and row['ms']>=0
        assert abs(row['ms']-(row['end']-row['start'])*scale)<=1e-6+1e-8*row['ms']
        assert 0<=row['command']<=r['partial_submissions']
    return scale
for r in records:validate(r)
last=body[-1];scale=validate(last)
assert abs(last['gpu_ms']-measured['gpu_ms'])<=1e-5

# Merge timestamp intervals and classify only exclusive coverage. Overlapping
# categories remain a separate bucket; encoder durations are never summed as
# elapsed GPU time or allocated to a family by assumption.
points=defaultdict(list);attention_index=0;counts=Counter();sites=Counter()
for row in last['rows']:
    category=row['kind']
    if category=='AttentionChunk':
        category='half_Q_KV_pack' if attention_index%2==0 else 'half_attention'
        attention_index+=1
    elif category in ['MatMul','BF16MatVec']:category='projections'
    counts[category]+=1;sites[(row['kind'],row['site'])]+=1
    if row['end']>row['start']:
        points[row['start']].append((category,1));points[row['end']].append((category,-1))
assert attention_index==64
active=Counter();exclusive=defaultdict(float);covered=0.;overlap=0.;gaps=0.;previous=None
for timestamp,changes in sorted(points.items()):
    if previous is not None:
        length=(timestamp-previous)*scale;kinds={k for k,v in active.items() if v}
        if kinds:
            covered+=length
            if len(kinds)==1:exclusive[next(iter(kinds))]+=length
            else:overlap+=length
        else:gaps+=length
    for category,delta in changes:
        active[category]+=delta;assert active[category]>=0
    previous=timestamp
assert not any(active.values()) and covered>0
assert abs(sum(exclusive.values())+overlap-covered)<=1e-6
denominator=covered+gaps
attribution=[dict(category=k,encoders=counts[k],exclusive_interval_ms=v,share_of_timestamp_covered_union=v/covered) for k,v in sorted(exclusive.items(),key=lambda item:-item[1])]
negative=[]
mutations={
    'missing_encoder':lambda r:r['rows'].pop(),
    'wrong_kind':lambda r:r['rows'][0].update(kind='unknown'),
    'negative_interval':lambda r:r['rows'][0].update(end=r['rows'][0]['start']-1),
    'outside_calibration':lambda r:r['rows'][0].update(start=r['gpu0']-1),
    'nonfinite_duration':lambda r:r['rows'][0].update(ms=float('nan')),
    'inconsistent_duration':lambda r:r['rows'][0].update(ms=r['rows'][0]['ms']+1),
    'wrong_command':lambda r:r['rows'][0].update(command=r['partial_submissions']+1),
}
for name,mutate in mutations.items():
    r=copy.deepcopy(last);mutate(r)
    try:validate(r)
    except (AssertionError,KeyError,ValueError,TypeError):negative.append(name)
    else:raise AssertionError('Corrupt timestamp evidence admitted: '+name)
result=dict(passed=True,scope=__doc__,diagnostic_only=True,instrumentation_changes_timing=True,
    body_records=19,total_records=len(records),last_body_encoders=len(last['rows']),instrumented_body_driver_gpu_ms=last['gpu_ms'],
    timestamp_covered_union_ms=covered,cross_category_overlap_ms=overlap,uncovered_gaps_within_encoder_envelope_ms=gaps,encoder_envelope_ms=denominator,
    attribution=attribution,site_counts=[dict(kind=k,site=s,encoders=n) for (k,s),n in sorted(sites.items())],
    same_complete_state_hidden_values_and_predictions_as_uninstrumented_candidate=True,negative_controls_rejected=negative,
    raw_stderr_sha256=sha(root/'run1.stderr'),diagnostic_result_sha256=sha(root/'diagnostic1.json'),foundation_provenance_sha256=preflight['foundation_provenance_sha256'],
    serving_speed_claim=False,quality_qualified=False,goal_complete=False,
    limits='Exclusive interval shares apply only to this instrumented encoder timeline. Session includes hoisted KV and other session work. No hardware-counter occupancy, bandwidth, or isolated causal speedup is established.')
dest=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else root/'attribution1.json';assert not dest.exists();dest.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='site_counts'},indent=2))
