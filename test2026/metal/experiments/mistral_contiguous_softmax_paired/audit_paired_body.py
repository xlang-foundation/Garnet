"""Independently audit serial within-engine model-body measurements; no inference.
Usage: python3 audit_paired_body1.py ABSOLUTE_STUDY
"""
from pathlib import Path
import hashlib
import json
import math
import re
import statistics
import sys

r=Path(sys.argv[1]).resolve()
load=lambda name:json.loads((r/name).read_text())
b=load('bindings1.json');status=load('collector-status1.json');plan=load('plan1.json')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
assert status['completed'] and status['status']=='collected_pending_audit' and status['source_hashing_finished_before_collection']
assert load('build-test-status1.json')['completed']
assert load('ctest-on1-status.json')['returncode']==load('ctest-off1-status.json')['returncode']==0
assert '100% tests passed out of 50' in (r/'ctest-on1.stdout').read_text()
assert '100% tests passed out of 1' in (r/'ctest-off1.stdout').read_text()
for path,h in load('preflight-hashes1.json').items():assert sha(Path(path))==h,path
assert sha(Path(b['worker']))==b['worker_sha256']
assert sha(Path(b['library'])/'libgarnet.dylib')==b['library_sha256']
assert sha(r/'source-audit1.json')==b['source_audit_sha256']
for name,h in load('source-audit1.json')['candidate_owned_files_sha256'].items():assert sha(Path(b['source'])/name)==h
expected=[[0,1,0],[0,4,0],[1,4,1],[1,1,1]];assert plan['order']==expected and len(status['runs'])==4
owner_keys=['garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets']
previous=0;pids=set();results=[];digest_pairs=0
for identity,run in zip(expected,status['runs']):
    cohort,owners,initial=identity;label=f'cohort{cohort}-owners{owners}'
    assert [run['cohort'],run['owners'],run['initial']]==identity
    assert run['status']=='terminal' and run['returncode']==0 and run['pid'] not in pids
    assert previous<=run['started_ns']<run['finished_ns'];previous=run['finished_ns'];pids.add(run['pid'])
    assert run['argv']==[b['xlang'],'-B',b['worker'],b['library'],b['source'],b['fixture'],str(r/(label+'.json')),str(owners),str(initial)]
    data=load(label+'.json')
    assert data['completed'] and data['closed_selector_rejected']
    assert 'chunk_mps_contiguous_softmax_optional' in data['backend']
    assert 'chunk_flash_attention_fp32_operands' not in data['backend']
    assert data['library']==b['library']+'/libgarnet.dylib'
    assert [data[k] for k in ['owners','initial_order','batch','capacity','chunk','offset','warmup_pairs','measured_pairs']]==[owners,initial,4,4096,128,1920,3,9]
    assert [s['offset'] for s in data['seeds']]==list(range(0,1920,128))
    assert len(data['measured'])==18
    times={}
    for i,v in enumerate(data['measured']):
        pair,order=divmod(i,2);enabled=(pair+order+initial)%2
        assert [v['pair'],v['order'],v['enabled'],v['contiguous_dispatches'],v['mps_dispatches']]==[pair,order,enabled,32*enabled,32]
        assert math.isfinite(v['wall_seconds']) and math.isfinite(v['gpu_ms'])
        assert v['wall_seconds']>0 and 0<v['gpu_ms']<=v['wall_seconds']*1000+1
        times[pair,enabled]=v
    a,c,closed=data['prepared'],data['after'],data['cleanup']
    assert a['metal_allocated_bytes']==c['metal_allocated_bytes']
    assert c['metal_allocated_bytes']<c['metal_recommended_working_set_bytes']-256*1024*1024
    for key in owner_keys:assert a[key]==c[key] and closed[key]==0
    assert closed['metal_allocated_bytes']<=4194304
    control=json.loads((Path(b['prior_body_study'])/f'cohort0-owners{owners}-off.json').read_text())
    for field in ['state_sha256','warm_state_sha256']:
        assert len(data[field])==32
        for pair in data[field]:assert len(pair)==2 and all(re.fullmatch('[0-9a-f]{64}',x) for x in pair)
        assert data[field]==control['state_sha256'];digest_pairs+=64
    assert data['warm_hidden_sha256']==data['hidden_sha256']==control['hidden_sha256'];digest_pairs+=2
    assert data['predictions']==control['predictions']
    assert 'paired_contiguous_stage_complete' in (r/(label+'.stdout')).read_text()
    wall=[times[i,0]['wall_seconds']/times[i,1]['wall_seconds'] for i in range(9)]
    gpu=[times[i,0]['gpu_ms']/times[i,1]['gpu_ms'] for i in range(9)]
    results.append(dict(cohort=cohort,owners=owners,
        original_body_median_s=statistics.median(times[i,0]['wall_seconds'] for i in range(9)),
        candidate_body_median_s=statistics.median(times[i,1]['wall_seconds'] for i in range(9)),
        paired_body_ratio_median=statistics.median(wall),paired_body_ratio_range=[min(wall),max(wall)],
        paired_body_faster_pairs=sum(v>1 for v in wall),paired_gpu_ratio_median=statistics.median(gpu),
        paired_gpu_ratio_range=[min(gpu),max(gpu)],complete_state_and_predictions_equal_prior_control=True,
        postclose_metal_bytes=closed['metal_allocated_bytes']))
assert digest_pairs==520
result=dict(passed=True,scope=b['scope'],runs=len(results),measured_body_calls=72,
    comparisons=results,complete_state_digest_pairs_against_frozen_control=digest_pairs,
    ctest=dict(on_full=50,off_focused=1),all_native_owners_closed=True,
    assessment='Within-engine pairing reduces fresh-process drift. Results describe only this repeated late chunk; neither full-prefill nor TTFT/decode/HTTP. Small body effects do not meet2x or justify default promotion without remaining serving/cache/quality gates.',
    new_native_default=False,serving_speed_claim=False,quality_gate_passed=False,goal_complete=False)
(r/'paired-body-audit1.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
