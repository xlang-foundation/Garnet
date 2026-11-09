"""Audit frozen-shape real-weight body work; no inference or benchmark collection."""
from pathlib import Path
import hashlib
import json
import math
import re
import statistics
import sys

r=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
b=json.loads((r/'bindings1.json').read_text());s=json.loads((r/'collector-status1.json').read_text())
assert s['completed'] and s['status']=='collected_pending_audit' and s['source_hashing_finished_before_collection']
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
for path,h in json.loads((r/'preflight-hashes1.json').read_text()).items():assert sha(Path(path))==h,path
assert sha(Path(b['body_worker']))==b['body_worker_sha256']
expected=[(0,1,0),(0,1,1),(0,4,0),(0,4,1),(1,4,1),(1,4,0),(1,1,1),(1,1,0)]
assert len(s['runs'])==len(expected)
previous=0;pids=set();results={}
owner_keys=['garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets']
for identity,run in zip(expected,s['runs']):
    cohort,owners,enabled=identity;mode='on' if enabled else 'off';label=f'cohort{cohort}-owners{owners}-{mode}'
    assert (run['cohort'],run['owners'],run['enabled'])==identity
    assert run['status']=='terminal' and run['returncode']==0 and run['pid'] not in pids
    assert previous<=run['started_ns']<run['finished_ns'];previous=run['finished_ns'];pids.add(run['pid'])
    assert run['argv']==[b['xlang'],'-B',b['body_worker'],b['libraries'][mode]['path'],b['source'],b['fixture'],str(r/(label+'.json')),str(owners),str(enabled)]
    data=json.loads((r/(label+'.json')).read_text())
    assert data['completed'] and data['library']==b['libraries'][mode]['path']+'/libgarnet.dylib'
    assert (data['owners'],data['enabled'],data['batch'],data['capacity'],data['chunk'],data['offset'],data['warmups'],data['repeats'])==(owners,enabled,4,4096,128,1920,3,9)
    assert ('committed_kv_mps_fp32_v1' in data['backend'])==bool(enabled)
    assert 'chunk_flash_attention_fp32_operands' not in data['backend']
    assert [v['offset'] for v in data['seeds']]==list(range(0,1920,128))
    assert all(math.isfinite(v['wall_seconds']) and v['wall_seconds']>0 for v in data['seeds'])
    assert len(data['measured'])==9
    for i,v in enumerate(data['measured']):
        assert (v['repeat'],v['committed_dispatches'],v['mps_dispatches'])==(i,32*enabled,32)
        assert math.isfinite(v['wall_seconds']) and math.isfinite(v['gpu_ms'])
        assert v['wall_seconds']>0 and 0<v['gpu_ms']<=v['wall_seconds']*1000+1
    a,c,closed=data['prepared'],data['after'],data['cleanup']
    assert a['metal_allocated_bytes']==c['metal_allocated_bytes']
    assert c['metal_allocated_bytes']<=c['metal_recommended_working_set_bytes']-256*1024*1024
    for key in owner_keys:assert a[key]==c[key] and closed[key]==0
    assert closed['metal_allocated_bytes']<=4194304
    assert len(data['state_sha256'])==32 and all(len(pair)==2 and all(re.fullmatch('[0-9a-f]{64}',h) for h in pair) for pair in data['state_sha256'])
    assert re.fullmatch('[0-9a-f]{64}',data['hidden_sha256'])
    assert len(data['predictions'])==4 and all((type(p) is int and 0<=p<32768) if row<owners else p==-1 for row,p in enumerate(data['predictions']))
    assert 'committed_stage_complete' in (r/(label+'.stdout')).read_text()
    results[identity]=data
comparisons=[]
for owners in [1,4]:
    for enabled in [0,1]:
        first,second=results[0,owners,enabled],results[1,owners,enabled]
        for field in ['state_sha256','hidden_sha256','predictions','hidden_values']:
            assert first[field]==second[field],('Same-policy opposite-cohort nondeterminism',owners,enabled,field)
    for cohort in [0,1]:
        a,c=results[cohort,owners,0],results[cohort,owners,1]
        assert a['predictions']==c['predictions'],('Next-token regression',owners,cohort)
        wall={k:statistics.median(v['wall_seconds'] for v in d['measured']) for k,d in [('off',a),('on',c)]}
        gpu={k:statistics.median(v['gpu_ms'] for v in d['measured']) for k,d in [('off',a),('on',c)]}
        def flat(v):return [x for sub in v for x in flat(sub)] if isinstance(v,list) else [v]
        x,y=flat(a['hidden_values']),flat(c['hidden_values'])
        assert len(x)==len(y)==4*128*4096 and all(math.isfinite(v) for v in x+y)
        differences=[abs(u-v) for u,v in zip(x,y)]
        comparisons.append(dict(owners=owners,cohort=cohort,control_body_wall_s=wall['off'],candidate_body_wall_s=wall['on'],wall_control_over_candidate=wall['off']/wall['on'],control_body_gpu_ms=gpu['off'],candidate_body_gpu_ms=gpu['on'],gpu_control_over_candidate=gpu['off']/gpu['on'],predictions_equal=True,full_hidden_max_absolute_difference=max(differences),full_hidden_rms_difference=math.sqrt(sum(e*e for e in differences)/len(differences))))
result=dict(passed=True,scope='Two reversed-order cohorts,8 fresh serial engines,72 measured real32-layer body calls. Warm identical late chunk replay after15 actual prefix chunks; excludes embedding/head/hashes/scheduling/HTTP. Ratios of per-process medians, not within-engine paired estimates. Different FP32 reduction grouping; no cross-policy full-state bit-identity assertion.',comparisons=comparisons,same_policy_cross_cohort_full_state_and_predictions_exact=True,cross_policy_next_predictions_equal=True,new_native_default=False,serving_speed_claim=False,quality_gate_passed=False,goal_complete=False)
(r/'body-audit1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
