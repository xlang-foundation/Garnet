"""Audit saved real-weight prefill body timings; never runs inference."""
import sys
from pathlib import Path
import copy,hashlib,json,math,re,statistics
root=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
study=root.parent
load=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
b=load(root/'bindings1.json');status=load(root/'collector-status1.json')
assert status['completed'] and status['status']=='collected_pending_audit' and status['source_hashing_finished_before_collection']
assert load(study/'model-state-audit7.json')['passed']
for path,h in load(root/'preflight-hashes1.json').items():assert sha(Path(path))==h,path
assert sha(Path(b['body_worker']))==b['body_worker_sha256']
expected=[(0,1,0),(0,1,1),(0,4,0),(0,4,1),(1,4,1),(1,4,0),(1,1,1),(1,1,0)]
previous=0;pids=set();data={};files={}
for identity,run in zip(expected,status['runs']):
    cohort,owners,enabled=identity;mode='on' if enabled else 'off';label=f'cohort{cohort}-owners{owners}-{mode}'
    assert (run['cohort'],run['owners'],run['enabled'])==identity
    assert run['status']=='terminal' and run['returncode']==0 and run['pid'] not in pids
    assert previous<=run['started_ns']<run['finished_ns'];previous=run['finished_ns'];pids.add(run['pid'])
    assert run['argv']==[b['xlang'],'-B',b['body_worker'],b['libraries'][mode]['path'],b['source'],b['fixture'],str(root/(label+'.json')),str(owners),str(enabled)]
    path=root/(label+'.json');data[identity]=load(path);files[path.name]=sha(path)
assert len(status['runs'])==len(expected)
owner_keys=['garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets']
def flat(v):return [x for item in v for x in flat(item)] if isinstance(v,list) else [v]
def verify_case(d,identity):
    cohort,owners,enabled=identity;mode='on' if enabled else 'off'
    assert d['completed'] and d['library']==b['libraries'][mode]['path']+'/libgarnet.dylib'
    assert (d['owners'],d['enabled'],d['batch'],d['capacity'],d['chunk'],d['offset'],d['warmups'],d['repeats'])==(owners,enabled,4,4096,128,1920,3,9)
    assert ('fp16_prefill_all_p16_gpupack_v1' in d['backend'])==bool(enabled)
    assert 'committed_kv_mps_fp32_v1' not in d['backend'] and 'chunk_flash_attention_fp32_operands' not in d['backend']
    assert [v['offset'] for v in d['seeds']]==list(range(0,1920,128))
    assert all(math.isfinite(v['wall_seconds']) and v['wall_seconds']>0 for v in d['seeds'])
    assert len(d['measured'])==9
    for i,v in enumerate(d['measured']):
        assert (v['repeat'],v['committed_dispatches'],v['mps_dispatches'],v['half_dispatches'])==(i,0,32*(1-enabled),32*enabled)
        assert math.isfinite(v['wall_seconds']) and math.isfinite(v['gpu_ms'])
        assert v['wall_seconds']>0 and 0<v['gpu_ms']<=v['wall_seconds']*1000+1
    a,c,closed=d['prepared'],d['after'],d['cleanup']
    assert a['metal_allocated_bytes']==c['metal_allocated_bytes']
    assert c['metal_allocated_bytes']<=c['metal_recommended_working_set_bytes']-256*1024*1024
    for key in owner_keys:assert a[key]==c[key] and closed[key]==0
    assert closed['metal_allocated_bytes']<=4194304
    assert len(d['state_sha256'])==32 and all(len(pair)==2 and all(re.fullmatch('[0-9a-f]{64}',h) for h in pair) for pair in d['state_sha256'])
    assert re.fullmatch('[0-9a-f]{64}',d['hidden_sha256'])
    assert len(d['predictions'])==4 and all((type(p) is int and 0<=p<32768) if row<owners else p==-1 for row,p in enumerate(d['predictions']))
    hidden=flat(d['hidden_values']);assert len(hidden)==4*128*4096 and all(math.isfinite(v) for v in hidden)
for identity,d in data.items():
    verify_case(d,identity)
    cohort,owners,enabled=identity;mode='on' if enabled else 'off'
    assert 'half_prefill_stage_complete' in (root/f'cohort{cohort}-owners{owners}-{mode}.stdout').read_text()

comparisons=[]
for owners in [1,4]:
    for enabled in [0,1]:
        a,c=data[0,owners,enabled],data[1,owners,enabled]
        for field in ['state_sha256','hidden_sha256','predictions','hidden_values']:assert a[field]==c[field],('Same-policy cross-cohort nondeterminism',owners,enabled,field)
    for cohort in [0,1]:
        a,c=data[cohort,owners,0],data[cohort,owners,1]
        assert a['predictions']==c['predictions'],'Next-token regression'
        wall={k:statistics.median(v['wall_seconds'] for v in d['measured']) for k,d in [('off',a),('on',c)]}
        gpu={k:statistics.median(v['gpu_ms'] for v in d['measured']) for k,d in [('off',a),('on',c)]}
        x,y=flat(a['hidden_values']),flat(c['hidden_values']);delta=[abs(u-v) for u,v in zip(x,y)]
        comparisons.append(dict(owners=owners,cohort=cohort,control_body_wall_s=wall['off'],candidate_body_wall_s=wall['on'],wall_control_over_candidate=wall['off']/wall['on'],
            control_body_gpu_ms=gpu['off'],candidate_body_gpu_ms=gpu['on'],gpu_control_over_candidate=gpu['off']/gpu['on'],predictions_equal=True,
            full_hidden_max_absolute_difference=max(delta),full_hidden_rms_difference=math.sqrt(sum(v*v for v in delta)/len(delta))))

# These are deliberate corruptions of real timing rows; exercise the same
# verifier used above without writing into frozen or newly collected payloads.
identity=(0,4,1);original=data[identity];rejected=[]
mutations={
    'missing_repeat':lambda d:d['measured'].pop(),
    'duplicate_repeat':lambda d:d['measured'][1].update(repeat=0),
    'wrong_half_route':lambda d:d['measured'][0].update(half_dispatches=0),
    'wrong_mps_route':lambda d:d['measured'][0].update(mps_dispatches=32),
    'nonfinite_wall':lambda d:d['measured'][0].update(wall_seconds=float('nan')),
    'zero_gpu':lambda d:d['measured'][0].update(gpu_ms=0),
    'gpu_exceeds_wall':lambda d:d['measured'][0].update(gpu_ms=d['measured'][0]['wall_seconds']*1000+2),
    'device_growth':lambda d:d['after'].update(metal_allocated_bytes=d['after']['metal_allocated_bytes']+16384),
    'owner_leak':lambda d:d['cleanup'].update(garnet_live_matrix_owners=1),
    'close_limit':lambda d:d['cleanup'].update(metal_allocated_bytes=4194305),
    'missing_seed':lambda d:d['seeds'].pop(),
}
for name,mutate in mutations.items():
    d=copy.deepcopy(original);mutate(d)
    try:verify_case(d,identity)
    except (AssertionError,KeyError,ValueError,TypeError):rejected.append(name)
    else:raise AssertionError('Corrupt body timing admitted: '+name)
result=dict(passed=True,scope=b['body_scope'],comparisons=comparisons,measured_body_calls=72,warmup_body_calls=24,fresh_serial_engines=8,
    same_policy_cross_cohort_full_state_and_predictions_exact=True,cross_policy_next_predictions_equal=True,negative_controls_rejected=rejected,
    postflight_payload_sha256=files,preflight_hashes_sha256=sha(root/'preflight-hashes1.json'),
    new_native_default=False,full_model_numerical_oracle=False,coding_quality_qualified=False,prefix_reuse_qualified=False,serving_speed_claim=False,goal_complete=False)
dest=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else root/'body-audit7.json';assert not dest.exists();dest.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='postflight_payload_sha256'},indent=2))
