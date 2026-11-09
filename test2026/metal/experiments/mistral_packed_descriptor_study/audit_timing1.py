"""Audit saved balanced packed-descriptor timings; no inference."""
import sys
from pathlib import Path
import copy,hashlib,json,math,statistics
root=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
load=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
status=load(root/'timing-status1.json');preflight=load(root/'timing-preflight1.json')
assert status['status']=='collected_pending_audit' and status['hashing_finished_before_collection'] and preflight['hashing_finished_before_collection']
assert load(root/'correctness-audit1.json')['passed']
for p,h in preflight['files_sha256'].items():assert sha(Path(p))==h,p
assert [r['order'] for r in status['runs']]==['forward','reverse']
previous=0;pids=set()
for run in status['runs']:
    assert run['status']=='terminal' and run['returncode']==0 and run['pid'] not in pids
    assert previous<=run['started_ns']<run['finished_ns'];previous=run['finished_ns'];pids.add(run['pid'])
data={order:load(root/('timing-'+order+'1/result1.json')) for order in ['forward','reverse']}
good_guards=load(root/'correctness1/result1.json')['guards']
def verify(d):
    assert set(d)=={'forward','reverse'}
    for order,r in d.items():
        assert r['completed'] and r['mode']=='timing' and r['order']==order and r['timing_admitted']
        assert r['math_mode']=='Safe' and r['relaxed_descriptor_only'] and not r['native_integration'] and not r['quality_qualified'] and not r['serving_speed_claim']
        assert r['guards']==good_guards and len(r['cases'])==72
        assert r['FP64_values']==2160 and 0<=r['maximum_FP64_fraction']<=1
        for i,c in enumerate(r['cases']):
            fi=i//24;round=(i%24)//2-3;step=i%2;variant=(step+round+3+(order=='reverse'))%2
            assert (c['fixture'],c['round'],c['step'],c['variant'])==(fi,round,step,variant)
            assert c['count']==512 and c['dense'] and not c['reverse'] and c['tile']==(128 if fi==2 else 256)
            assert math.isfinite(c['gpu_ms_including_gather']) and c['gpu_ms_including_gather']>0
            assert c['output_file']==(f'case{fi}-n512-dense-v{variant}.f32' if round==8 else '')
verify(data)
comparisons=[];files={}
for order,r in data.items():
    folder=root/('timing-'+order+'1')
    for fi,(n,k) in enumerate([(14336,4096),(4096,14336),(4096,4096)]):
        for v in [0,1]:
            p=folder/f'case{fi}-n512-dense-v{v}.f32'
            assert sha(p)==sha(root/'correctness1'/f'case{fi}-n512-dense-v1.f32')
            files[str(p.relative_to(root))]=sha(p)
        rows=[c for c in r['cases'] if c['fixture']==fi and c['round']>=0]
        samples={v:[c['gpu_ms_including_gather'] for c in rows if c['variant']==v] for v in [0,1]}
        assert all(len(v)==9 for v in samples.values())
        med=[statistics.median(samples[v]) for v in [0,1]]
        ratios=[]
        for round in range(9):
            pair={c['variant']:c['gpu_ms_including_gather'] for c in rows if c['round']==round};assert set(pair)=={0,1};ratios.append(pair[0]/pair[1])
        comparisons.append(dict(order=order,rows=512,outputs=n,inner=k,control_gpu_ms_including_gather=med[0],candidate_gpu_ms_including_gather=med[1],ratio_of_medians=med[0]/med[1],median_within_round_ratio=statistics.median(ratios),within_round_ratio_range=[min(ratios),max(ratios)],candidate_wins=sum(x>1 for x in ratios)))
negative=[]
mutations={
    'missing_cohort':lambda d:d.pop('reverse'),
    'wrong_order':lambda d:d['reverse'].update(order='forward'),
    'missing_call':lambda d:d['forward']['cases'].pop(),
    'duplicate_round':lambda d:d['forward']['cases'][0].update(round=0),
    'wrong_variant':lambda d:d['forward']['cases'][0].update(variant=1),
    'wrong_geometry':lambda d:d['forward']['cases'][0].update(tile=128),
    'nonfinite':lambda d:d['forward']['cases'][0].update(gpu_ms_including_gather=float('nan')),
    'zero_duration':lambda d:d['forward']['cases'][0].update(gpu_ms_including_gather=0),
    'bad_guard':lambda d:d['forward']['guards'][0]['values'].__setitem__(0,1),
    'wrong_sample_coverage':lambda d:d['forward'].update(FP64_values=0),
}
for label,mutate in mutations.items():
    bad=copy.deepcopy(data);mutate(bad)
    try:verify(bad)
    except (AssertionError,KeyError,ValueError,TypeError):negative.append(label)
    else:raise AssertionError('Corrupt timing admitted: '+label)
result=dict(passed=True,scope=__doc__,comparisons=comparisons,fresh_serial_processes=2,measured_gpu_commands=108,warmup_gpu_commands=36,
    gather_included=True,complete_parent_outputs_exact_to_qualified_correctness=True,negative_controls_rejected=negative,
    payload_sha256=files,correctness_audit_sha256=sha(root/'correctness-audit1.json'),preflight_sha256=sha(root/'timing-preflight1.json'),
    runtime_changed=False,native_integration=False,quality_qualified=False,serving_speed_claim=False,goal_complete=False)
path=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else root/'timing-audit1.json';assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='payload_sha256'},indent=2))
