"""Audit fused MLP pressure coverage after the unchanged pressure invariants."""
from pathlib import Path
import copy,hashlib,json,runpy,sys
ROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
DEST=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else ROOT/'fused-pressure-audit1.json'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
argv=sys.argv[:]
try:
    inherited=DEST.with_name(DEST.stem+'-inherited.json')
    sys.argv=[str(ROOT/'inherited-audit_pressure1.py'),str(ROOT),str(inherited)]
    module=runpy.run_path(str(ROOT/'inherited-audit_pressure1.py'))
finally:sys.argv=argv
assert read(ROOT/'postflight1.json')['passed']
data=module['data'];inventory=read(ROOT/'pressure/engine-inventory1.json')
def verify(d,inv):
    module['verify'](d)
    es=[r['event'] for r in d['raw']];select=lambda k:[e for e in es if e.get('event')==k]
    ready=select('ready');assert len(ready)==1 and ready[0]['profile']['fused_mlp_prefill'] is True
    assert 'fused_mistral_mlp_prefill_fast_v1' in ready[0]['backend']
    assert select('native_fused_mlp_prepared')==[dict(event='native_fused_mlp_prepared',sites=[1]*32,scratch_bytes=14680064,producer='fused-mistral-mlp-fast-v1')]
    fused=select('diagnostic_fused_mlp');routes=select('diagnostic_native_kv_hoist');assert len(fused)==len(routes)>0
    for event,route in zip(fused,routes):
        active=sum(x!=0 for x in route['active']);blocks=32 if len(route['ids'])>4 and active>=128 else 0
        assert event==dict(event='diagnostic_fused_mlp',blocks=blocks,active_tokens=active,logical_affine_products=blocks*3,producer_kernels=blocks,down_kernels=blocks)
    assert sum(e['blocks'] for e in fused)>0
    refs=[r for r in inv['references'] if r['role']=='fused_mlp_half_input'];assert len(refs)==1
    arena=next(b for b in inv['buffers'] if b['id']==refs[0]['buffer_id']);assert arena['length_bytes']==arena['allocated_bytes']==14680064
    assert len([r for r in inv['references'] if r['buffer_id']==arena['id']])==1
    return sum(e['blocks'] for e in fused)
blocks=verify(data,inventory);rejected=[]
def mutate(d,i):next(r['event'] for r in d['raw'] if r['event'].get('event')=='diagnostic_fused_mlp')['blocks']=-1
for name,change in {
    'wrong_fused_blocks':mutate,
    'missing_fused_site':lambda d,i:next(r['event'] for r in d['raw'] if r['event'].get('event')=='native_fused_mlp_prepared')['sites'].__setitem__(0,0),
    'missing_private_arena':lambda d,i:i.update(references=[r for r in i['references'] if r['role']!='fused_mlp_half_input']),
}.items():
    d=copy.deepcopy(data);i=copy.deepcopy(inventory);change(d,i)
    try:verify(d,i)
    except (AssertionError,KeyError,ValueError,TypeError):rejected.append(name)
    else:raise AssertionError('Corrupt fused pressure data admitted: '+name)
result=dict(read(inherited));result.update(scope=__doc__,fused_blocks=blocks,fused_native_pressure_regression_passed=True,
    inherited_audit_sha256=sha(inherited),additional_negative_controls_rejected=rejected,inventory_sha256=sha(ROOT/'pressure/engine-inventory1.json'))
assert not DEST.exists();DEST.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
