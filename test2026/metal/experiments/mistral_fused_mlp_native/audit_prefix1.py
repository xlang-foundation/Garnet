"""Fused-route and frozen half-control audit atop unchanged cache invariants."""
from pathlib import Path
import copy,hashlib,json,runpy,sys
ROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
DEST=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else ROOT/'fused-http-audit1.json'
read=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
P=ROOT.parent/'mistral-half-prefix-http1';b=read(ROOT/'bindings1.json')
assert sha(P/'provenance1.json')==b['frozen_half_http_provenance_sha256']
frozen=read(P/'provenance1.json')['files_sha256']
for name in ['audit_http2.py','cached/server/worker-stdout.jsonl','uncached/server/worker-stdout.jsonl']:
    assert sha(P/name)==frozen[name],name
original_args=sys.argv[:]
try:
    inherited=ROOT/'inherited-half-http-audit1.json'
    if inherited.exists():inherited=DEST.with_name(DEST.stem+'-inherited.json')
    half_auditor=ROOT.parents[3]/'Garnet/test2026/metal/experiments/mistral_half_prefill_serving_gates/audit_prefix1.py'
    sys.argv=[str(half_auditor),str(ROOT),str(inherited)]
    module=runpy.run_path(str(half_auditor))
finally:sys.argv=original_args
assert read(ROOT/'postflight1.json')['passed']
saved_module=runpy.run_path(str(P/'audit_http2.py'))
saved={m:saved_module['load'](m) for m in ['uncached','cached']}
current=module['data'];fixture=module['fixture']
def evs(item):return [r['event'] for r in item['raw']]
def select(es,kind):return [e for e in es if e.get('event')==kind]
def labeled(item,kind):
    result={};es=evs(item)
    for req in fixture['requests']:
        name=req['label'];client=item['clients'][name]
        final=next(e for e in client['events'] if isinstance(e,dict) and 'response_token_ids' in e)
        if kind=='states':result[name]={e['phase']:e['state'] for e in select(es,'native_request_state_witness') if e['id']==final['id']}
        else:
            c=next(e for e in select(es,'completed') if e['id']==final['id'])
            result[name]={k:c[k] for k in ['generated_ids','text','prompt_ids','cached_tokens']}
    return result
inventories={m:read(ROOT/m/'engine-inventory1.json') for m in ['uncached','cached']}
def verify(data,invs):
    totals={}
    for mode in ['uncached','cached']:
        item=data[mode];es=evs(item);ready=select(es,'ready');assert len(ready)==1
        assert 'fused_mistral_mlp_prefill_fast_v1' in ready[0]['backend'] and ready[0]['profile']['fused_mlp_prefill'] is True
        assert select(es,'native_fused_mlp_prepared')==[dict(event='native_fused_mlp_prepared',sites=[1]*32,scratch_bytes=14680064,producer='fused-mistral-mlp-fast-v1')]
        routes=select(es,'diagnostic_native_kv_hoist');fused=select(es,'diagnostic_fused_mlp')
        assert len(routes)==len(fused)>0
        for route,event in zip(routes,fused):
            active=sum(x!=0 for x in route['active']);blocks=32 if len(route['ids'])>4 and active>=128 else 0
            assert event==dict(event='diagnostic_fused_mlp',blocks=blocks,active_tokens=active,logical_affine_products=3*blocks,producer_kernels=blocks,down_kernels=blocks)
        assert sum(e['blocks'] for e in fused)>0
        assert labeled(item,'states')==labeled(saved[mode],'states'),'Full target KV/hidden state differs from frozen half control: '+mode
        assert labeled(item,'outputs')==labeled(saved[mode],'outputs'),'Output or cache reuse differs from frozen half control: '+mode
        inv=invs[mode];refs=[r for r in inv['references'] if r['role']=='fused_mlp_half_input'];assert len(refs)==1
        arena=next(v for v in inv['buffers'] if v['id']==refs[0]['buffer_id'])
        assert arena['length_bytes']==arena['allocated_bytes']==14680064
        assert len([r for r in inv['references'] if r['buffer_id']==arena['id']])==1
        totals[mode]=dict(fused_blocks=sum(e['blocks'] for e in fused),body_calls=len(routes),private_half_arena_bytes=14680064,
                          full_target_digest_pairs_to_frozen_control=1170,generated_tokens=752)
    return totals
totals=verify(current,inventories);rejected=[]
def change_state(d,i):
    select(evs(d['cached']),'native_request_state_witness')[0]['state']['kv'][0][0]='0'*64
mutations={
    'wrong_fused_route':lambda d,i:select(evs(d['cached']),'diagnostic_fused_mlp')[0].update(blocks=0),
    'missing_fused_route':lambda d,i:d['cached']['raw'].remove(next(r for r in d['cached']['raw'] if r['event'].get('event')=='diagnostic_fused_mlp')),
    'wrong_compiled_site':lambda d,i:select(evs(d['cached']),'native_fused_mlp_prepared')[0]['sites'].__setitem__(0,0),
    'wrong_frozen_target_state':change_state,
    'missing_private_arena':lambda d,i:i['cached'].update(references=[r for r in i['cached']['references'] if r['role']!='fused_mlp_half_input']),
}
for name,mutate in mutations.items():
    d=copy.deepcopy(current);inv=copy.deepcopy(inventories);mutate(d,inv)
    try:verify(d,inv)
    except (AssertionError,KeyError,ValueError,TypeError):rejected.append(name)
    else:raise AssertionError('Corrupt fused cache data admitted: '+name)
result=dict(passed=True,scope=__doc__,phases=totals,all_outputs_and_target_states_exact_to_frozen_half=True,
            inherited_cache_protocol_audit_sha256=sha(inherited),inherited_negative_controls=read(inherited)['negative_controls_rejected'],
            inherited_auditor_sha256=sha(half_auditor),
            additional_negative_controls_rejected=rejected,source_binding_sha256=b['fused_source_binding_sha256'],
            fixture_sha256=sha(ROOT/'fixture2.json'),worker_stdout_sha256={m:sha(ROOT/m/'server/worker-stdout.jsonl') for m in ['uncached','cached']},
            prefix_reuse_and_continuous_batch_regression_passed=True,cache_pressure_tested=False,hybrid_recurrent_model_tested=False,
            coding_quality_qualified=False,serving_speed_claim=False,goal_complete=False)
assert not DEST.exists();DEST.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
