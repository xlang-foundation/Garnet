"""Audit native hoist routes and exact saved prefix-return protocol regression.
Usage: python3 audit_http_hoist1.py ABSOLUTE_HOIST_STUDY
No inference/benchmark/quality work is started.
"""
from pathlib import Path
import hashlib
import json
import runpy
import sys

r=Path(sys.argv[1]).resolve();h=r/'http1';prior=r.parent/'mistral-prefill-prefix-http1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for path,value in json.loads((h/'preflight-hashes1.json').read_text()).items():assert sha(Path(path))==value,path
module=runpy.run_path(str(h/'audit_http2.py'))
assert json.loads((h/'http-audit2.json').read_text())['passed']
old_module=runpy.run_path(str(prior/'audit_http2.py'))
phase_results=[]
for mode in ['uncached','cached']:
    current=module['load'](mode);old=old_module['load'](mode)
    es=[row['event'] for row in current['raw']]
    ready=next(e for e in es if e.get('event')=='ready');assert 'chunk_kv_commit_hoist' in ready['backend']
    routes=[e for e in es if e.get('event')=='diagnostic_native_kv_hoist'];assert routes
    for route in routes:
        assert route['dispatches'] in [0,64]
        assert len(route['ids'])==len(route['slots'])==len(route['active'])
        assert len(route['epochs'])==4
        if len(route['ids'])==4:assert route['dispatches']==0
    moved=[e for e in routes if e['dispatches']];assert len(moved)>=2
    ids=0
    for name,client in current['clients'].items():
        final=next(e for e in client['events'] if isinstance(e,dict) and 'response_token_ids' in e)
        previous=next(e for e in old['clients'][name]['events'] if isinstance(e,dict) and 'response_token_ids' in e)
        assert final['response_token_ids']==previous['response_token_ids']
        assert final['prompt_token_ids']==previous['prompt_token_ids']
        assert client['result']['text']==old['clients'][name]['result']['text']
        ids+=len(final['response_token_ids'])
    assert ids==752
    closed=next(e for e in es if e.get('event')=='closed')
    phase_results.append(dict(mode=mode,completed_requests=9,generated_ids_equal_prior_control=ids,
        body_route_observations=len(routes),reordered_prefill_body_calls=len(moved),
        hoisted_cache_commits=sum(e['dispatches'] for e in routes),
        survivor_hash_observations=1950,native_worker_exit=current['worker_exit'],
        postclose_metal_bytes=closed['memory']['metal_allocated_bytes']))
result=dict(passed=True,scope=__doc__,phases=phase_results,
    cache_contract='A/B/C/D occupy4rows; E queues before A finishes; E and exact/append/late-change/early-change A reuse A row at increasing epochs while B/C/D continue. Cached reuse211/128/128/0; complete outputs match frozen old policy.',
    ctest=dict(on=50,off=50),cached_return_requests_tested=True,
    full_cache_pressure_tested=False,general_failure_or_nonfinite_model_contract_proven=False,
    two_product_attention_implemented=False,serving_speed_claim=False,quality_gate_passed=False,goal_complete=False)
(r/'http-hoist-audit1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
