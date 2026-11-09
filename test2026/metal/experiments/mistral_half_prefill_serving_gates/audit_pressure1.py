"""Audit bounded host-cache eviction with exact saved uncached half control."""
import sys
from pathlib import Path
import copy,hashlib,json,runpy,re
root=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
base=root.parent/'mistral-half-prefix-http1'
load=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
assert load(root/'launch1-status.json')['status']=='terminal' and load(root/'launch1-status.json')['returncode']==0
b=load(root/'bindings1.json');assert sha(base/'half-http-audit1.json')==b['baseline_half_http_audit_sha256']
assert sha(base/'uncached/server/worker-stdout.jsonl')==b['baseline_uncached_worker_stdout_sha256']
for p,h in load(root/'preflight-hashes1.json').items():assert sha(Path(p))==h,p
fixture=load(root/'fixture2.json');assert sha(root/'fixture2.json')==sha(base/'fixture2.json')
status=load(root/'status1.json');assert status['status']=='collected_pending_independent_audit' and len(status['phases'])==1
module=runpy.run_path(str(root/'audit_pressure_protocol1.py'))
reference_module=runpy.run_path(str(base/'audit_http2.py'))
data=module['load']('pressure');reference=reference_module['load']('uncached')
def target_states(item):
    es=[v['event'] for v in item['raw']];result={}
    for request in fixture['requests']:
        name=request['label'];final=next(e for e in item['clients'][name]['events'] if isinstance(e,dict) and 'response_token_ids' in e)
        complete=next(e for e in es if e.get('event')=='completed' and e['id']==final['id'])
        witnesses=[e for e in es if e.get('event')=='native_request_state_witness' and e['id']==final['id']]
        assert [e['phase'] for e in witnesses]==['prefill','final']
        for e in witnesses:
            assert (e['native_request'],e['row'],e['epoch'])==(complete['native_request'],complete['native_row'],complete['native_epoch'])
            assert len(e['state']['kv'])==32 and all(len(pair)==2 for pair in e['state']['kv'])
            assert all(re.fullmatch('[0-9a-f]{64}',h) for pair in e['state']['kv'] for h in pair) and re.fullmatch('[0-9a-f]{64}',e['state']['hidden'])
        result[name]={e['phase']:e['state'] for e in witnesses}
    return result
reference_states=target_states(reference)
def verify(d):
    protocol=module['check'](d,fixture,status['phases'][0])
    es=[v['event'] for v in d['raw']];ready=next(e for e in es if e.get('event')=='ready')
    assert ready['profile']['prefix_host_bytes']==83886080 and ready['profile']['half_attention_policy']
    assert 'fp16_prefill_all_p16_gpupack_v1' in ready['backend']
    for request in fixture['requests']:
        name=request['label'];a=d['clients'][name];c=reference['clients'][name]
        final=lambda item:next(e for e in item['events'] if isinstance(e,dict) and 'response_token_ids' in e)
        assert final(a)['response_token_ids']==final(c)['response_token_ids'] and a['result']['text']==c['result']['text']
    assert target_states(d)==reference_states,'Eviction changed complete target KV/hidden state'
    halves=[e for e in es if e.get('event')=='diagnostic_half_attention'];hoist=[e for e in es if e.get('event')=='diagnostic_native_kv_hoist']
    assert len(halves)==len(hoist)>0
    assert all(h['dispatches']==(32 if len(k['ids'])>4 else 0) and h['scratch_bytes']==71303552 for h,k in zip(halves,hoist))
    assert all(e['dispatches']==0 for e in es if e.get('event')=='native_chunk_attention_mps')
    stats=[e['stats'] for e in es if e.get('event')=='native_prefix_stats'];assert stats
    assert max(s['accounted_host_bytes'] for s in stats)<=83886080 and stats[-1]['evictions']>0
    return dict(protocol=protocol,maximum_observed_accounted_host_bytes=max(s['accounted_host_bytes'] for s in stats),evictions=stats[-1]['evictions'],
        half_prefill_dispatches=sum(e['dispatches'] for e in halves),complete_target_state_digest_pairs=1170,all_generated_ids_and_text_exact_to_uncached=True)
result=verify(data);negative=[]
def mutate_stats(d,key,value):
    for row in d['raw']:
        if row['event'].get('event')=='native_prefix_stats':row['event']['stats'][key]=value
def corrupt_target(d):next(v['event'] for v in d['raw'] if v['event'].get('event')=='native_request_state_witness')['state']['hidden']='0'*64
mutations={
    'budget_overrun':lambda d:mutate_stats(d,'accounted_host_bytes',83886081),
    'no_eviction':lambda d:mutate_stats(d,'evictions',0),
    'wrong_target_hidden':corrupt_target,
    'wrong_half_route':lambda d:next(v['event'] for v in d['raw'] if v['event'].get('event')=='diagnostic_half_attention').update(dispatches=0),
    'wrong_reuse':lambda d:d['clients']['A-return']['result'].update(cached_prompt_tokens=129),
    'owner_leak':lambda d:next(v['event'] for v in d['raw'] if v['event'].get('event')=='closed')['memory'].update(garnet_live_matrix_owners=1),
}
for name,mutate in mutations.items():
    bad=copy.deepcopy(data);mutate(bad)
    try:verify(bad)
    except (AssertionError,KeyError,ValueError,TypeError):negative.append(name)
    else:raise AssertionError('Corrupt pressure result admitted: '+name)
del result['protocol']['generated_ids']
result.update(passed=True,scope=__doc__,negative_controls_rejected=negative,configured_host_prefix_budget_bytes=83886080,
    pressure_profile_tested=True,general_oom_or_all_failure_paths_proven=False,quality_qualified=False,serving_speed_claim=False,goal_complete=False,
    worker_stdout_sha256=sha(root/'pressure/server/worker-stdout.jsonl'),baseline_uncached_worker_stdout_sha256=b['baseline_uncached_worker_stdout_sha256'])
dest=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else root/'pressure-audit1.json';assert not dest.exists();dest.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
