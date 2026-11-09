"""Audit native half-policy HTTP row ownership, cache reuse and target state."""
import sys
from pathlib import Path
import copy,hashlib,json,re,runpy
root=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
load_json=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
assert load_json(root/'launch1-status.json')['returncode']==0
for p,h in load_json(root/'preflight-hashes1.json').items():assert sha(Path(p))==h,p
module=runpy.run_path(str(root/'audit_http2.py'))
fixture=load_json(root/'fixture2.json');status=load_json(root/'status1.json')
assert status['status']=='collected_pending_independent_audit'
assert [s['label'] for s in status['phases']]==['uncached','cached']
data={mode:module['load'](mode) for mode in ['uncached','cached']}
states={mode:state for mode,state in zip(['uncached','cached'],status['phases'])}
def verify(d):
    results={};target={};routes={}
    for mode in ['uncached','cached']:
        item=d[mode];results[mode]=module['check'](item,fixture,states[mode])
        es=[v['event'] for v in item['raw']]
        ready=next(e for e in es if e.get('event')=='ready')
        assert 'fp16_prefill_all_p16_gpupack_v1' in ready['backend']
        assert ready['profile']['half_attention_policy'] is True
        assert 'committed_kv_mps_fp32_v1' not in ready['backend']
        prepared=[e for e in es if e.get('event')=='native_chunk_half_attention_prepared']
        assert prepared==[dict(event='native_chunk_half_attention_prepared',enabled=True,scratch_bytes=71303552)]
        hoist=[e for e in es if e.get('event')=='diagnostic_native_kv_hoist']
        half=[e for e in es if e.get('event')=='diagnostic_half_attention']
        assert len(half)==len(hoist)>0
        for h,k in zip(half,hoist):
            prefill=len(k['ids'])>4
            assert h['dispatches']==(32 if prefill else 0) and h['scratch_bytes']==71303552
            assert k['dispatches']==(64 if prefill else 0)
        mps=[e for e in es if e.get('event')=='native_chunk_attention_mps']
        assert mps and all(e['dispatches']==0 for e in mps)
        assert all(e['dispatches']==0 for e in es if e.get('event')=='native_chunk_attention_index32')
        assert all(e['matrices']==e['skipped_matrices']==0 for e in es if e.get('event')=='native_chunk_attention_request_products')
        target[mode]={};witness=[e for e in es if e.get('event')=='native_request_state_witness'];assert len(witness)==18
        for request in fixture['requests']:
            label=request['label'];client=item['clients'][label]
            final=next(e for e in client['events'] if isinstance(e,dict) and 'response_token_ids' in e)
            complete=next(e for e in es if e.get('event')=='completed' and e['id']==final['id'])
            rows=[e for e in witness if e['id']==final['id']]
            assert [e['phase'] for e in rows]==['prefill','final']
            for row in rows:
                assert row['native_request']==complete['native_request'] and row['row']==complete['native_row'] and row['epoch']==complete['native_epoch']
                state=row['state'];assert len(state['kv'])==32 and all(len(pair)==2 for pair in state['kv'])
                assert all(re.fullmatch('[0-9a-f]{64}',value) for pair in state['kv'] for value in pair) and re.fullmatch('[0-9a-f]{64}',state['hidden'])
            target[mode][label]={e['phase']:e['state'] for e in rows}
        routes[mode]=dict(body_calls=len(hoist),prefill_body_calls=sum(len(e['ids'])>4 for e in hoist),half_dispatches=sum(e['dispatches'] for e in half),hoisted_commits=sum(e['dispatches'] for e in hoist),executed_mps_attention_products=0)
    assert results['uncached']['generated_ids']==results['cached']['generated_ids'],'Cached and uncached generated sequences differ'
    assert target['uncached']==target['cached'],'Cached and uncached full target KV/hidden states differ'
    for mode in ['uncached','cached']:
        assert target[mode]['A']['prefill']==target[mode]['A-return']['prefill'],'Exact request return changed prefill state'
    return dict(protocol=results,routes=routes,full_target_state_digest_pairs=9*2*65,generated_ids_per_phase=752,complete_target_states_exact=True)
result=verify(data)
negative=[]
def corrupt_target(d):
    event=next(v['event'] for v in d['cached']['raw'] if v['event'].get('event')=='native_request_state_witness')
    event['state']['kv'][0][0]='0'*64
def corrupt_target_hidden(d):
    event=next(v['event'] for v in d['cached']['raw'] if v['event'].get('event')=='native_request_state_witness')
    event['state']['hidden']='0'*64
def corrupt_target_epoch(d):
    next(v['event'] for v in d['cached']['raw'] if v['event'].get('event')=='native_request_state_witness')['epoch']+=1
def corrupt_survivor(d):
    event=next(v['event'] for v in d['cached']['raw'] if v['event'].get('event')=='native_survivor_restore_witness');key=next(iter(event['after']));event['after'][key]['hidden']='0'*64
mutations={
    'wrong_half_route':lambda d:next(v['event'] for v in d['cached']['raw'] if v['event'].get('event')=='diagnostic_half_attention').update(dispatches=0),
    'missing_half_route':lambda d:d['cached']['raw'].remove(next(v for v in d['cached']['raw'] if v['event'].get('event')=='diagnostic_half_attention')),
    'wrong_target_KV':corrupt_target,'wrong_target_hidden':corrupt_target_hidden,'wrong_target_epoch':corrupt_target_epoch,
    'wrong_survivor_hidden':corrupt_survivor,
    'wrong_cache_hit':lambda d:d['cached']['clients']['A-return']['result'].update(cached_prompt_tokens=0),
    'owner_leak':lambda d:next(v['event'] for v in d['cached']['raw'] if v['event'].get('event')=='closed')['memory'].update(garnet_live_matrix_owners=1),
}
for name,mutate in mutations.items():
    bad=copy.deepcopy(data);mutate(bad)
    try:verify(bad)
    except (AssertionError,KeyError,ValueError,TypeError):negative.append(name)
    else:raise AssertionError('Corrupt cached-state evidence admitted: '+name)
for mode in ['uncached','cached']:del result['protocol'][mode]['generated_ids']
result.update(passed=True,scope=__doc__,negative_controls_rejected=negative,native_runtime_changed=False,
    row_reuse_append_edit_and_queue_tested=True,full_cache_pressure_tested=False,hybrid_recurrent_model_tested=False,
    quality_qualified=False,serving_speed_claim=False,goal_complete=False,
    binding_sha256=sha(root/'bindings1.json'),fixture_sha256=sha(root/'fixture2.json'),
    worker_stdout_sha256={m:sha(root/m/'server/worker-stdout.jsonl') for m in ['uncached','cached']})
dest=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else root/'half-http-audit1.json';assert not dest.exists();dest.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
