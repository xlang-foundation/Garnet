"""Audit half-prefill coding regression; an audit pass is not a coding pass."""
import sys
from pathlib import Path
import copy,hashlib,json
root=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
old=root.parent/'mistral-current-coding-quality1'
load=lambda p:json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
b=load(root/'bindings1.json');frozen=load(old/'provenance1.json')['files_sha256']
assert sha(old/'provenance1.json')==b['old_quality_provenance_sha256']
assert load(root/'launch1-status.json')['status']=='terminal' and load(root/'launch1-status.json')['returncode']==0
assert load(root/'status1.json')['status']=='collected_pending_quality_audit'
for p,h in [(root/'fixture1.json',b['fixture_sha256']),(root/'profile1.json',b['profile_sha256']),
    (root/'tools1/local_coder_batch_worker.py',b['worker_sha256']),(root/'tools1/suite.json',b['suite_sha256']),
    (root/'tools1/evaluate_local_coder_quality.py',b['evaluator_sha256']),
    (Path(b['library'])/'libgarnet.dylib',b['library_sha256'])]:assert sha(p)==h,str(p)
assert b['suite_sha256']=='675ea667c357f361ce39e3863b61193dc6b32f4a229eb7d29a17803b75408786'
assert b['evaluator_sha256']=='a73809aab308180338ab1df644afdd8c4970bc136d6b0bc93c049c8862ebcc7a'
def events(path):
    out=[]
    for line in path.read_text().splitlines():
        try:out.append(json.loads(json.loads(line)['line']))
        except ValueError:continue
    return out
def client(base,name):
    p=base/'responses1'/name/'request-0'
    wire=[json.loads(v)['data'] for v in p.with_suffix('.events.jsonl').read_text().splitlines()]
    return dict(result=load(p.with_suffix('.result.json')),request=load(p.with_suffix('.request.json')),events=[v if v=='[DONE]' else json.loads(v) for v in wire])
fixture=load(root/'fixture1.json');suite=load(root/'tools1/suite.json')
for name in ['strict-quality1.json','block-diagnosis1.json']:
    assert sha(old/name)==frozen[name],name
prior_quality=load(old/'strict-quality1.json');prior_diagnosis=load(old/'block-diagnosis1.json')
prior={}
for task in fixture['requests']:
    name=task['id']
    for suffix in ['events.jsonl','result.json','request.json']:
        p=old/'responses1'/name/('request-0.'+suffix);assert sha(p)==frozen[str(p.relative_to(old))]
    prior[name]=client(old,name)
data=dict(native=events(root/'server1/worker-stdout.jsonl'),clients={t['id']:client(root,t['id']) for t in fixture['requests']},strict=load(root/'strict-quality1.json'),diagnosis=load(root/'block-diagnosis1.json'))
def verify(d):
    es=d['native'];assert not any(e.get('event')=='error' for e in es)
    assert load(root/'server1/worker-exit.json')==dict(returncode=0,expected_close=True)
    assert load(root/'status1.json')['shutdown']['graceful_native_closed'] and not load(root/'status1.json')['shutdown']['forced_kill']
    ready=[e for e in es if e.get('event')=='ready'];closed=[e for e in es if e.get('event')=='closed'];assert len(ready)==len(closed)==1
    assert ready[0]['profile']==load(root/'profile1.json') and ready[0]['prefix_cache'] is False and ready[0]['active_requests']==4
    assert 'fp16_prefill_all_p16_gpupack_v1' in ready[0]['backend'] and ready[0]['profile']['half_attention_policy']
    for key in ['garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets']:assert closed[0]['memory'][key]==0,key
    assert closed[0]['memory']['metal_allocated_bytes']<=4194304
    complete={e['id']:e for e in es if e.get('event')=='completed'};assert len(complete)==8
    total=0
    for task in fixture['requests']:
        name=task['id'];c=d['clients'][name];assert c['request']==task['body']==prior[name]['request']
        result=c['result'];assert result['status']=='completed' and result['finish_reason']=='stop' and result['cached_prompt_tokens']==0
        wire=c['events'];assert wire[-1]=='[DONE]' and wire.count('[DONE]')==1
        final=[e for e in wire[:-1] if 'response_token_ids' in e];assert len(final)==1;final=final[0]
        ids=[e['token_id'] for e in wire[:-1] if 'token_id' in e]
        assert ids==final['response_token_ids'] and ids[-1]==2 and len(ids)==result['generated_tokens']<=1024
        assert final['prompt_token_ids']==task['workload']['token_ids']
        native=complete[final['id']];assert native['generated_ids']==ids and native['prompt_ids']==final['prompt_token_ids'] and native['text']==result['text']
        previous=next(e for e in prior[name]['events'] if isinstance(e,dict) and 'response_token_ids' in e)
        assert ids==previous['response_token_ids'] and result['text']==prior[name]['result']['text'],'Coding output changed from frozen FP32 control'
        total+=len(ids)
    assert total==3052
    live={};maximum=0
    for e in es:
        if e.get('event')=='native_batch':
            assert len(e['active_rows'])==4 and sum(e['active_rows'])==len(e['requests'])
            for q in e['requests']:
                assert live.get(q['row']) in (None,(q['id'],q['epoch']));live[q['row']]=(q['id'],q['epoch'])
            maximum=max(maximum,len(live));assert maximum<=4
        elif e.get('event')=='completed':assert live.pop(e['native_row'])==(e['native_request'],e['native_epoch'])
        elif e.get('event')=='native_affine_profile':assert e['misses']==0 and 0<e['pipeline_count']<=512
    assert maximum==4 and not live
    half=[e for e in es if e.get('event')=='native_chunk_half_attention'];assert half and all(e['dispatches']==(32 if e['stage']=='prefill' else 0) for e in half)
    assert any(e['stage']=='prefill' for e in half)
    assert all(e['dispatches']==0 for e in es if e.get('event') in ['native_chunk_attention_mps','native_chunk_attention_index32'])
    q=d['strict'];assert q['sandbox_probe']['status']=='passed' and q['suite_sha256']==b['suite_sha256'] and q['evaluator_sha256']==b['evaluator_sha256']
    assert q['total']==8 and q['passed']==sum(v['status']=='passed' for v in q['results'])==prior_quality['passed']==0
    for row,previous,task in zip(q['results'],prior_quality['results'],suite['tasks']):
        assert row['task']==previous['task']==task['id'] and row['status']==previous['status']=='invalid_code'
        assert row['response_sha256']==sha(root/'responses1'/task['id']/'request-0.result.json')
    diagnosis=d['diagnosis'];assert diagnosis['diagnostic_only'] and diagnosis['strict_passed']==0 and diagnosis['strict_total']==8
    assert diagnosis['extracted_contract_passed']==prior_diagnosis['extracted_contract_passed']==1
    assert [(e['task'],e['status']) for e in diagnosis['results']]==[(e['task'],e['status']) for e in prior_diagnosis['results']]
    return dict(completed_requests=8,generated_ids=total,all_generated_ids_and_text_exact_to_frozen_FP32=True,maximum_live_rows=maximum,
        half_prefill_calls=sum(e['dispatches'] for e in half),postclose_metal_bytes=closed[0]['memory']['metal_allocated_bytes'],strict_passed=0,strict_total=8,diagnostic_single_fence_passed=1)
result=verify(data);negative=[]
mutations={
    'false_quality_pass':lambda d:d['strict'].update(passed=1),
    'sandbox_failure':lambda d:d['strict']['sandbox_probe'].update(status='failed'),
    'wrong_half_route':lambda d:next(e for e in d['native'] if e.get('event')=='native_chunk_half_attention' and e['stage']=='prefill').update(dispatches=0),
    'owner_leak':lambda d:next(e for e in d['native'] if e.get('event')=='closed')['memory'].update(garnet_live_matrix_owners=1),
    'cached_control':lambda d:next(iter(d['clients'].values()))['result'].update(cached_prompt_tokens=128),
    'truncated_output':lambda d:next(iter(d['clients'].values()))['result'].update(finish_reason='length'),
    'changed_token':lambda d:next(e for e in next(iter(d['clients'].values()))['events'] if isinstance(e,dict) and 'response_token_ids' in e)['response_token_ids'].__setitem__(0,-1),
}
for name,mutate in mutations.items():
    bad=copy.deepcopy(data);mutate(bad)
    try:verify(bad)
    except (AssertionError,KeyError,ValueError,TypeError):negative.append(name)
    else:raise AssertionError('Corrupt coding result admitted: '+name)
result.update(passed=True,scope=__doc__,negative_controls_rejected=negative,quality_gate_passed=False,coding_model_qualified=False,no_reference_inference_rerun=True,no_performance_claim=True,goal_complete=False,
    fixture_sha256=sha(root/'fixture1.json'),suite_sha256=b['suite_sha256'],evaluator_sha256=b['evaluator_sha256'],worker_stdout_sha256=sha(root/'server1/worker-stdout.jsonl'),strict_report_sha256=sha(root/'strict-quality1.json'),diagnosis_sha256=sha(root/'block-diagnosis1.json'))
dest=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else root/'collection-audit1.json';assert not dest.exists();dest.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
