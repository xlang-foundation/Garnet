"""Audit the serial native OFF/ON collection without starting inference."""
from pathlib import Path
import hashlib
import json
import re
import sys

z=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
load=lambda name:json.loads((z/name).read_text())
b=load('bindings1.json');status=load('model-state-status1.json');fixture=load('fixture1.json')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert status['status']=='collected_pending_audit'
assert [v['mode'] for v in status['waves']]==['off','on']
assert all(v['status']=='complete' and v['returncode']==0 and v['last_event']=='closed' for v in status['waves'])
assert sha(z/'worker-state1.py')==b['worker_sha256']
assert sha(z/'fixture1.json')==b['fixture_sha256']
assert sha(z/'source-audit1.json')==b['source_audit_sha256']
assert sha(Path(b['original_control']))==b['original_control_sha256']
for name,expected in load('source-audit1.json')['candidate_owned_files_sha256'].items():
    assert sha(Path(b['source'])/name)==expected,name
control=json.loads(Path(b['original_control']).read_text())
events={};completed={};routes={};states={};postclose={}
for mode in ['off','on']:
    assert sha(Path(b['libraries'][mode]['path'])/'libgarnet.dylib')==b['libraries'][mode]['sha256']
    assert sha(z/('profile-'+mode+'1.json'))==b['libraries'][mode]['profile_sha256']
    rows=[json.loads(line) for line in (z/('model-'+mode+'1.stdout')).read_text().splitlines() if line.startswith('{')]
    assert rows and not [r for r in rows if r['event']=='error']
    assert len([r for r in rows if r['event']=='ready'])==1
    ends=[r for r in rows if r['event']=='completed'];assert len(ends)==4
    completed[mode]={r['id']:r for r in ends};assert len(completed[mode])==4
    for request in fixture['requests']:
        key='trained-'+str(request['row']);r=completed[mode][key]
        assert r['prompt_ids']==request['token_ids']==control[key]['prompt_ids']
        assert r['generated_ids']==control[key]['generated_ids'] and len(r['generated_ids'])==8
        assert r['text']==control[key]['text'] and r['cached_tokens']==0 and r['finish_reason']=='length'
    assert rows[-1]['event']=='closed' and len([r for r in rows if r['event']=='closed'])==1
    mem=rows[-1]['memory']
    for key in ['garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets']:
        assert mem[key]==0,key
    assert mem['metal_allocated_bytes']<=4194304;postclose[mode]=mem['metal_allocated_bytes']
    route=[r for r in rows if r['event']=='diagnostic_native_kv_hoist'];assert len(route)==25
    assert [r['invocation'] for r in route]==list(range(1,26))
    for r in route:
        assert len(r['ids'])==len(r['slots'])==len(r['active']) and len(r['epochs'])==4
        assert r['dispatches']==(64 if mode=='on' and len(r['ids'])>4 else 0)
    assert sum(r['mps_dispatches'] for r in route)==512
    witness=[r for r in rows if r['event']=='diagnostic_full_state'];assert [r['invocation'] for r in witness]==[1,16,18,25]
    for r in witness:
        assert len(r['layers'])==32 and all(len(pair)==2 for pair in r['layers'])
        assert all(re.fullmatch('[0-9a-f]{64}',h) for pair in r['layers'] for h in pair)
        assert re.fullmatch('[0-9a-f]{64}',r['hidden'])
    affine=[r for r in rows if r['event']=='native_affine_profile'];assert affine and all(r['misses']==0 for r in affine)
    events[mode]=rows;routes[mode]=route;states[mode]=witness
for a,broute in zip(routes['off'],routes['on']):
    assert {k:v for k,v in a.items() if k!='dispatches'}=={k:v for k,v in broute.items() if k!='dispatches'}
assert states['off']==states['on'], 'Full native KV/hidden digest changed'
prior=Path(b['prior_state_study'])
old=[json.loads(line) for line in (prior/'model-off1.stdout').read_text().splitlines() if line.startswith('{')]
assert states['off']==[r for r in old if r['event']=='diagnostic_full_state'], 'New OFF control differs from prior qualified whole state'
assert sum(r['dispatches'] for r in routes['on'])==1088
assert sum(r['dispatches'] for r in routes['off'])==0
for invocation,start in [(1,0),(16,1920)]:
    r=routes['on'][invocation-1]
    assert r['ids']==[token for request in fixture['requests'] for token in request['token_ids'][start:start+128]]
    assert r['slots']==[float(t) for _ in range(4) for t in range(start,start+128)]
    assert r['active']==[1.0]*512 and r['dispatches']==64
for invocation in [17,25]:
    assert len(routes['on'][invocation-1]['ids'])==4 and routes['on'][invocation-1]['dispatches']==0
assert len(routes['on'][17]['ids'])==512 and sum(routes['on'][17]['active'])==4 and routes['on'][17]['dispatches']==64
for stem,count in [('ctest-on1',50),('ctest-off1',50)]:
    assert load(stem+'-status.json')['returncode']==0
    assert '100% tests passed out of '+str(count) in (z/(stem+'.stdout')).read_text()
result=dict(passed=True,scope='Two fresh native engines strictly serial OFF then ON; compiler-proved KV commit reordering with original attention arithmetic. Four frozen long prompts and8 output tokens each. Complete physical32-layer FP32 KV and4-row hidden-bank at body calls1,16,18,25 match both modes and prior qualified control. Bounded exercised regression, not a full-model numerical/logits oracle, serving speed or coding qualification.',
    prompt_lengths=[len(r['token_ids']) for r in fixture['requests']],completed_requests_per_engine=4,
    generated_ids_per_engine=32,all_outputs_equal_saved_control=True,body_calls_per_engine=25,
    hoisted_cache_commits=dict(off=0,on=1088),matching_complete_state_digest_pairs=260,
    observed_state_digests=520,state_boundaries=[1,16,18,25],boundary_roles={'1':'first full prefill','16':'last full prefill','18':'sparse tail prefill after first decode of earlier ready rows','25':'final decode'},native_owners_closed=True,
    postclose_metal_bytes=postclose,ctest=dict(on=50,off=50),
    warning='Context leak detected, CoreAnalytics returned false occurs with capture disabled. Native owner counts/bounded device snapshots do not explain or dismiss that framework warning.',
    cached_return_requests_tested=False,full_model_performance_measured=False,
    serving_speed_claim=False,quality_gate_passed=False,goal_complete=False)
(z/'model-state-audit1.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
