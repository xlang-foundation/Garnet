"""Saved-data-only audit of serial FP32/half full-model regression witnesses."""
import sys
from pathlib import Path
import copy,hashlib,json,math,re
root=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
load=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def events(p):return [json.loads(l) for l in p.read_text().splitlines() if l.startswith('{')]
def flat(a):return [x for item in a for x in flat(item)] if isinstance(a,list) else [a]
b=load(root/'model-bindings7.json');status=load(root/'model-state-status7.json')
assert status['status']=='collected_pending_audit'
assert [w['mode'] for w in status['waves']]==['off','on']
assert all(w['status']=='terminal' and w['returncode']==0 and w['last_event']=='closed' for w in status['waves'])
for p,h in [(b['worker'],b['worker_sha256']),(b['fixture'],b['fixture_sha256']),
    (b['original_control'],b['original_control_sha256']),
    (root/'source-audit7.json',b['source_audit_sha256']),(root/'trained-audit7.json',b['trained_audit_sha256']),
    (Path(b['prior_state_study'])/'model-on1.stdout',b['prior_state_stdout_sha256'])]:assert sha(p)==h,str(p)
assert load(root/'source-audit7.json')['passed'] and load(root/'trained-audit7.json')['passed']
for name,h in load(root/'candidate-bindings7.json')['files_sha256'].items():assert sha(Path(b['source'])/name)==h,name
for mode in ['off','on']:
    e=b['libraries'][mode]
    assert sha(Path(e['path'])/'libgarnet.dylib')==e['sha256'] and sha(e['profile'])==e['profile_sha256']
prior=events(Path(b['prior_state_study'])/'model-on1.stdout')
prior_states=[e for e in prior if e['event']=='diagnostic_full_state']
prior_routes=[e for e in prior if e['event']=='diagnostic_native_kv_hoist']
control=load(Path(b['original_control']));fixture=load(Path(b['fixture']))
data={mode:events(root/('model-'+mode+'7.stdout')) for mode in ['off','on']}
def select(es,kind):return [e for e in es if e['event']==kind]
def verify(d):
    states={};post={};products={};half={}
    for mode in ['off','on']:
        es=d[mode];enabled=mode=='on';assert es and not select(es,'error')
        ready=select(es,'ready');assert len(ready)==1
        assert ready[0]['profile']==load(Path(b['libraries'][mode]['profile']))
        assert ('fp16_prefill_all_p16_gpupack_v1' in ready[0]['backend'])==enabled
        prep=select(es,'native_chunk_half_attention_prepared');assert prep==[dict(event='native_chunk_half_attention_prepared',enabled=enabled,scratch_bytes=71303552 if enabled else 0)]
        done=select(es,'completed');assert len(done)==4
        ends={e['id']:e for e in done};assert len(ends)==4
        for request in fixture['requests']:
            key='trained-'+str(request['row']);e=ends[key]
            assert e['prompt_ids']==request['token_ids']==control[key]['prompt_ids']
            assert e['generated_ids']==control[key]['generated_ids'] and len(e['generated_ids'])==8,'Generated IDs differ from frozen control'
            assert e['text']==control[key]['text'] and e['cached_tokens']==0 and e['finish_reason']=='length'
        assert es[-1]['event']=='closed' and len(select(es,'closed'))==1
        memory=es[-1]['memory']
        for name in ['garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets']:assert memory[name]==0,name
        assert memory['metal_allocated_bytes']<=4194304
        post[mode]=memory['metal_allocated_bytes']
        routes=select(es,'diagnostic_native_kv_hoist');assert len(routes)==25
        strip=lambda rows:[{k:v for k,v in e.items() if k!='mps_dispatches'} for e in rows]
        assert strip(routes)==strip(prior_routes),'Native input, slot, epoch or commit sequence changed'
        committed=select(es,'diagnostic_committed_attention');halves=select(es,'diagnostic_half_attention')
        assert len(committed)==len(halves)==25
        assert [e['invocation'] for e in halves]==list(range(1,26))
        assert sum(e['dispatches'] for e in routes)==1088
        for route,c,h in zip(routes,committed,halves):
            assert route['invocation']==c['invocation']==h['invocation']
            prefill=len(route['ids'])>4
            active=sum(x!=0 for x in route['active'])
            expected_mps=32 if prefill and active>8 and not enabled else 0
            expected_half=32 if prefill and enabled else 0
            assert route['mps_dispatches']==expected_mps
            assert h['dispatches']==expected_half and h['scratch_bytes']==(71303552 if enabled else 0)
            assert c['dispatches']==0
            width=len(route['ids'])//4;owners=sum(any(route['active'][i*width:(i+1)*width]) for i in range(4))
            assert c['matrices']==expected_mps*owners*8*4
            assert c['skipped_matrices']==expected_mps*(4-owners)*8*4
        half[mode]=sum(e['dispatches'] for e in halves);products[mode]=sum(e['matrices'] for e in committed)
        witness=select(es,'diagnostic_full_state');assert [e['invocation'] for e in witness]==[1,16,18,25]
        for e in witness:
            assert len(e['layers'])==32 and all(len(pair)==2 for pair in e['layers'])
            assert all(re.fullmatch('[0-9a-f]{64}',h) for pair in e['layers'] for h in pair) and re.fullmatch('[0-9a-f]{64}',e['hidden'])
            hidden=flat(e['hidden_values']);assert len(hidden)==16384 and all(type(x) in [int,float] and math.isfinite(x) for x in hidden)
        states[mode]=witness
        affine=select(es,'native_affine_profile');assert affine and all(e['misses']==0 for e in affine)
    assert [{k:v for k,v in e.items() if k!='hidden_values'} for e in states['off']]==prior_states,'FP32 control whole state changed'
    numeric={};bits={}
    for a,c in zip(states['off'],states['on']):
        x,y=flat(a['hidden_values']),flat(c['hidden_values']);delta=[v-u for u,v in zip(x,y)]
        numeric[str(a['invocation'])]=dict(words=len(x),max_absolute_error=max(abs(v) for v in delta),rms_error=math.sqrt(sum(v*v for v in delta)/len(delta)),max_control_magnitude=max(abs(v) for v in x))
        ah=[h for pair in a['layers'] for h in pair]+[a['hidden']];ch=[h for pair in c['layers'] for h in pair]+[c['hidden']]
        bits[str(a['invocation'])]=dict(equal_digest_pairs=sum(u==v for u,v in zip(ah,ch)),total_pairs=65)
    return dict(postclose_metal_bytes=post,half_prefill_dispatches=half,actual_mps_products=products,hidden_errors_descriptive_only=numeric,whole_state_bit_identity_descriptive_only=bits)
result=verify(data)
mutations={
    'missing_half_route':lambda d:d['on'].__delitem__(next(i for i,e in enumerate(d['on']) if e['event']=='diagnostic_half_attention')),
    'wrong_half_route':lambda d:select(d['on'],'diagnostic_half_attention')[0].update(dispatches=0),
    'wrong_decode_half':lambda d:select(d['on'],'diagnostic_half_attention')[-1].update(dispatches=32),
    'wrong_mps_products':lambda d:select(d['on'],'diagnostic_committed_attention')[0].update(matrices=1),
    'wrong_ids':lambda d:select(d['on'],'completed')[0].update(generated_ids=[0]*8),
    'wrong_epoch':lambda d:select(d['on'],'diagnostic_native_kv_hoist')[0].update(epochs=[0]*4),
    'hidden_nan':lambda d:select(d['on'],'diagnostic_full_state')[0].update(hidden_values=[float('nan')]*16384),
    'owner_leak':lambda d:d['on'][-1]['memory'].update(garnet_live_matrix_owners=1),
    'device_leak':lambda d:d['on'][-1]['memory'].update(metal_allocated_bytes=4194305),
    'off_state_drift':lambda d:select(d['off'],'diagnostic_full_state')[0].update(hidden='0'*64),
    'extra_error':lambda d:d['on'].insert(-1,dict(event='error',error='injected')),
}
rejected=[]
for name,mutate in mutations.items():
    d=copy.deepcopy(data);mutate(d)
    try:verify(d)
    except (AssertionError,KeyError,ValueError,TypeError):rejected.append(name)
    else:raise AssertionError('Corrupt saved model data admitted: '+name)
result.update(passed=True,scope=b['scope'],completed_requests_per_engine=4,generated_tokens_per_engine=32,all_ids_and_text_match_frozen_control=True,
    full_state_witnesses_per_engine=260,original_off_state_exact_to_prior=True,negative_controls_rejected=rejected,
    model_binding_sha256=sha(root/'model-bindings7.json'),stdout_sha256={m:sha(root/('model-'+m+'7.stdout')) for m in ['off','on']},
    full_model_numerical_oracle=False,coding_quality_qualified=False,prefix_reuse_qualified=False,serving_speed_measured=False,goal_complete=False,
    warning='CoreAnalytics context-leak warnings retained in both logs; bounded native owners do not explain them.')
dest=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else root/'model-state-audit7.json';assert not dest.exists();dest.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
