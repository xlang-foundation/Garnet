"""Audit retained full-model fused MLP regression without running inference."""
from pathlib import Path
import copy,hashlib,json,math,sys
ROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
DEST=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else ROOT/'model1-audit.json'
read=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def events(p):return [json.loads(l) for l in p.read_text().splitlines() if l.startswith('{')]
def select(es,kind):return [e for e in es if e.get('event')==kind]
pre=read(ROOT/'model1-preflight.json');post=read(ROOT/'model1-postflight.json');status=read(ROOT/'model1-status.json')
assert pre['passed'] and pre['no_comparator_execution'] and post['passed'] and post['process_terminal']
assert status['status']=='terminal' and status['returncode']==0 and status['last_event']=='closed'
assert post['checked_files']==len(pre['files_sha256'])
for name,digest in pre['files_sha256'].items():assert sha(Path(name))==digest,name
reference=next(Path(name) for name in pre['files_sha256'] if name.endswith('/mistral-half-attention-native1/model-on7.stdout'))
control=events(reference);candidate=events(ROOT/'model1.stdout')
profile=read(ROOT/'profile-model1.json');inventory=read(ROOT/'engine-inventory1.json')
fixture=read(Path(profile['diagnostic_fixture']))
def verify(es,inv):
    assert es and not select(es,'error') and es[-1]['event']=='closed'
    ready=select(es,'ready');assert len(ready)==1 and ready[0]['profile']==profile
    assert 'fused_mistral_mlp_prefill_fast_v1' in ready[0]['backend'] and 'fp16_prefill_all_p16_gpupack_v1' in ready[0]['backend']
    prep=select(es,'native_fused_mlp_prepared')
    assert prep==[dict(event='native_fused_mlp_prepared',sites=[1]*32,scratch_bytes=14680064,producer='fused-mistral-mlp-fast-v1')]
    done=select(es,'completed');assert len(done)==4
    outputs={e['id']:e for e in done};expected={e['id']:e for e in select(control,'completed')}
    assert set(outputs)==set(expected)
    for req in fixture['requests']:
        key='trained-'+str(req['row']);actual=outputs[key];saved=expected[key]
        for field in ['generated_ids','prompt_ids','text','cached_tokens','finish_reason']:
            assert actual[field]==saved[field],(key,field)
        assert len(actual['generated_ids'])==8 and actual['prompt_ids']==req['token_ids'] and actual['cached_tokens']==0
    # Native scheduling, physical slots, epochs, active masks and arithmetic routes
    # must remain identical to the selected half-policy control.
    for kind in ['native_batch','diagnostic_native_kv_hoist','diagnostic_committed_attention','diagnostic_half_attention',
                 'native_affine_profile','native_cooperative_affine','native_indexed','native_indexed_decode',
                 'native_chunk_kv_scatter','native_chunk_layout_vector','native_prefill_pool']:
        assert select(es,kind)==select(control,kind),kind
    routes=select(es,'diagnostic_native_kv_hoist');fused=select(es,'diagnostic_fused_mlp')
    assert len(routes)==len(fused)==25 and [e['invocation'] for e in fused]==list(range(1,26))
    for route,event in zip(routes,fused):
        active=sum(x!=0 for x in route['active']);blocks=32 if len(route['ids'])>4 and active>=128 else 0
        assert event==dict(event='diagnostic_fused_mlp',invocation=route['invocation'],blocks=blocks,active_tokens=active,
                          logical_affine_products=blocks*3,producer_kernels=blocks,down_kernels=blocks)
    assert sum(e['blocks'] for e in fused)==512
    assert sum(e['dispatches'] for e in select(es,'diagnostic_half_attention'))==544
    witness=select(es,'diagnostic_full_state');saved=select(control,'diagnostic_full_state')
    assert [e['invocation'] for e in witness]==[1,16,18,25]
    assert witness==saved,'Complete physical KV, hidden digest or hidden values changed'
    for e in witness:
        assert len(e['layers'])==32 and all(len(pair)==2 for pair in e['layers'])
        flat=lambda a:[x for value in a for x in flat(value)] if isinstance(a,list) else [a]
        words=flat(e['hidden_values']);assert len(words)==16384 and all(math.isfinite(x) for x in words)
    refs=[r for r in inv['references'] if r['role']=='fused_mlp_half_input']
    assert len(refs)==1 and refs[0]['tag']=='session' and not refs[0]['input'] and not refs[0]['output_part']
    buffers={b['id']:b for b in inv['buffers']};arena=buffers[refs[0]['buffer_id']]
    assert arena['length_bytes']==arena['allocated_bytes']==14680064
    assert len([r for r in inv['references'] if r['buffer_id']==arena['id']])==1,'Private fused arena aliases another inventory role'
    closed=select(es,'closed');assert len(closed)==1
    memory=closed[0]['memory']
    for key in ['garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_matrix_owners','garnet_live_pipeline_sets']:
        assert memory[key]==0,key
    assert memory['metal_allocated_bytes']<=4194304
    return dict(completed_requests=4,generated_tokens=32,full_state_witnesses=260,hidden_values_compared=65536,
                fused_blocks=512,fused_producer_dispatches=512,fused_down_dispatches=512,half_attention_dispatches=544,
                private_half_arena_bytes=arena['length_bytes'],postclose_metal_bytes=memory['metal_allocated_bytes'])
result=verify(candidate,inventory)
mutations={
    'wrong_generated_token':lambda e,i:select(e,'completed')[0]['generated_ids'].__setitem__(0,0),
    'wrong_text':lambda e,i:select(e,'completed')[0].update(text='changed'),
    'wrong_prompt':lambda e,i:select(e,'completed')[0]['prompt_ids'].__setitem__(0,0),
    'missing_compiled_site':lambda e,i:select(e,'native_fused_mlp_prepared')[0]['sites'].__setitem__(0,0),
    'missing_fused_dispatch':lambda e,i:select(e,'diagnostic_fused_mlp')[0].update(blocks=0),
    'decode_fused_dispatch':lambda e,i:select(e,'diagnostic_fused_mlp')[-1].update(blocks=32),
    'wrong_native_epoch':lambda e,i:select(e,'diagnostic_native_kv_hoist')[0]['epochs'].__setitem__(0,0),
    'wrong_kv_state':lambda e,i:select(e,'diagnostic_full_state')[0]['layers'][0].__setitem__(0,'0'*64),
    'wrong_hidden_state':lambda e,i:select(e,'diagnostic_full_state')[-1].update(hidden='0'*64),
    'nonfinite_hidden':lambda e,i:select(e,'diagnostic_full_state')[-1].update(hidden_values=[float('nan')]*16384),
    'owner_leak':lambda e,i:select(e,'closed')[0]['memory'].update(garnet_live_matrix_owners=1),
    'device_leak':lambda e,i:select(e,'closed')[0]['memory'].update(metal_allocated_bytes=4194305),
    'missing_private_arena':lambda e,i:i.update(references=[r for r in i['references'] if r['role']!='fused_mlp_half_input']),
    'wrong_private_arena_size':lambda e,i:next(b for b in i['buffers'] if b['id']==next(r['buffer_id'] for r in i['references'] if r['role']=='fused_mlp_half_input')).update(length_bytes=0),
}
rejected=[]
for name,mutate in mutations.items():
    es=copy.deepcopy(candidate);inv=copy.deepcopy(inventory);mutate(es,inv)
    try:verify(es,inv)
    except (AssertionError,KeyError,ValueError,TypeError):rejected.append(name)
    else:raise AssertionError('Corrupted evidence admitted: '+name)
result.update(passed=True,scope='Untimed native fused MLP full-model exact regression against frozen selected half policy.',
              all_tokens_text_and_full_state_exact=True,negative_controls_rejected=rejected,
              stdout_sha256=sha(ROOT/'model1.stdout'),reference_stdout_sha256=sha(reference),
              inventory_sha256=sha(ROOT/'engine-inventory1.json'),preflight_sha256=sha(ROOT/'model1-preflight.json'),
              coding_quality_qualified=False,prefix_reuse_qualified=False,serving_speed_measured=False,goal_complete=False,
              warning='CoreAnalytics context-leak messages remain retained; zero native owners do not explain those warnings.')
assert not DEST.exists();DEST.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
