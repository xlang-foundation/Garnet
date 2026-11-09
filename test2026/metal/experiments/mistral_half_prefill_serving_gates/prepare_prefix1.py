"""Prepare the unchanged return/edit protocol for the qualified half engine."""
import sys
from pathlib import Path
import hashlib,json,shutil
root=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
base=root.parent/'mistral-kv-commit-hoist1'
old=base/'http1';native=root.parent/'mistral-half-attention-native1'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
load=lambda p:json.loads(p.read_text())
assert sha(native/'provenance7.json')=='f47d1073955b9943d1038bc066d9fc0d04aff79599a6c951051fcaae74fefe08'
assert load(native/'model-state-audit7.json')['passed']
frozen=load(base/'provenance1.json')['files_sha256']
tools=root/'tools1';assert not tools.exists();tools.mkdir()
originals={}
for p in sorted((old/'tools1').iterdir()):
    assert p.is_file();name=str(p.relative_to(base));assert sha(p)==frozen[name],name
    originals[name]=sha(p);shutil.copy2(p,tools/p.name)
for name in ['fixture2.json','profile-template1.json','run_http1.py','audit_http2.py']:
    assert sha(old/name)==frozen['http1/'+name]
    shutil.copy2(old/name,root/name)
assert sha(root/'fixture2.json')==sha(root.parent/'mistral-prefill-prefix-http1/fixture2.json')
worker=tools/'local_coder_batch_worker.py';s=worker.read_text()
def edit(old_text,new_text):
    global s
    assert s.count(old_text)==1,(old_text[:80],s.count(old_text))
    s=s.replace(old_text,new_text)
edit('    backend=garnet.backend_info()', '''    backend=garnet.backend_info()
    half_policy=profile.get('half_attention_policy',False)
    if type(half_policy) is not bool or half_policy!=('fp16_prefill_all_p16_gpupack_v1' in backend):raise ValueError('Half-prefill numerical profile differs from compiled engine')
    if 'committed_kv_mps_fp32_v1' in backend:raise ValueError('Half HTTP study excludes the distinct committed-MPS policy')''')
edit("                if attention_buckets:expected=profile['expected_chunk_attention_dispatches'] if len(ids)>batch and attention_valid<=8 else 0", "                if attention_buckets:expected=profile['expected_chunk_attention_dispatches'] if len(ids)>batch and attention_valid<=8 and not half_policy else 0")
edit('                if attention_buckets and attention_valid<=8:expected=0', '''                if attention_buckets and attention_valid<=8:expected=0
                if half_policy:
                    half_count=decoder.session.chunk_half_attention_dispatches
                    half_expected=profile['expected_chunk_attention_dispatches'] if len(ids)>batch else 0
                    emit(dict(event='native_chunk_half_attention',stage='prefill' if len(ids)>batch else 'decode',dispatches=half_count,active_tokens=attention_valid))
                    if half_count!=half_expected:raise ValueError('Half-prefill route coverage differs')
                    expected=0''')
edit('                expected_tail=expected if attention_valid<=8 else 0;expected_extent=0', '                expected_tail=expected if attention_valid<=8 and not half_policy else 0;expected_extent=0')
edit('        if packed:\n            decoder.session.prefill_set_packed_affine(True)', '''        half_scratch=decoder.session.chunk_half_attention_scratch_bytes
        if half_scratch!=(71303552 if half_policy else 0):raise ValueError('Fixed half scratch differs from the compiled profile')
        emit(dict(event='native_chunk_half_attention_prepared',enabled=half_policy,scratch_bytes=half_scratch))
        if packed:
            decoder.session.prefill_set_packed_affine(True)''')
edit("            emit(dict(event='diagnostic_native_kv_hoist',dispatches=hoisted,ids=ids,slots=slots.tolist(),active=active.tolist(),epochs=epochs))", """            emit(dict(event='diagnostic_native_kv_hoist',dispatches=hoisted,ids=ids,slots=slots.tolist(),active=active.tolist(),epochs=epochs))
            half_count=decoder.session.chunk_half_attention_dispatches
            if half_count!=(32 if len(ids)>batch and half_policy else 0):raise ValueError('Diagnostic half-prefill route differs')
            emit(dict(event='diagnostic_half_attention',dispatches=half_count,scratch_bytes=decoder.session.chunk_half_attention_scratch_bytes))""")
edit('        def deliver(request,token,stage):', '''        def capture_target_state(request,phase):
            row=request['row'];epoch=request['epoch'];kv=[]
            for layer in range(32):
                pair=decoder.session.snapshot(layer)
                kv.append([decoder.io.verification_f32_sha256(decoder.io.rows(value,row,1)) for value in pair])
            hidden=decoder.session.diagnostic_final_hidden()
            state=dict(kv=kv,hidden=decoder.io.verification_f32_sha256(decoder.io.rows(hidden,row,1)))
            emit(dict(event='native_request_state_witness',id=jobs[request['id']]['id'],native_request=request['id'],row=row,epoch=epoch,phase=phase,state=state))
        def deliver(request,token,stage):''')
edit("            finished=token in stops or len(job['generated'])>=job['limit']", """            finished=token in stops or len(job['generated'])>=job['limit']
            if stage=='prefill':capture_target_state(request,'prefill')
            if finished:capture_target_state(request,'final')""")
compile(s,str(worker),'exec');worker.chmod(0o644);worker.write_text(s)
profile=load(root/'profile-template1.json')
profile.update(profile_id='mistral-half-prefill-prefix-http1',half_attention_policy=True,scope='Untimed half-prefill native HTTP return/edit/queued-fifth regression. Full target and survivor row witnesses perturb timing; no benchmark or coding-quality admission.')
profile['prefix_identity'][1]='mistral-half-prefill-http-correctness1';profile['prefix_identity'][4]='native-half-prefill-http1'
(root/'profile-template1.json').chmod(0o644);(root/'profile-template1.json').write_text(json.dumps(profile,indent=2)+'\n')
bindings=load(native/'model-bindings7.json');lib=bindings['libraries']['on']
b=dict(source=bindings['source'],library=lib['path'],library_sha256=lib['sha256'],worker_observer_sha256=sha(worker),
    scope=profile['scope'],foundation_provenance_sha256=sha(native/'provenance7.json'),fixture_sha256=sha(root/'fixture2.json'),
    original_tool_files_sha256=originals,model=bindings['model'],model_files_sha256=bindings['model_files_sha256'],xlang=bindings['xlang'],xlang_sha256=bindings['xlang_sha256'],
    original_KV_hoist_provenance_sha256=sha(base/'provenance1.json'),native_candidate_binding_sha256=sha(native/'candidate-bindings7.json'))
(root/'bindings1.json').write_text(json.dumps(b,indent=2)+'\n')
print(json.dumps(dict(prepared=True,worker_sha256=b['worker_observer_sha256'],unchanged_fixture=True,half_policy=True)))
