"""Prepare the frozen return/edit/queued-fifth protocol for the fused engine."""
from pathlib import Path
import hashlib,json,shutil,sys
N=Path(__file__).resolve().parent
P=N.parent/'mistral-half-prefix-http1'
ROOT=Path(sys.argv[1]).resolve()
assert ROOT.parent==N.parent and not ROOT.exists();ROOT.mkdir()
read=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
assert sha(P/'provenance1.json')=='895758ad0fef87a35b4d0749fb5179b824e964a7153f88fd85f12194fe42f7c1'
assert read(N/'model1-audit.json')['passed']
frozen=read(P/'provenance1.json')['files_sha256'];originals={}
def copy(name):
    p=P/name;assert sha(p)==frozen[name],name
    q=ROOT/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q);originals[name]=sha(p)
for p in sorted((P/'tools1').iterdir()):
    assert p.is_file();copy(str(p.relative_to(P)))
for name in ['fixture2.json','run_http1.py','audit_http2.py','audit_half1.py','profile-template1.json']:copy(name)
worker=ROOT/'tools1/local_coder_batch_worker.py';text=worker.read_text()
def edit(a,b):
    global text
    assert text.count(a)==1,(a[:80],text.count(a));text=text.replace(a,b)
edit('    backend=garnet.backend_info()', '''    backend=garnet.backend_info()
    fused_mlp=profile.get('fused_mlp_prefill',False)
    if type(fused_mlp) is not bool or fused_mlp!=('fused_mistral_mlp_prefill_fast_v1' in backend):raise ValueError('Fused MLP compile/profile policy differs')''')
edit('        graphs=decoder._resident_prefill_graphs(chunk)', '''        graphs=decoder._resident_prefill_graphs(chunk)
        if fused_mlp:
            sites=[graph.fused_mlp_sites for graph in graphs]
            if sites!=[1]*32:raise ValueError('Fused compiled MLP sites missing: '+str(sites))
            emit(dict(event='native_fused_mlp_prepared',sites=sites,scratch_bytes=14680064,producer='fused-mistral-mlp-fast-v1'))''')
anchor="            emit(dict(event='diagnostic_native_kv_hoist',dispatches=hoisted,ids=ids,slots=slots.tolist(),active=active.tolist(),epochs=epochs))"
edit(anchor,anchor+'''
            blocks=decoder.session.fused_mlp_blocks if fused_mlp else 0
            active_count=sum(1 for flag in active.tolist() if flag!=0)
            expected_blocks=32 if fused_mlp and len(ids)>batch and active_count>=128 else 0
            if blocks!=expected_blocks:raise ValueError('Unexpected fused MLP block coverage: '+str(blocks)+' vs '+str(expected_blocks))
            emit(dict(event='diagnostic_fused_mlp',blocks=blocks,active_tokens=active_count,logical_affine_products=blocks*3,producer_kernels=blocks,down_kernels=blocks))''')
compile(text,str(worker),'exec');worker.write_text(text)
runner=ROOT/'run_http1.py';text=runner.read_text()
anchor="active.update(prefix_cache=cache,native_cached_chunks=cache,profile_id=base['profile_id']+'-'+label)"
assert text.count(anchor)==1;text=text.replace(anchor,anchor+";active['memory_inventory_path']=str(folder/'engine-inventory1.json')")
compile(text,str(runner),'exec');runner.write_text(text)
profile=read(ROOT/'profile-template1.json')
profile.update(profile_id='mistral-fused-mlp-prefix-http1',fused_mlp_prefill=True,scope='Untimed fused native MLP return/edit/queued-fifth regression with target and survivor state observers. No speed or coding-quality claim.')
profile['prefix_identity'][1]='mistral-fused-mlp-http-correctness1';profile['prefix_identity'][4]='native-fused-mlp-http1'
save(ROOT/'profile-template1.json',profile)
original=read(N.parent/'mistral-half-attention-native1/model-bindings7.json')
bindings=read(P/'bindings1.json')
bindings.update(source=str(N/'source2'),library=str(N/'library2'),library_sha256=sha(N/'library2/libgarnet.dylib'),
                worker_observer_sha256=sha(worker),scope=profile['scope'],fused_native_root=str(N),fused_model_audit_sha256=sha(N/'model1-audit.json'),
                fused_source_binding_sha256=sha(N/'source-bindings2.json'),frozen_half_http_provenance_sha256=sha(P/'provenance1.json'),original_half_files_sha256=originals)
save(ROOT/'bindings1.json',bindings)
shutil.copyfile(N/'launch_prefix1.py',ROOT/'launch1.py')
shutil.copyfile(N/'audit_prefix1.py',ROOT/'audit_fused1.py')
save(ROOT/'preparation1.json',dict(passed=True,original_half_files_sha256=originals,unchanged_fixture=sha(ROOT/'fixture2.json')==sha(P/'fixture2.json'),
                                 prepare_sha256=sha(Path(__file__)),fused_model_audit_sha256=sha(N/'model1-audit.json'),source_binding_sha256=sha(N/'source-bindings2.json')))
print(json.dumps(dict(prepared=True,destination=str(ROOT),worker_sha256=sha(worker))))
