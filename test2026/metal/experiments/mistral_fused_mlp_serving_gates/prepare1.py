"""Prepare unchanged pressure and coding protocols for the frozen fused engine."""
from pathlib import Path
import hashlib,json,shutil,sys
ROOT=Path(__file__).resolve().parent;M=Path(sys.argv[2]).resolve().parent if len(sys.argv)>2 else ROOT.parent
assert M.name=='mistral7b-local-coder'
N=M/'mistral-fused-mlp-native1';P=M/'mistral-fused-mlp-prefix-http1'
OLD_PRESSURE=M/'mistral-half-prefix-pressure1';OLD_QUALITY=M/'mistral-half-coding-quality1'
read=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
assert sha(N/'provenance1.json')=='b94f6e7ee34d17079439e88cd0d5e419ff9c11e8d2514fbe2dbd00f589917ebe'
assert sha(P/'provenance1.json')=='d9abf05f94f2c8f92c066717eee3ca5824ed39766c7412e8923d0bf2ab4ed96c'
assert read(P/'fused-http-audit1.json')['passed']
kind=sys.argv[1];assert kind in ['pressure','quality']
dest=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else M/('mistral-fused-mlp-prefix-pressure1' if kind=='pressure' else 'mistral-fused-mlp-coding-quality1')
assert not dest.exists();dest.mkdir();old=OLD_PRESSURE if kind=='pressure' else OLD_QUALITY
pins={'pressure':'14e338f9565f86cf3f8b4e566fe6b85afa08db782723942f6b8d6fb1f2830e9f','quality':'78b2a8d7b1d95c093cfa042f2377aee37a5c77f6ee3cb0ab7241ed1da574de4e'}
assert sha(old/'provenance1.json')==pins[kind]
frozen=read(old/'provenance1.json')['files_sha256'];copied={}
def copy(name):
    path=old/name;assert sha(path)==frozen[name],name
    target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target);copied[name]=sha(path)
for path in sorted((old/'tools1').iterdir()):
    if path.is_file():copy(str(path.relative_to(old)))
names=['fixture2.json','run_http1.py','profile-template1.json','audit_pressure_protocol1.py'] if kind=='pressure' else ['fixture1.json','profile1.json','run_quality1.py']
for name in names:copy(name)
worker=dest/'tools1/local_coder_batch_worker.py'
if kind=='pressure':
    assert sha(P/'tools1/local_coder_batch_worker.py')==read(P/'provenance1.json')['files_sha256']['tools1/local_coder_batch_worker.py']
    shutil.copyfile(P/'tools1/local_coder_batch_worker.py',worker)
else:
    text=worker.read_text()
    def edit(a,b):
        global text
        assert text.count(a)==1,(a,text.count(a));text=text.replace(a,b)
    edit('    backend=garnet.backend_info()', '''    backend=garnet.backend_info()
    fused_mlp=profile.get('fused_mlp_prefill',False)
    if type(fused_mlp) is not bool or fused_mlp!=('fused_mistral_mlp_prefill_fast_v1' in backend):raise ValueError('Fused MLP compile/profile policy differs')''')
    edit('        graphs=decoder._resident_prefill_graphs(chunk)', '''        graphs=decoder._resident_prefill_graphs(chunk)
        if fused_mlp:
            sites=[graph.fused_mlp_sites for graph in graphs]
            if sites!=[1]*32:raise ValueError('Missing fused MLP sites: '+str(sites))
            emit(dict(event='native_fused_mlp_prepared',sites=sites,scratch_bytes=14680064,producer='fused-mistral-mlp-fast-v1'))''')
    edit('            return result\n        def predict_rows(epochs):', '''            blocks=decoder.session.fused_mlp_blocks if fused_mlp else 0
            active_count=sum(1 for flag in active.tolist() if flag!=0)
            expected_blocks=32 if fused_mlp and len(ids)>batch and active_count>=128 else 0
            if blocks!=expected_blocks:raise ValueError('Fused MLP execution coverage differs')
            emit(dict(event='native_fused_mlp',blocks=blocks,active_tokens=active_count,physical_rows=len(ids),logical_affine_products=blocks*3,producer_kernels=blocks,down_kernels=blocks))
            return result
        def predict_rows(epochs):''')
    compile(text,str(worker),'exec');worker.write_text(text)
profile_name='profile-template1.json' if kind=='pressure' else 'profile1.json'
profile=read(dest/profile_name);profile.update(fused_mlp_prefill=True,profile_id='mistral-fused-mlp-'+kind+'1',scope='Unchanged '+kind+' correctness/quality regression for the frozen native fused MLP producer. No speed claim.')
if kind=='pressure':
    profile['prefix_identity']=read(P/'profile-template1.json')['prefix_identity']
    runner=dest/'run_http1.py';text=runner.read_text();anchor="active.update(prefix_cache=cache,native_cached_chunks=cache,profile_id=base['profile_id']+'-'+label)"
    assert text.count(anchor)==1;text=text.replace(anchor,anchor+";active['memory_inventory_path']=str(folder/'engine-inventory1.json')");runner.write_text(text)
else:profile['memory_inventory_path']=str(dest/'engine-inventory1.json')
save(dest/profile_name,profile)
bindings=read(P/'bindings1.json')
if kind=='quality':bindings.update(read(old/'bindings1.json'))
bindings.update(source=str(N/'source2'),library=str(N/'library2'),library_sha256=sha(N/'library2/libgarnet.dylib'),worker_observer_sha256=sha(worker),worker_sha256=sha(worker),fused_native_root=str(N),fused_model_audit_sha256=sha(N/'model1-audit.json'),fused_source_binding_sha256=sha(N/'source-bindings2.json'),scope=profile['scope'],profile_sha256=sha(dest/profile_name),fixture_sha256=sha(dest/('fixture2.json' if kind=='pressure' else 'fixture1.json')))
if kind=='pressure':
    baseline=M/'mistral-half-prefix-http1'
    bindings.update(baseline_half_http_audit_sha256=sha(baseline/'half-http-audit1.json'),baseline_uncached_worker_stdout_sha256=sha(baseline/'uncached/server/worker-stdout.jsonl'),predecessor_audit=str(P/'fused-http-audit1.json'),predecessor_audit_sha256=sha(P/'fused-http-audit1.json'))
else:
    pressure=M/'mistral-fused-mlp-prefix-pressure1'
    bindings.update(predecessor_audit=str(pressure/'fused-pressure-audit1.json'))
save(dest/'bindings1.json',bindings)
for name in ['launch1.py','audit_pressure1.py','audit_quality1.py']:
    shutil.copyfile(ROOT/name,dest/name)
package=M.parents[2]/'Garnet/test2026/metal/experiments/mistral_half_prefill_serving_gates'
for name in ['audit_pressure1.py','audit_quality1.py']:shutil.copyfile(package/name,dest/('inherited-'+name))
save(dest/'preparation1.json',dict(passed=True,kind=kind,original_protocol_provenance_sha256=sha(old/'provenance1.json'),original_files_sha256=copied,worker_sha256=sha(worker),prepare_sha256=sha(Path(__file__)),fused_native_provenance_sha256=sha(N/'provenance1.json')))
print(json.dumps(dict(prepared=True,kind=kind,destination=str(dest),worker_sha256=sha(worker))))
