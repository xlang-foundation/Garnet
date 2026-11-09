"""Prepare serial C1/C4 serving with the frozen validated native fusion."""
from pathlib import Path
import hashlib,json,shutil,socket,subprocess,sys
PACKAGE=Path(__file__).resolve().parent
ROOT=Path(sys.argv[1]).resolve();M=ROOT.parent
assert M.name=='mistral7b-local-coder' and not ROOT.exists();ROOT.mkdir()
S=M/'mistral-half-serving2';N=M/'mistral-fused-mlp-native1';Q=M/'mistral-fused-mlp-coding-quality1'
read=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
assert sha(S/'provenance1.json')=='d99264195c7b93d21ac4de5d31dd451edeb3fc1192f15b4381b1481802e7d890'
assert sha(N/'provenance1.json')=='b94f6e7ee34d17079439e88cd0d5e419ff9c11e8d2514fbe2dbd00f589917ebe'
assert sha(Q/'provenance1.json')=='6ac91f90b23d98f2bf82dacc3a99747e0f50f85f6a28cdbdb87a5eae1dee806f'
assert read(Q/'fused-quality-audit1.json')['passed'] and not read(Q/'fused-quality-audit1.json')['quality_gate_passed']
frozen=read(S/'provenance1.json')['files_sha256'];copied={}
def copy(name):
    p=S/name;assert sha(p)==frozen[name],name
    target=ROOT/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target);copied[name]=sha(p)
for directory in ['tools1','audit-tools1']:
    for p in sorted((S/directory).iterdir()):
        if p.is_file():copy(str(p.relative_to(S)))
for name in ['fixture1.json','serving-manifest1.json','plan1.json']:copy(name)
for p in sorted((S/'cases1').iterdir()):copy(str(p.relative_to(S)))
for wave in read(S/'plan1.json')['waves']:
    for name in ['engine.json','profile.json']:copy(wave['label']+'/'+name)
worker=ROOT/'tools1/local_coder_batch_worker.py'
assert sha(Q/'tools1/local_coder_batch_worker.py')==read(Q/'provenance1.json')['files_sha256']['tools1/local_coder_batch_worker.py']
shutil.copyfile(Q/'tools1/local_coder_batch_worker.py',worker)
assert all(t not in worker.read_text() for t in ['verification_f32_sha256','native_request_state_witness','native_survivor_restore_witness','prepare_gpu_counters'])
manifest=read(ROOT/'serving-manifest1.json')
for row in manifest['plans']:row['cases']=str(ROOT/'cases1'/(row['id']+'.json'));assert sha(Path(row['cases']))==row['cases_sha256']
save(ROOT/'serving-manifest1.json',manifest)
plan=read(ROOT/'plan1.json');held=[];commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=M.parents[2]/'Garnet',text=True).strip()
for wave in plan['waves']:
    folder=ROOT/wave['label'];profile=read(folder/'profile.json')
    profile.update(fused_mlp_prefill=True,profile_id='mistral-fused-mlp-serving-'+wave['label'],scope='Frozen fused native MLP, serial C1/C4 long-prompt serving; no prefix retention or state/GPU profiling. Strict coding remains0/8; frozen reference engines are never rerun.')
    save(folder/'profile.json',profile)
    sock=socket.socket();sock.bind(('127.0.0.1',0));held.append(sock);port=sock.getsockname()[1]
    spec=read(folder/'engine.json');argv=spec['argv'];argv[2]=str(ROOT/'tools1/local_coder_batch_http.py')
    argv[argv.index('--log-root')+1]=str(folder/'server-template');argv[argv.index('--port')+1]=str(port);at=argv.index('--')
    argv[at+1:]=[str(M.parents[2]/'out/build/macos-arm64-release/bin/xlang3'),'-B',str(worker),str(N/'library2'),str(N/'source2'),plan['model'],str(folder/'profile.json')]
    spec.update(engine='garnet-native-fused-mlp-'+wave['label'],url='http://127.0.0.1:'+str(port),argv=argv,native_library_sha256=sha(N/'library2/libgarnet.dylib'),source_manifest_sha256=sha(N/'source-bindings2.json'),profile_sha256=sha(folder/'profile.json'),source_commit=commit,scope=profile['scope'],frozen_case_file=next(p['cases'] for p in manifest['plans'] if p['id']==wave['case']))
    save(folder/'engine.json',spec)
    wave['command']=[str(M.parents[2]/'out/research/qwen-local-coder/asset-env/bin/python'),'-B',str(ROOT/'tools1/run_mistral_cold_prefix.py'),'--engine-spec',str(folder/'engine.json'),'--fixture',str(ROOT/'fixture1.json'),'--serving-manifest',str(ROOT/'serving-manifest1.json'),'--output',str(folder/'campaign'),'--case-id',wave['case']]
    wave.update(engine_sha256=sha(folder/'engine.json'),profile_sha256=sha(folder/'profile.json'))
for sock in held:sock.close()
original_pre=read(S/'preflight1.json')['files_sha256']
assert sha(Path(plan['native_model_binding']))==original_pre[plan['native_model_binding']]
plan.update(scope=profile['scope'],library=str(N/'library2'),library_sha256=sha(N/'library2/libgarnet.dylib'),source=str(N/'source2'),unprofiled_worker_sha256=sha(worker),native_model_binding_sha256=sha(Path(plan['native_model_binding'])),fused_native_source_binding_sha256=sha(N/'source-bindings2.json'),original_half_serving_provenance_sha256=sha(S/'provenance1.json'))
save(ROOT/'plan1.json',plan)
for name in ['run1.py','audit1.py']:shutil.copyfile(PACKAGE/name,ROOT/name)
shutil.copyfile(Path(__file__),ROOT/'prepare1.py')
save(ROOT/'preparation1.json',dict(passed=True,source_half_serving_provenance_sha256=sha(S/'provenance1.json'),copied_files_sha256=copied,worker_sha256=sha(worker),prepare_sha256=sha(Path(__file__)),no_comparator_execution=True))
print(json.dumps(dict(prepared=True,destination=str(ROOT),waves=4,order=[w['case'] for w in plan['waves']],worker_sha256=sha(worker))))
