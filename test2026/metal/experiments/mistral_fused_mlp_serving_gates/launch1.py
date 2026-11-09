"""Serial pressure or coding regression with the immutable fused library."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,v):
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
assert not (ROOT/'launch1-status.json').exists() and not (ROOT/'status1.json').exists()
b=read(ROOT/'bindings1.json');N=Path(b['fused_native_root'])
sys.path.insert(0,str(ROOT/'tools1'))
from run_local_coder_baseline import engine_processes
assert not engine_processes(),engine_processes()
assert read(N/'suite1-status.json')['status']=='terminal' and read(N/'suite1-status.json')['returncode']==0
assert read(N/'use-proof1-status.json')['returncode']==0
checks={}
def bind(p,expected=None):
    h=sha(p)
    if expected is not None:assert h==expected,str(p)
    checks[str(p)]=h
bind(N/'model1-audit.json',b['fused_model_audit_sha256'])
bind(N/'source-bindings2.json',b['fused_source_binding_sha256'])
for name,h in read(N/'source-bindings2.json')['files_sha256'].items():bind(Path(b['source'])/name,h)
for name,h in read(N.parent/'mistral-half-attention-native1/source-audit7.json')['original_dirty_sha256'].items():bind(ROOT.parents[3]/'Garnet'/name,h)
for name,h in b['model_files_sha256'].items():bind(Path(b['model'])/name,h)
bind(Path(b['library'])/'libgarnet.dylib',b['library_sha256']);bind(Path(b['xlang']),b['xlang_sha256'])
bind(ROOT/'tools1/local_coder_batch_worker.py',b['worker_observer_sha256']);bind(ROOT/('fixture2.json' if (ROOT/'fixture2.json').exists() else 'fixture1.json'),b['fixture_sha256'])
predecessor=Path(b['predecessor_audit']);assert read(predecessor)['passed'];bind(predecessor,b.get('predecessor_audit_sha256'))
bind(N/'provenance1.json',read(ROOT/'preparation1.json')['fused_native_provenance_sha256'])
for p in sorted(ROOT.iterdir())+[N/'suite1-status.json',N/'use-proof1.stdout']+sorted((ROOT/'tools1').iterdir()):
    if p.is_file():bind(p)
save(ROOT/'preflight-hashes1.json',checks)
command=[str(ROOT.parents[3]/'out/research/qwen-local-coder/asset-env/bin/python'),'-B',str(ROOT/('run_quality1.py' if (ROOT/'run_quality1.py').exists() else 'run_http1.py'))]
save(ROOT/'launch1-command.json',command)
with (ROOT/'collection1.stdout').open('x') as out,(ROOT/'collection1.stderr').open('x') as err:
    p=subprocess.Popen(command,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MTL_CAPTURE_ENABLED='0',GARNET_METAL_PROFILE_OPS='0'),start_new_session=True)
    state=dict(status='running',pid=p.pid,parent_pid=os.getpid(),started_ns=time.time_ns());save(ROOT/'launch1-status.json',state)
    print(json.dumps(state),flush=True)
    try:code=p.wait()
    finally:
        if p.poll() is None:
            p.terminate()
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        state.update(status='terminal',returncode=p.returncode,finished_ns=time.time_ns());save(ROOT/'launch1-status.json',state)
assert not engine_processes(),engine_processes()
for name,h in checks.items():assert sha(Path(name))==h,name
save(ROOT/'postflight1.json',dict(passed=True,checked_files=len(checks),all_owned_processes_terminal=True))
print(json.dumps(state),flush=True)
raise SystemExit(code)
