"""Bind all inputs, then collect two serial untimed native prefix-lifecycle arms."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parent;W=ROOT.parents[3]
read=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,v):
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
assert not (ROOT/'launch1-status.json').exists() and not (ROOT/'status1.json').exists()
b=read(ROOT/'bindings1.json');J=Path(b['wide_native_root']);I=ROOT.parent/'mistral-half-attention-native1'
sys.path.insert(0,str(ROOT/'tools1'))
from run_local_coder_baseline import engine_processes
assert not engine_processes(),engine_processes()
checks={}
def bind(p,expected=None):
    h=sha(p)
    if expected is not None:assert h==expected,str(p)
    checks[str(p)]=h
bind(J/'provenance1.json',b['native_provenance_sha256'])
bind(J/'model-audit2.json',b['native_model_gate_sha256']);assert read(J/'model-audit2.json')['passed']
bind(J/'admission-audit1.json',b['native_admission_gate_sha256']);assert read(J/'admission-audit1.json')['passed']
bind(Path(b['source_bindings']),b['source_binding_sha256'])
for name,h in read(Path(b['source_bindings']))['files_sha256'].items():bind(Path(b['source'])/name,h)
for name,h in read(I/'source-audit7.json')['original_dirty_sha256'].items():bind(W/'Garnet'/name,h)
for name,h in b['model_files_sha256'].items():bind(Path(b['model'])/name,h)
bind(Path(b['library'])/'libgarnet.dylib',b['library_sha256'])
bind(W/'out/build/macos-arm64-release/bin/xlang3',read(I/'model-bindings7.json')['xlang_sha256'])
bind(ROOT/'tools1/local_coder_batch_worker.py',b['worker_sha256']);bind(ROOT/'fixture1.json',b['fixture_sha256'])
bind(J/'verification-audit3.json');assert read(J/'verification-audit3.json')['passed']
for p in [ROOT/'bindings1.json',ROOT/'profile-template1.json',ROOT/'run_http1.py',ROOT/'launch1.py',ROOT/'prepare1.py',ROOT/'preparation1.json']+sorted((ROOT/'tools1').glob('*.py'))+sorted((ROOT/'audit-tools1').glob('*.py'))+[ROOT/'audit_lifecycle1.py']:bind(p)
save(ROOT/'preflight-hashes1.json',dict(passed=True,files_sha256=checks,inference_serial=True,performance_measurement=False))
command=[str(W/'out/research/qwen-local-coder/asset-env/bin/python'),'-B',str(ROOT/'run_http1.py')]
save(ROOT/'launch1-command.json',command)
with (ROOT/'collection1.stdout').open('x') as out,(ROOT/'collection1.stderr').open('x') as err:
    p=subprocess.Popen(command,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MTL_CAPTURE_ENABLED='0',GARNET_METAL_PROFILE_OPS='0'),start_new_session=True)
    state=dict(status='running',pid=p.pid,parent_pid=os.getpid(),started_ns=time.time_ns());save(ROOT/'launch1-status.json',state)
    print(json.dumps(state),flush=True)
    code=p.wait()
    state.update(status='terminal',returncode=code,finished_ns=time.time_ns());save(ROOT/'launch1-status.json',state)
assert not engine_processes(),engine_processes()
for name,h in checks.items():assert sha(Path(name))==h,name
save(ROOT/'postflight1.json',dict(passed=True,checked_files=len(checks),all_owned_processes_terminal=True))
print(json.dumps(state),flush=True)
raise SystemExit(code)
