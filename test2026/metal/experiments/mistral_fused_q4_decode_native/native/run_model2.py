"""Run one new fused-Q4 decode engine against saved state controls; no comparator reruns."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parent;M=ROOT.parent;W=M.parents[2];J=M/'mistral-prefill-completion-wide1';SG=M/'mistral-fused-mlp-serving1'
read=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,v):
    t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
sys.path.insert(0,str(M/'mistral-wide-half-prefix-http1/tools1'));from run_local_coder_baseline import engine_processes
assert not engine_processes(),engine_processes();assert not(ROOT/'model2-status.json').exists();assert read(ROOT/'build-audit3.json')['passed']
assert Path(read(ROOT/'profile-model2.json')['memory_inventory_path']).is_absolute()
checks={}
def bind(p,h=None):
    actual=sha(p)
    if h is not None:assert actual==h,str(p)
    checks[str(p)]=actual
for name,h in read(ROOT/'source-bindings3.json')['files_sha256'].items():bind(ROOT/'source3'/name,h)
for name,h in read(M/'mistral-half-attention-native1/source-audit7.json')['original_dirty_sha256'].items():bind(W/'Garnet'/name,h)
plan=read(SG/'plan1.json');model=Path(plan['model'])
for name,h in plan['model_files_sha256'].items():bind(model/name,h)
b=read(Path(plan['native_model_binding']));xlang=Path(b['xlang']);bind(xlang,b['xlang_sha256'])
for name in ['source-bindings3.json','build-audit3.json','worker-model2.py','model-preparation2.json','profile-model2.json','run_model2.py']:bind(ROOT/name)
bind(ROOT/'library3/libgarnet.dylib',read(ROOT/'build-audit3.json')['library_sha256']);bind(J/'model-control2.stdout');bind(J/'model-audit2.json');bind(Path(read(ROOT/'profile-model2.json')['diagnostic_fixture']))
save(ROOT/'model2-preflight.json',dict(passed=True,files_sha256=checks,performance_measurement=False,no_comparator_execution=True))
command=[str(xlang),'-B',str(ROOT/'worker-model2.py'),str(ROOT/'library3'),str(ROOT/'source3'),str(model),str(ROOT/'profile-model2.json')];save(ROOT/'model2-command.json',command)
with (ROOT/'model2.stdout').open('x') as out,(ROOT/'model2.stderr').open('x') as err:
    p=subprocess.Popen(command,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',GARNET_METAL_PROFILE_OPS='0'));state=dict(status='running',pid=p.pid,started_ns=time.time_ns());save(ROOT/'model2-status.json',state);print(json.dumps(state),flush=True);code=p.wait();state.update(status='terminal',returncode=code,finished_ns=time.time_ns());save(ROOT/'model2-status.json',state)
assert not engine_processes(),engine_processes()
for name,h in checks.items():assert sha(Path(name))==h,name
save(ROOT/'model2-postflight.json',dict(passed=True,checked_files=len(checks),all_owned_processes_terminal=True,performance_measurement=False));print(json.dumps(state),flush=True);raise SystemExit(code)
