"""Serial native rollback/state checks at C256 and C4096; oracle proposals only."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
R=Path(__file__).resolve().parent;M=R.parent;SG=M/'mistral-fused-mlp-serving1'
read=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,v):
    t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
sys.path.insert(0,str(M/'mistral-wide-half-prefix-http1/tools1'))
from run_local_coder_baseline import engine_processes
assert not engine_processes(),engine_processes();assert read(R/'model-audit2.json')['passed'];assert not(R/'verification3-preflight.json').exists()
checks={}
def bind(p,h=None):
    actual=sha(p)
    if h is not None:assert actual==h,str(p)
    checks[str(p)]=actual
for name,h in read(R/'source-bindings3.json')['files_sha256'].items():bind(R/'source3'/name,h)
for name,h in read(M/'mistral-half-attention-native1/source-audit7.json')['original_dirty_sha256'].items():bind(M.parents[2]/'Garnet'/name,h)
plan=read(SG/'plan1.json');model=Path(plan['model'])
for name,h in plan['model_files_sha256'].items():bind(model/name,h)
b=read(Path(plan['native_model_binding']));xlang=Path(b['xlang']);bind(xlang,b['xlang_sha256'])
for name in ['worker-verification3.py','verification-preparation3.json','run_verification3.py','model-audit2.json','build-audit3.json','source-bindings3.json']:bind(R/name)
fixture=Path(read(R/'verification-preparation3.json')['fixture']);bind(fixture,read(R/'verification-preparation3.json')['fixture_sha256']);bind(R/'library3/libgarnet.dylib',read(R/'build-audit3.json')['library_sha256'])
save(R/'verification3-preflight.json',dict(passed=True,files_sha256=checks,oracle_proposals_only=True,performance_measurement=False,no_comparator_execution=True))
for capacity in [256,4096]:
    assert not engine_processes(),engine_processes()
    name='verification-c'+str(capacity)+'-3';command=[str(xlang),'-B',str(R/'worker-verification3.py'),str(model),str(R/'library3'),str(R/'source3'),str(fixture),str(R/name),str(capacity)];save(R/(name+'-command.json'),command)
    with (R/(name+'.stdout')).open('x') as out,(R/(name+'.stderr')).open('x') as err:
        p=subprocess.Popen(command,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',GARNET_METAL_PROFILE_OPS='0'));state=dict(status='running',pid=p.pid,started_ns=time.time_ns());save(R/(name+'-status.json'),state);print(json.dumps(dict(step=name,**state)),flush=True);code=p.wait();state.update(status='terminal',returncode=code,finished_ns=time.time_ns());save(R/(name+'-status.json'),state)
    assert code==0,(name,code)
assert not engine_processes(),engine_processes()
for name,h in checks.items():assert sha(Path(name))==h,name
save(R/'verification3-postflight.json',dict(passed=True,checked_files=len(checks),all_owned_processes_terminal=True,performance_measurement=False))
