"""New32-output state control and clipped candidate, strictly serial and untimed."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parent;M=ROOT.parent;J=M/'mistral-prefill-completion-wide1';SG=M/'mistral-fused-mlp-serving1'
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
assert not engine_processes(),engine_processes()
assert read(ROOT/'build-audit1.json')['passed'] and not (ROOT/'model1-preflight.json').exists()
checks={}
def bind(p,h=None):
    actual=sha(p)
    if h is not None:assert actual==h,str(p)
    checks[str(p)]=actual
for directory,manifest in [(ROOT/'source1',ROOT/'source-bindings1.json'),(J/'source2',J/'source-bindings2.json')]:
    for name,h in read(manifest)['files_sha256'].items():bind(directory/name,h)
for name,h in read(M/'mistral-half-attention-native1/source-audit7.json')['original_dirty_sha256'].items():bind(M.parents[2]/'Garnet'/name,h)
plan=read(SG/'plan1.json');model=Path(plan['model'])
for name,h in plan['model_files_sha256'].items():bind(model/name,h)
b=read(Path(plan['native_model_binding']));xlang=Path(b['xlang']);bind(xlang,b['xlang_sha256'])
for name in ['source-bindings1.json','build-audit1.json','worker-model1.py','fixture-model1.json','model-preparation1.json','profile-model-control1.json','profile-model-clip1.json','run_model1.py']:bind(ROOT/name)
bind(ROOT/'library1/libgarnet.dylib',read(ROOT/'build-audit1.json')['library_sha256'])
bind(J/'library2/libgarnet.dylib',read(J/'build-audit2.json')['library_sha256'])
bind(J/'model-control2.stdout',read(J/'model-audit2.json')['stdout_sha256']['control']);bind(J/'model-audit2.json')
save(ROOT/'model1-preflight.json',dict(passed=True,files_sha256=checks,no_comparator_execution=True,performance_measurement=False,new_control_reason='Keep all three surviving decoders live to exercise509/510/511 actual prefill counts.'))
for label in ['control','clip']:
    assert not engine_processes(),engine_processes()
    name='model-'+label+'1';assert not (ROOT/(name+'-status.json')).exists()
    library=J/'library2' if label=='control' else ROOT/'library1'
    source=J/'source2' if label=='control' else ROOT/'source1'
    command=[str(xlang),'-B',str(ROOT/'worker-model1.py'),str(library),str(source),str(model),str(ROOT/('profile-model-'+label+'1.json'))]
    save(ROOT/(name+'-command.json'),command)
    with (ROOT/(name+'.stdout')).open('x') as out,(ROOT/(name+'.stderr')).open('x') as err:
        p=subprocess.Popen(command,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',GARNET_METAL_PROFILE_OPS='0'))
        state=dict(status='running',pid=p.pid,started_ns=time.time_ns());save(ROOT/(name+'-status.json'),state);print(json.dumps(dict(step=name,**state)),flush=True)
        try:code=p.wait()
        finally:
            if p.poll() is None:
                p.terminate()
                try:p.wait(timeout=30)
                except subprocess.TimeoutExpired:p.kill();p.wait()
            state.update(status='terminal',returncode=p.returncode,finished_ns=time.time_ns());save(ROOT/(name+'-status.json'),state)
    print(json.dumps(dict(step=name,**state)),flush=True);assert code==0,(name,code)
assert not engine_processes(),engine_processes()
for name,h in checks.items():assert sha(Path(name))==h,name
save(ROOT/'model1-postflight.json',dict(passed=True,checked_files=len(checks),all_owned_processes_terminal=True,performance_measurement=False))
