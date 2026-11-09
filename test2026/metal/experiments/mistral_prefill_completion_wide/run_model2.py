"""Run one native control then one wide candidate, with untimed state witnesses."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parent;M=ROOT.parent;SG=M/'mistral-fused-mlp-serving1';N=M/'mistral-fused-mlp-native1'
read=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,v):
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
sys.path.insert(0,str(ROOT/'tools1'))
from run_local_coder_baseline import engine_processes
assert not engine_processes(),engine_processes()
assert not (ROOT/'model2-preflight.json').exists()
assert read(ROOT/'build-audit2.json')['passed']
checks={}
def bind(p,h=None):
    actual=sha(p)
    if h is not None:assert actual==h,str(p)
    checks[str(p)]=actual
for name,h in read(ROOT/'source-bindings2.json')['files_sha256'].items():bind(ROOT/'source2'/name,h)
for name,h in read(M/'mistral-half-attention-native1/source-audit7.json')['original_dirty_sha256'].items():bind(M.parents[2]/'Garnet'/name,h)
sgplan=read(SG/'plan1.json');model=Path(sgplan['model'])
for name,h in sgplan['model_files_sha256'].items():bind(model/name,h)
binding=read(Path(sgplan['native_model_binding']));xlang=Path(binding['xlang']);bind(xlang,binding['xlang_sha256'])
for name in ['source-bindings2.json','build-audit2.json','library2/libgarnet.dylib','native-plan2.json','worker-model2.py','model-preparation2.json','profile-model-control2.json','profile-model-wide2.json','run_model2.py']:bind(ROOT/name)
bind(ROOT/'library2/libgarnet.dylib',read(ROOT/'build-audit2.json')['library_sha256'])
for p in [N/'provenance1.json',N/'model1.stdout',N/'model1-audit.json',Path(read(ROOT/'profile-model-control2.json')['diagnostic_fixture'])]:bind(p)
save(ROOT/'model2-preflight.json',dict(passed=True,files_sha256=checks,no_comparator_execution=True,performance_measurement=False,state_witnesses=True))
for label in ['control','wide']:
    assert not engine_processes(),engine_processes()
    name='model-'+label+'2';assert not (ROOT/(name+'-status.json')).exists()
    command=[str(xlang),'-B',str(ROOT/'worker-model2.py'),str(ROOT/'library2'),str(ROOT/'source2'),str(model),str(ROOT/('profile-model-'+label+'2.json'))]
    save(ROOT/(name+'-command.json'),command)
    with (ROOT/(name+'.stdout')).open('x') as out,(ROOT/(name+'.stderr')).open('x') as err:
        p=subprocess.Popen(command,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',GARNET_METAL_PROFILE_OPS='0'))
        state=dict(status='running',pid=p.pid,parent_pid=os.getpid(),started_ns=time.time_ns());save(ROOT/(name+'-status.json'),state)
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
save(ROOT/'model2-postflight.json',dict(passed=True,checked_files=len(checks),all_engines_terminal=True,performance_measurement=False))
