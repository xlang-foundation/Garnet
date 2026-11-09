"""Collect native candidate admission only; no comparator inference or builds."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parent;M=ROOT.parent;SG=M/'mistral-fused-mlp-serving1'
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
assert not (ROOT/'admission1-status.json').exists()
build=read(ROOT/'build-audit1.json');plan=read(ROOT/'admission-plan1.json')
assert build['passed'] and '100% tests passed out of 52' in (ROOT/'suite1.stdout').read_text()
assert sha(ROOT/'library1/libgarnet.dylib')==build['library_sha256']
spec=read(ROOT/'engine1.json');assert sha(ROOT/'engine1.json')==plan['engine_sha256'] and spec['native_library_sha256'] is None
spec['native_library_sha256']=build['library_sha256'];assert not (ROOT/'engine2.json').exists();save(ROOT/'engine2.json',spec)
checks={}
def bind(p,h=None):
    actual=sha(p)
    if h is not None:assert actual==h,str(p)
    checks[str(p)]=actual
for name,h in read(ROOT/'source-bindings1.json')['files_sha256'].items():bind(ROOT/'source1'/name,h)
for relative,h in read(M/'mistral-half-attention-native1/source-audit7.json')['original_dirty_sha256'].items():bind(M.parents[2]/'Garnet'/relative,h)
for p in (ROOT/'tools1').glob('*.py'):bind(p)
for relative,h in read(SG/'plan1.json')['model_files_sha256'].items():bind(Path(read(SG/'plan1.json')['model'])/relative,h)
binding=read(Path(read(SG/'plan1.json')['native_model_binding']));bind(Path(binding['xlang']),binding['xlang_sha256'])
for name in ['plan1.json','source-bindings1.json','build-audit1.json','library1/libgarnet.dylib','admission-plan1.json','profile1.json','engine1.json','engine2.json','run_admission1.py']:
    bind(ROOT/name)
for p in [Path(plan['fixture']),Path(plan['serving_manifest']),Path(spec['saved_native_control'])]:bind(p)
for entry in read(Path(plan['serving_manifest']))['plans']:
    bind(Path(entry['cases']),entry['cases_sha256']);bind(Path(entry['workload']))
save(ROOT/'admission1-preflight.json',dict(passed=True,files_sha256=checks,no_comparator_execution=True,source_model_hashing_during_collection=False,serving_qualified=False,started_ns=time.time_ns()))
command=[sys.executable,'-B',str(ROOT/'tools1/run_mistral_cold_prefix.py'),'--engine-spec',str(ROOT/'engine2.json'),'--fixture',plan['fixture'],'--serving-manifest',plan['serving_manifest'],'--case-id',plan['case'],'--output',str(ROOT/'admission1')]
save(ROOT/'admission1-command.json',command)
with (ROOT/'admission1.stdout').open('x') as out,(ROOT/'admission1.stderr').open('x') as err:
    p=subprocess.Popen(command,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
    state=dict(status='running',pid=p.pid,parent_pid=os.getpid(),started_ns=time.time_ns());save(ROOT/'admission1-status.json',state)
    print(json.dumps(state),flush=True)
    try:code=p.wait()
    finally:
        if p.poll() is None:
            p.terminate()
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        state.update(status='terminal',returncode=p.returncode,finished_ns=time.time_ns());save(ROOT/'admission1-status.json',state)
assert code==0,(code,(ROOT/'admission1.stderr').read_text())
assert not engine_processes(),engine_processes()
for name,h in checks.items():assert sha(Path(name))==h,name
save(ROOT/'admission1-postflight.json',dict(passed=True,checked_files=len(checks),all_engines_terminal=True,finished_ns=time.time_ns()))
print(json.dumps(state),flush=True)
