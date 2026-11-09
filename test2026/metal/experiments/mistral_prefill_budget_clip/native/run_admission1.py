"""Collect one new native budget-clip serving candidate; no builds/hashing during collection."""
from pathlib import Path
import hashlib,json,subprocess,sys,time,os
ROOT=Path(__file__).resolve().parent;M=ROOT.parent;W=M.parents[2];SG=M/'mistral-fused-mlp-serving1'
read=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,v):
    t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
sys.path.insert(0,str(ROOT/'tools1'));from run_local_coder_baseline import engine_processes
assert not engine_processes(),engine_processes();assert not(ROOT/'admission1-status.json').exists();assert read(ROOT/'model-audit1.json')['passed'] and read(ROOT/'build-audit1.json')['passed']
plan=read(ROOT/'admission-plan1.json');spec=read(ROOT/'engine-admission1.json');checks={}
def bind(p,h=None):
    actual=sha(p)
    if h is not None:assert actual==h,str(p)
    checks[str(p)]=actual
for name,h in read(ROOT/'source-bindings1.json')['files_sha256'].items():bind(ROOT/'source1'/name,h)
for name,h in read(M/'mistral-half-attention-native1/source-audit7.json')['original_dirty_sha256'].items():bind(W/'Garnet'/name,h)
for name,h in read(SG/'plan1.json')['model_files_sha256'].items():bind(Path(read(SG/'plan1.json')['model'])/name,h)
b=read(Path(read(SG/'plan1.json')['native_model_binding']));bind(Path(b['xlang']),b['xlang_sha256'])
for p in (ROOT/'tools1').glob('*.py'):bind(p)
for p in (ROOT/'audit-tools1').glob('*.py'):bind(p)
for name in ['run_admission1.py','audit_admission1.py','source-bindings1.json','model-audit1.json','build-audit1.json','admission-plan1.json','admission-preparation1.json','profile-admission1.json','engine-admission1.json']:bind(ROOT/name)
bind(ROOT/'library1/libgarnet.dylib',spec['native_library_sha256'])
for p in [Path(plan['fixture']),Path(plan['serving_manifest']),Path(spec['saved_native_control'])]:bind(p)
for p in read(Path(plan['serving_manifest']))['plans']:bind(Path(p['cases']),p['cases_sha256']);bind(Path(p['workload']))
save(ROOT/'admission1-preflight.json',dict(passed=True,files_sha256=checks,no_comparator_execution=True,source_model_hashing_during_collection=False))
command=[sys.executable,'-B',str(ROOT/'tools1/run_mistral_cold_prefix.py'),'--engine-spec',str(ROOT/'engine-admission1.json'),'--fixture',plan['fixture'],'--serving-manifest',plan['serving_manifest'],'--case-id',plan['case'],'--output',str(ROOT/'admission1')];save(ROOT/'admission1-command.json',command)
with (ROOT/'admission1.stdout').open('x') as out,(ROOT/'admission1.stderr').open('x') as err:
    p=subprocess.Popen(command,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));state=dict(status='running',pid=p.pid,started_ns=time.time_ns());save(ROOT/'admission1-status.json',state);print(json.dumps(state),flush=True);code=p.wait();state.update(status='terminal',returncode=code,finished_ns=time.time_ns());save(ROOT/'admission1-status.json',state)
assert not engine_processes(),engine_processes()
for name,h in checks.items():assert sha(Path(name))==h,name
save(ROOT/'admission1-postflight.json',dict(passed=True,checked_files=len(checks),all_engines_terminal=True));print(json.dumps(state),flush=True);raise SystemExit(code)
