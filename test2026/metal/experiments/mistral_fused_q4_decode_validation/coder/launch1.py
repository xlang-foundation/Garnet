"""Bind then run serial fresh native control/candidate/candidate/control stages."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
R=Path(__file__).resolve().parent;M=R.parent;N=M/'mistral-fused-q4-decode-native1';C=M/'mistral-coder-draft-native1';W=M.parents[2]
read=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,v):
    t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
sys.path.insert(0,str(N/'tools1'));from run_local_coder_baseline import engine_processes
assert not (R/'preflight1.json').exists() and not engine_processes(),engine_processes()
plan=read(R/'plan1.json');assert read(Path(plan['prefix_gate']))['passed']
checks={}
def bind(p,h=None):
    v=sha(p)
    if h is not None:assert h==v,str(p)
    checks[str(p)]=v
for n,h in read(Path(plan['source_bindings']))['files_sha256'].items():bind(Path(plan['source'])/n,h)
for n,h in read(M/'mistral-half-attention-native1/source-audit7.json')['original_dirty_sha256'].items():bind(W/'Garnet'/n,h)
bind(N/'provenance1.json',plan['source_native_provenance_sha256'])
for n in ['build-audit3.json','model-audit2.json','verification-audit3.json','admission-audit1.json']:assert read(N/n)['passed'];bind(N/n)
bind(Path(plan['prefix_gate']));bind(Path(plan['source_bindings']))
def options(folder):
    result={}
    for line in (folder/'CMakeCache.txt').read_text().splitlines():
        if line.startswith('GARNET_') and '=' in line:
            key,val=line.split('=',1);result[key.split(':')[0]]=val
    return result
on=options(N/'build3');off=options(N/'build-default-off3')
assert on['GARNET_METAL_FUSED_Q4_DECODE']=='ON' and off['GARNET_METAL_FUSED_Q4_DECODE']=='OFF'
assert {k:v for k,v in on.items() if k!='GARNET_METAL_FUSED_Q4_DECODE'}=={k:v for k,v in off.items() if k!='GARNET_METAL_FUSED_Q4_DECODE'}
for folder in [N/'build3',N/'build-default-off3']:bind(folder/'CMakeCache.txt')
for role,folder in plan['libraries'].items():bind(Path(folder)/'libgarnet.dylib')
assert checks[str(Path(plan['libraries']['candidate'])/'libgarnet.dylib')]==read(N/'build-audit3.json')['library_sha256']
sg=read(M/'mistral-fused-mlp-serving1/plan1.json')
for n,h in sg['model_files_sha256'].items():bind(Path(sg['model'])/n,h)
draft=read(C/'asset-manifest1.json')
for n,row in draft['files'].items():bind(Path(draft['model'])/n,row['sha256'])
inputs=read(Path(plan['input']));bind(Path(plan['input']),plan['input_sha256']);assert sha(Path(plan['input']))==plan['base_input_sha256']
for case in inputs['cases']:bind(Path(case['control']))
for p in [R/'plan1.json',R/'worker1.py',R/'launch1.py',R/'audit1.py',R/'audit_mistral_prediction_confidence.py',Path(plan['reference']),Path(plan['fixture'])]:bind(p)
xlang=W/'out/build/macos-arm64-release/bin/xlang3';bind(xlang,read(M/'mistral-half-attention-native1/model-bindings7.json')['xlang_sha256'])
save(R/'preflight1.json',dict(passed=True,files_sha256=checks,native_option_difference_only='GARNET_METAL_FUSED_Q4_DECODE',sequence=plan['sequence'],no_comparator_execution=True,source_model_hashing_during_collection=False))
for cohort,role in plan['sequence']:
    assert not engine_processes(),engine_processes()
    label='cohort'+str(cohort)+'-'+role;assert not (R/(label+'-status1.json')).exists()
    out=R/(label+'-result1.json');command=[str(xlang),'-B',str(R/'worker1.py'),plan['libraries'][role],plan['source'],plan['input'],str(out),'occupancy-control']
    save(R/(label+'-command1.json'),command)
    with (R/(label+'-stdout1.log')).open('x') as stdout,(R/(label+'-stderr1.log')).open('x') as stderr:
        p=subprocess.Popen(command,stdout=stdout,stderr=stderr,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',GARNET_METAL_PROFILE_OPS='0'))
        state=dict(status='running',pid=p.pid,started_ns=time.time_ns());save(R/(label+'-status1.json'),state);print(json.dumps(dict(label=label,**state)),flush=True)
        code=p.wait();state.update(status='terminal',returncode=code,finished_ns=time.time_ns());save(R/(label+'-status1.json'),state)
    assert code==0 and read(out)['completed'],(label,code)
assert not engine_processes(),engine_processes()
for n,h in checks.items():assert sha(Path(n))==h,n
save(R/'postflight1.json',dict(passed=True,all_owned_processes_terminal=True,checked_files=len(checks)));print('All four native stages terminal',flush=True)
