"""Serial current ordinary versus bounded-draft state qualification; no HTTP."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
R=Path(__file__).resolve().parent;M=R.parent;S=M/'mistral-fused-q4-decode-coder1';N=M/'mistral-fused-q4-decode-native1';W=M.parents[2]
read=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,v):
    t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
sys.path.insert(0,str(N/'tools1'));from run_local_coder_baseline import engine_processes
assert not engine_processes(),engine_processes()
assert not (R/'qualification1-status.json').exists()
assert read(S/'audit3.json')['passed']
checks=read(S/'reference3-preflight.json')['files_sha256']
for n,h in checks.items():assert sha(Path(n))==h,n
for n in ['prepare1.py','preparation1.json','input1.json','frozen-control1.json','worker1.py','run_qualification1.py']:checks[str(R/n)]=sha(R/n)
for source in read(R/'frozen-control1.json')['sources']:assert sha(Path(source['path']))==source['sha256'];checks[source['path']]=source['sha256']
save(R/'qualification1-preflight.json',dict(passed=True,files_sha256=checks,performance_admission=False,no_comparator_execution=True))
command=[str(W/'out/build/macos-arm64-release/bin/xlang3'),'-B',str(R/'worker1.py'),str(N/'build-default-off3'),str(N/'source3'),str(R/'input1.json'),str(R/'qualification1-result.json'),'qualify'];save(R/'qualification1-command.json',command)
with (R/'qualification1.stdout').open('x') as out,(R/'qualification1.stderr').open('x') as err:
    p=subprocess.Popen(command,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',GARNET_METAL_PROFILE_OPS='0'))
    state=dict(status='running',pid=p.pid,started_ns=time.time_ns());save(R/'qualification1-status.json',state);print(json.dumps(state),flush=True)
    code=p.wait();state.update(status='terminal',returncode=code,finished_ns=time.time_ns());save(R/'qualification1-status.json',state)
assert not engine_processes(),engine_processes()
for n,h in checks.items():assert sha(Path(n))==h,n
save(R/'qualification1-postflight.json',dict(passed=True,all_owned_processes_terminal=True,checked_files=len(checks),performance_admission=False));print(json.dumps(state),flush=True)
raise SystemExit(code)
