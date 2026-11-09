"""Run a new current-build ordinary reference; saved timed stages stay untouched."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
R=Path(__file__).resolve().parent;M=R.parent;N=M/'mistral-fused-q4-decode-native1';W=M.parents[2]
read=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,v):
    t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
sys.path.insert(0,str(N/'tools1'));from run_local_coder_baseline import engine_processes
assert not (R/'reference3-status.json').exists() and not engine_processes(),engine_processes()
assert read(R/'postflight1.json')['passed'];checks=read(R/'preflight1.json')['files_sha256']
for n,h in checks.items():assert sha(Path(n))==h,n
for n in ['worker-reference3.py','reference-plan3.json','run_reference3.py']:checks[str(R/n)]=sha(R/n)
assert checks[str(R/'worker-reference3.py')]==read(R/'reference-plan3.json')['worker_sha256']
save(R/'reference3-preflight.json',dict(passed=True,files_sha256=checks,performance_measurement=False))
plan=read(R/'plan1.json');command=[str(W/'out/build/macos-arm64-release/bin/xlang3'),'-B',str(R/'worker-reference3.py'),plan['libraries']['control'],plan['source'],plan['input'],str(R/'reference3-result.json'),'occupancy-control'];save(R/'reference3-command.json',command)
with (R/'reference3.stdout').open('x') as out,(R/'reference3.stderr').open('x') as err:
    p=subprocess.Popen(command,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',GARNET_METAL_PROFILE_OPS='0'))
    state=dict(status='running',pid=p.pid,started_ns=time.time_ns());save(R/'reference3-status.json',state);print(json.dumps(state),flush=True)
    code=p.wait();state.update(status='terminal',returncode=code,finished_ns=time.time_ns());save(R/'reference3-status.json',state)
assert code==0 and read(R/'reference3-result.json')['completed']
assert not engine_processes(),engine_processes()
for n,h in checks.items():assert sha(Path(n))==h,n
save(R/'reference3-postflight.json',dict(passed=True,all_owned_processes_terminal=True,checked_files=len(checks),performance_measurement=False));print(json.dumps(state),flush=True)
