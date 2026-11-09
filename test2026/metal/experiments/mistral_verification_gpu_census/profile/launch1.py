"""Serial counter replay of the qualified current target, with pinned raw inputs."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
R=Path(__file__).resolve().parent;M=R.parent;N=M/'mistral-fused-q4-decode-native1';S=M/'mistral-fused-q4-decode-coder1';W=M.parents[2]
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
assert not (R/'status1.json').exists()
assert read(S/'audit3.json')['passed']
checks=read(S/'reference3-preflight.json')['files_sha256']
for n,h in checks.items():assert sha(Path(n))==h,n
for p in [R/'plan1.json',R/'worker1.py',R/'launch1.py',S/'cohort1-control-result1.json',S/'audit3.json']:checks[str(p)]=sha(p)
save(R/'preflight1.json',dict(passed=True,files_sha256=checks,diagnostic_only=True,preserve_graph_fusions=True,no_comparator_execution=True))
command=[str(W/'out/build/macos-arm64-release/bin/xlang3'),'-B',str(R/'worker1.py'),str(N/'build-default-off3'),str(N/'source3'),str(S/'cohort1-control-result1.json'),str(R/'result1.json')];save(R/'command1.json',command)
with (R/'capture1.log').open('x') as out:
    p=subprocess.Popen(command,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',GARNET_METAL_PROFILE_OPS='0'))
    state=dict(status='running',pid=p.pid,started_ns=time.time_ns());save(R/'status1.json',state);print(json.dumps(state),flush=True)
    code=p.wait();state.update(status='terminal',returncode=code,finished_ns=time.time_ns());save(R/'status1.json',state)
assert not engine_processes(),engine_processes()
for n,h in checks.items():assert sha(Path(n))==h,n
save(R/'postflight1.json',dict(passed=True,all_owned_processes_terminal=True,checked_files=len(checks),diagnostic_only=True));print(json.dumps(state),flush=True)
raise SystemExit(code)
