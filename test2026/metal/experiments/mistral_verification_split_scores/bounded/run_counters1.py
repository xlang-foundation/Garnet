"""One serial diagnostic replay of qualified long-context target calls."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
R=Path(__file__).resolve().parent;M=R.parent;N=M/'mistral-fused-q4-decode-native1';W=M.parents[2]
read=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,d):
    t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
sys.path.insert(0,str(N/'tools1'));from run_local_coder_baseline import engine_processes
assert not engine_processes(),engine_processes()
assert read(R/'measurement-audit1.json')['passed'] and not (R/'counters1-status.json').exists()
checks=read(R/'measurement1-preflight.json')['files_sha256']
for n,h in checks.items():assert sha(Path(n))==h,n
for n in ['measurement-audit1.json','counter-input1.json','worker_counters1.py','audit_counters1.py','run_counters1.py']:checks[str(R/n)]=sha(R/n)
assert read(R/'counter-input1.json')['diagnostic_source_sha256']==sha(R/'qualification1-result.json')
save(R/'counters1-preflight.json',dict(passed=True,files_sha256=checks,diagnostic_only=True,no_comparator_execution=True))
command=[str(W/'out/build/macos-arm64-release/bin/xlang3'),'-B',str(R/'worker_counters1.py'),str(N/'build-default-off3'),str(N/'source3'),str(R/'counter-input1.json'),str(R/'counters1-result.json')];save(R/'counters1-command.json',command)
with (R/'counters1.log').open('x') as out:
    p=subprocess.Popen(command,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',GARNET_METAL_PROFILE_OPS='0'))
    state=dict(status='running',pid=p.pid,started_ns=time.time_ns());save(R/'counters1-status.json',state);print(json.dumps(state),flush=True)
    code=p.wait();state.update(status='terminal',returncode=code,finished_ns=time.time_ns());save(R/'counters1-status.json',state)
assert not engine_processes(),engine_processes()
for n,h in checks.items():assert sha(Path(n))==h,n
save(R/'counters1-postflight.json',dict(passed=True,all_owned_processes_terminal=True,checked_files=len(checks),diagnostic_only=True));print(json.dumps(state),flush=True)
raise SystemExit(code)
