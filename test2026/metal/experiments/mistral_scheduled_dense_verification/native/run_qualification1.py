"""Run one managed native scheduler qualification; no serving or performance claim."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
R=Path(__file__).resolve().parent;M=R.parent;B=M/'mistral-bounded-draft-long1';N=M/'mistral-fused-q4-decode-native1';W=M.parents[2]
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,d):
 t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
sys.path.insert(0,str(N/'tools1'));from run_local_coder_baseline import engine_processes
assert not engine_processes(),engine_processes();assert read(R/'build-audit1.json')['passed'];assert not (R/'qualification1-status.json').exists()
checks=read(B/'qualification1-preflight.json')['files_sha256'];checks.update(read(R/'build-preflight1.json')['files_sha256'])
for p in [R/'run_qualification1.py',R/'worker2.py',R/'worker-repair2.json',R/'worker-preparation1.json',R/'input1.json',R/'plan1.json',R/'build-audit1.json',R/'build-on1/libgarnet.dylib',B/'qualification-audit1.json',B/'qualification1-result.json']:checks[str(p)]=sha(p)
for n,h in checks.items():assert sha(Path(n))==h,n
save(R/'qualification1-preflight.json',dict(passed=True,files_sha256=checks,performance_measurement=False,no_comparator_execution=True))
cmd=[str(W/'out/build/macos-arm64-release/bin/xlang3'),'-B',str(R/'worker2.py'),str(R/'build-on1'),str(R/'source1'),str(R/'input1.json'),str(R/'qualification1-result.json'),'qualify'];save(R/'qualification1-command.json',cmd)
with (R/'qualification1.stdout').open('x') as out,(R/'qualification1.stderr').open('x') as err:
 p=subprocess.Popen(cmd,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',GARNET_METAL_PROFILE_OPS='0'));state=dict(status='running',pid=p.pid,started_ns=time.time_ns());save(R/'qualification1-status.json',state);print(json.dumps(state),flush=True);code=p.wait();state.update(status='terminal',returncode=code,finished_ns=time.time_ns());save(R/'qualification1-status.json',state)
assert not engine_processes(),engine_processes()
for n,h in checks.items():assert sha(Path(n))==h,n
save(R/'qualification1-postflight.json',dict(passed=True,all_owned_engines_terminal=True,checked_files=len(checks),no_comparator_execution=True));print(json.dumps(state),flush=True);raise SystemExit(code)
