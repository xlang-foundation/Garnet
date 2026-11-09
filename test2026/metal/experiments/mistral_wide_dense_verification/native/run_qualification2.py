"""Serial compact/verification preparation guards and three managed real-model arms."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
R=Path(__file__).resolve().parent;M=R.parent;L=M/'mistral-scheduled-c4-ownership2';C=M/'mistral-scheduled-dense-verification1';N=M/'mistral-fused-q4-decode-native1';W=M.parents[2]
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,d):
 t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
sys.path.insert(0,str(N/'tools1'));from run_local_coder_baseline import engine_processes
assert not engine_processes(),engine_processes();assert read(R/'build-audit2.json')['passed'] and read(L/'audit1.json')['passed'];assert not (R/'qualification2-preflight.json').exists()
checks=read(C/'qualification1-preflight.json')['files_sha256'];checks.update(read(R/'build-preflight2.json')['files_sha256'])
for p in [R/'run_qualification2.py',R/'worker1.py',R/'input2.json',R/'qualification-plan2.json',R/'build-audit2.json',R/'build-on2/libgarnet.dylib',R/'build-default-off2/libgarnet.dylib',L/'audit1.json']:checks[str(p)]=sha(p)
for n,h in checks.items():assert sha(Path(n))==h,n
save(R/'qualification2-preflight.json',dict(passed=True,files_sha256=checks,collection_serial=True,no_comparator_execution=True,performance_measured=False))
sequence=[('ordinary','worker1.py','build-on2','ordinary'),('verify','worker1.py','build-on2','verify'),('reverse','worker1.py','build-on2','reverse')]
save(R/'qualification2-sequence.json',sequence)
for label,worker,build,mode in sequence:
 if mode!='ordinary':
  control=R/'ordinary-result2.json';assert read(control)['completed'];checks[str(control)]=sha(control);save(R/(label+'-proposal-oracle-binding2.json'),dict(path=str(control),sha256=sha(control),correctness_only=True))
 assert not engine_processes(),engine_processes();cmd=[str(W/'out/build/macos-arm64-release/bin/xlang3'),'-B',str(R/worker),str(R/build),str(R/'source2'),str(R/'input2.json'),str(R/(label+'-result2.json')),mode];save(R/(label+'-command2.json'),cmd)
 with (R/(label+'-stdout2.log')).open('x') as out,(R/(label+'-stderr2.log')).open('x') as err:
  p=subprocess.Popen(cmd,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',GARNET_METAL_PROFILE_OPS='0'));state=dict(status='running',pid=p.pid,started_ns=time.time_ns());save(R/(label+'-status2.json'),state);print(json.dumps(dict(step=label,**state)),flush=True);code=p.wait();state.update(status='terminal',returncode=code,finished_ns=time.time_ns());save(R/(label+'-status2.json'),state)
 print(json.dumps(dict(step=label,**state)),flush=True);assert code==0 and read(R/(label+'-result2.json'))['completed'],(label,code)
assert not engine_processes(),engine_processes()
for n,h in checks.items():assert sha(Path(n))==h,n
save(R/'qualification2-postflight.json',dict(passed=True,checked_files=len(checks),all_owned_engines_terminal=True,no_comparator_execution=True));print('All compact/verification guards and managed model arms terminal',flush=True)
