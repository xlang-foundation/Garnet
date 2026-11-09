"""Serial lifetime guards, full C4 ownership, then B2 interleave; no benchmarks."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
R=Path(__file__).resolve().parent;M=R.parent;W=M.parents[2];C=M/'mistral-scheduled-dense-verification1';Q=M/'mistral-wide-dense-verification1'
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
sys.path.insert(0,str(M/'mistral-fused-q4-decode-native1/tools1'));from run_local_coder_baseline import engine_processes
assert not engine_processes(),engine_processes();assert read(R/'build-audit1.json')['passed'];assert not (R/'run1-preflight.json').exists()
checks=read(C/'qualification1-preflight.json')['files_sha256'];checks.update(read(R/'build-preflight1.json')['files_sha256'])
for name in ['run1.py','worker1.py','probe_worker1.py','guard_worker1.py','prepare_workers1.py','input1.json','probe-input1.json','plan1.json','build-audit1.json','build-on1/libgarnet.dylib','build-default-off1/libgarnet.dylib']:checks[str(R/name)]=sha(R/name)
for p,h in checks.items():assert sha(Path(p))==h,p
save(R/'run1-preflight.json',dict(passed=True,files_sha256=checks,serial=True,no_comparator_execution=True,performance_measured=False))
sequence=[]
for mode in read(R/'plan1.json')['guards']:sequence.append(('guard-'+mode,'guard_worker1.py','build-default-off1' if mode=='off' else 'build-on1','input1.json',mode))
for mode in ['ordinary','verify','reverse']:sequence.append((mode,'worker1.py','build-on1','input1.json',mode))
for mode in ['ordinary','verify']:sequence.append(('probe-'+mode,'probe_worker1.py','build-on1','probe-input1.json',mode))
save(R/'sequence1.json',sequence)
for label,worker,build,fixture,mode in sequence:
 assert not engine_processes(),engine_processes()
 cmd=[str(W/'out/build/macos-arm64-release/bin/xlang3'),'-B',str(R/worker),str(R/build),str(R/'source1'),str(R/fixture),str(R/(label+'-result1.json')),mode];save(R/(label+'-command1.json'),cmd)
 with (R/(label+'-stdout1.log')).open('x') as out,(R/(label+'-stderr1.log')).open('x') as err:
  p=subprocess.Popen(cmd,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',GARNET_METAL_PROFILE_OPS='0'));state=dict(status='running',pid=p.pid,started_ns=time.time_ns());save(R/(label+'-status1.json'),state);print(json.dumps(dict(step=label,**state)),flush=True)
  code=p.wait();state.update(status='terminal',returncode=code,finished_ns=time.time_ns());save(R/(label+'-status1.json'),state)
 print(json.dumps(dict(step=label,**state)),flush=True);assert code==0 and read(R/(label+'-result1.json'))['completed'],(label,code)
assert not engine_processes(),engine_processes()
for p,h in checks.items():assert sha(Path(p))==h,p
save(R/'run1-postflight.json',dict(passed=True,checked_files=len(checks),all_owned_engines_terminal=True,no_comparator_execution=True,performance_measured=False))
print('Wide lifetime, C4 ownership and B2 interleave arms terminal',flush=True)
