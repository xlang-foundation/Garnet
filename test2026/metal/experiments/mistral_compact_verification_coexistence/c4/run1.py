"""Serial current ordinary then scheduled verification C4 ownership gate."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
R=Path(__file__).resolve().parent;M=R.parent;C=M/'mistral-scheduled-dense-verification1';N=M/'mistral-fused-q4-decode-native1';W=M.parents[2]
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,d):
 t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
sys.path.insert(0,str(N/'tools1'));from run_local_coder_baseline import engine_processes
assert not engine_processes(),engine_processes();assert not (R/'preflight1.json').exists();assert read(C/'qualification-audit2.json')['passed']
checks=read(C/'qualification1-preflight.json')['files_sha256']
for p in [R/'plan1.json',R/'input1.json',R/'worker1.py',R/'run1.py',C/'qualification-audit2.json',C/'provenance1.json']:checks[str(p)]=sha(p)
for n,h in checks.items():assert sha(Path(n))==h,n
save(R/'preflight1.json',dict(passed=True,files_sha256=checks,inference_serial=True,no_comparator_execution=True,performance_measured=False))
for mode in read(R/'plan1.json')['sequence']:
 assert not engine_processes(),engine_processes();cmd=[str(W/'out/build/macos-arm64-release/bin/xlang3'),'-B',str(R/'worker1.py'),str(C/'build-on1'),str(C/'source1'),str(R/'input1.json'),str(R/(mode+'-result1.json')),mode];save(R/(mode+'-command1.json'),cmd)
 with (R/(mode+'-stdout1.log')).open('x') as out,(R/(mode+'-stderr1.log')).open('x') as err:
  p=subprocess.Popen(cmd,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',GARNET_METAL_PROFILE_OPS='0'));state=dict(status='running',pid=p.pid,started_ns=time.time_ns());save(R/(mode+'-status1.json'),state);print(json.dumps(dict(mode=mode,**state)),flush=True);code=p.wait();state.update(status='terminal',returncode=code,finished_ns=time.time_ns());save(R/(mode+'-status1.json'),state)
 print(json.dumps(dict(mode=mode,**state)),flush=True);assert code==0 and read(R/(mode+'-result1.json'))['completed'],(mode,code)
assert not engine_processes(),engine_processes()
for n,h in checks.items():assert sha(Path(n))==h,n
save(R/'postflight1.json',dict(passed=True,checked_files=len(checks),all_owned_engines_terminal=True,no_comparator_execution=True));print('Both serial C4 ownership arms terminal',flush=True)
