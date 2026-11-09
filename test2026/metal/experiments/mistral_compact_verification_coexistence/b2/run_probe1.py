"""Serial ordinary/verify probe for contiguous B2 owner-span interleaving."""
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
assert not engine_processes(),engine_processes();assert read(R/'qualification1-postflight.json')['passed'];assert not (R/'probe1-preflight.json').exists()
checks=read(R/'qualification1-preflight.json')['files_sha256']
for p in [R/'probe_worker1.py',R/'probe-input1.json',R/'probe-plan1.json',R/'run_probe1.py']:checks[str(p)]=sha(p)
for n,h in checks.items():assert sha(Path(n))==h,n
save(R/'probe1-preflight.json',dict(passed=True,files_sha256=checks,collection_serial=True,no_comparator_execution=True,performance_measured=False))
for mode in ['ordinary','verify']:
 assert not engine_processes(),engine_processes();label='probe-'+mode;cmd=[str(W/'out/build/macos-arm64-release/bin/xlang3'),'-B',str(R/'probe_worker1.py'),str(R/'build-on1'),str(R/'source1'),str(R/'probe-input1.json'),str(R/(label+'-result1.json')),mode];save(R/(label+'-command1.json'),cmd)
 with (R/(label+'-stdout1.log')).open('x') as out,(R/(label+'-stderr1.log')).open('x') as err:
  p=subprocess.Popen(cmd,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',GARNET_METAL_PROFILE_OPS='0'));state=dict(status='running',pid=p.pid,started_ns=time.time_ns());save(R/(label+'-status1.json'),state);print(json.dumps(dict(step=label,**state)),flush=True);code=p.wait();state.update(status='terminal',returncode=code,finished_ns=time.time_ns());save(R/(label+'-status1.json'),state)
 print(json.dumps(dict(step=label,**state)),flush=True);assert code==0 and read(R/(label+'-result1.json'))['completed'],(label,code)
assert not engine_processes(),engine_processes()
for n,h in checks.items():assert sha(Path(n))==h,n
save(R/'probe1-postflight.json',dict(passed=True,checked_files=len(checks),all_owned_engines_terminal=True,no_comparator_execution=True));print('B2 mixed ownership probe terminal',flush=True)
