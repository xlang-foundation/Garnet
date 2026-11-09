"""Four fresh bounded native stages, OFF/ON then ON/OFF, strictly serial."""
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
assert not engine_processes(),engine_processes();assert read(R/'qualification-audit1.json')['passed']
assert not (R/'measurement1-preflight.json').exists();plan=read(R/'stage-plan1.json')
checks=read(R/'qualification1-preflight.json')['files_sha256']
for p in [R/'run_measurement1.py',R/'qualification-audit1.json',R/'audit_qualification1.py',R/'audit_measurement1.py',R/'qualification-on1-result.json',R/'qualification-off1-result.json']:checks[str(p)]=sha(p)
for n,h in checks.items():assert sha(Path(n))==h,n
save(R/'measurement1-preflight.json',dict(passed=True,files_sha256=checks,collection_serial=True,no_comparator_execution=True,hashes_during_collection=False))
for cohort,mode in plan['sequence']:
 assert not engine_processes(),engine_processes()
 name=f'cohort{cohort}-{mode}';cmd=[str(W/'out/build/macos-arm64-release/bin/xlang3'),'-B',str(R/'worker1.py'),plan['libraries'][mode],plan['source'],plan['input'],str(R/(name+'-result1.json')),'bounded'];save(R/(name+'-command1.json'),cmd)
 with (R/(name+'-stdout1.log')).open('x') as out,(R/(name+'-stderr1.log')).open('x') as err:
  p=subprocess.Popen(cmd,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',GARNET_METAL_PROFILE_OPS='0'));state=dict(status='running',pid=p.pid,started_ns=time.time_ns());save(R/(name+'-status1.json'),state);print(json.dumps(dict(stage=name,**state)),flush=True);code=p.wait();state.update(status='terminal',returncode=code,finished_ns=time.time_ns());save(R/(name+'-status1.json'),state)
 print(json.dumps(dict(stage=name,**state)),flush=True);assert code==0 and read(R/(name+'-result1.json'))['completed'],(name,code)
assert not engine_processes(),engine_processes()
for n,h in checks.items():assert sha(Path(n))==h,n
save(R/'measurement1-postflight.json',dict(passed=True,all_owned_processes_terminal=True,checked_files=len(checks),no_comparator_execution=True));print('All four native measurements terminal',flush=True)
