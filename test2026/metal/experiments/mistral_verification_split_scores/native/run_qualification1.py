"""Qualify fresh native ON/OFF ordinary+bounded runs serially before measurement."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
R=Path(__file__).resolve().parent;M=R.parent;N=M/'mistral-fused-q4-decode-native1';B=M/'mistral-bounded-draft-long1';W=M.parents[2]
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,d):
 t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
sys.path.insert(0,str(N/'tools1'));from run_local_coder_baseline import engine_processes
assert not engine_processes(),engine_processes();assert read(R/'build-audit1.json')['passed']
assert not (R/'qualification1-preflight.json').exists();plan=read(R/'stage-plan1.json')
checks=read(B/'qualification1-preflight.json')['files_sha256']
checks.update(read(R/'build-preflight1.json')['files_sha256'])
for mode,path in plan['libraries'].items():checks[path+'/libgarnet.dylib']=read(R/'build-audit1.json')['library_sha256'][mode]
for p in [R/'run_qualification1.py',R/'stage-plan1.json',R/'worker-preparation1.json',R/'worker1.py',R/'build-audit1.json',B/'qualification-audit1.json',B/'qualification1-result.json',B/'audit_qualification1.py',B/'audit_bounded_common1.py',B/'input1.json']:checks[str(p)]=sha(p)
assert sha(R/'worker1.py')==plan['worker_sha256']
for n,h in checks.items():assert sha(Path(n))==h,n
save(R/'qualification1-preflight.json',dict(passed=True,files_sha256=checks,collection_serial=True,no_comparator_execution=True,hashes_during_collection=False))
for mode in plan['qualification_modes']:
 assert not engine_processes(),engine_processes()
 name='qualification-'+mode+'1';cmd=[str(W/'out/build/macos-arm64-release/bin/xlang3'),'-B',str(R/'worker1.py'),plan['libraries'][mode],plan['source'],plan['input'],str(R/(name+'-result.json')),'qualify'];save(R/(name+'-command.json'),cmd)
 with (R/(name+'.stdout')).open('x') as out,(R/(name+'.stderr')).open('x') as err:
  p=subprocess.Popen(cmd,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',GARNET_METAL_PROFILE_OPS='0'));state=dict(status='running',pid=p.pid,started_ns=time.time_ns());save(R/(name+'-status.json'),state);print(json.dumps(dict(mode=mode,**state)),flush=True);code=p.wait();state.update(status='terminal',returncode=code,finished_ns=time.time_ns());save(R/(name+'-status.json'),state)
 print(json.dumps(dict(mode=mode,**state)),flush=True);assert code==0 and read(R/(name+'-result.json'))['completed']
assert not engine_processes(),engine_processes()
for n,h in checks.items():assert sha(Path(n))==h,n
save(R/'qualification1-postflight.json',dict(passed=True,all_owned_processes_terminal=True,checked_files=len(checks),no_comparator_execution=True));print('Native ON/OFF qualification terminal',flush=True)
