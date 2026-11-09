"""Serial fresh native model qualification of helper prefill upper bounds16 and256."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
R=Path(__file__).resolve().parent;M=R.parent;W=Path('/Users/shawn/CantorAI');V=M/'mistral-fused-verify-gqa-profile1';read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(n,d):
 p=R/n;assert not p.exists();p.write_text(json.dumps(d,indent=2)+'\n')
sys.path.insert(0,str(M/'mistral-fused-q4-decode-native1/tools1'));from run_local_coder_baseline import engine_processes
assert not engine_processes(),engine_processes();plan=read(R/'plan1.json');assert read(V/'qualification-audit1.json')['passed']
checks={str(R/n):sha(R/n) for n in ['run_qualification1.py','b1_worker1.py','input1.json','plan1.json','prepare1.py','prepare2.py']};lib=Path(plan['library'])/'libgarnet.dylib';assert sha(lib)==plan['library_sha256'];checks[str(lib)]=sha(lib)
for n,h in read(V/'source-bindings1.json')['files_sha256'].items():assert sha(Path(plan['source'])/n)==h,n
inputs=read(R/'input1.json')
for key in ['target_model','draft_model']:
 for n in ['model.safetensors','config.json','tokenizer.json','tokenizer_config.json']:p=Path(inputs[key])/n;checks[str(p)]=sha(p)
for item in inputs['cases']:p=Path(item['control']);checks[str(p)]=sha(p)
checks[str(W/'out/build/macos-arm64-release/bin/xlang3')]=sha(W/'out/build/macos-arm64-release/bin/xlang3')
save('qualification1-preflight.json',dict(passed=True,files_sha256=checks,collection_serial=True,performance_admitted=False,no_comparator_execution=True))
for cap in [16,256]:
 assert not engine_processes(),engine_processes();name='cap'+str(cap);cmd=[str(W/'out/build/macos-arm64-release/bin/xlang3'),'-B',str(R/'b1_worker1.py'),plan['library'],plan['source'],str(R/'input1.json'),str(R/(name+'-result1.json')),'qualify',str(cap)];save(name+'-command1.json',cmd);start=time.time_ns()
 with (R/(name+'-stdout1.log')).open('x') as o,(R/(name+'-stderr1.log')).open('x') as e:
  p=subprocess.Popen(cmd,stdout=o,stderr=e,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',GARNET_METAL_PROFILE_OPS='0'));save(name+'-pid1.json',dict(pid=p.pid,started_ns=start));print(name,p.pid,'running',flush=True);rc=p.wait()
 save(name+'-status1.json',dict(status='terminal',returncode=rc,pid=p.pid,started_ns=start,finished_ns=time.time_ns()));print(name,rc,'terminal',flush=True)
 if rc:print((R/(name+'-stderr1.log')).read_text()[-4000:],flush=True)
 assert rc==0 and read(R/(name+'-result1.json'))['completed']
assert not engine_processes()
for n,h in checks.items():assert sha(Path(n))==h,n
save('qualification1-postflight.json',dict(passed=True,checked_files=len(checks),all_owned_processes_terminal=True,no_comparator_execution=True))
