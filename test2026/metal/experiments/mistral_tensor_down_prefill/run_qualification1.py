"""Fresh-process serial native correctness qualification, with frozen model and source bindings."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
R=Path(__file__).resolve().parent;W=Path('/Users/shawn/CantorAI');M=R.parent
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(n,v):
 p=R/n;assert not p.exists(),n;p.write_text(json.dumps(v,indent=2)+'\n')
sys.path.insert(0,str(M/'mistral-fused-q4-decode-native1/tools1'))
from run_local_coder_baseline import engine_processes
def idle():
 assert not engine_processes(),engine_processes()
 args={p[0]:p[1] for l in subprocess.check_output(['ps','-axww','-o','pid=,args='],text=True).splitlines() if len(p:=l.strip().split(None,1))==2}
 live=[]
 for l in subprocess.check_output(['ps','-axo','pid=,comm='],text=True).splitlines():
  pid,comm=l.strip().split(None,1)
  if Path(comm).name!='xlang3':continue
  if '/WorldSense/' in args.get(pid,'') and ('package_runtime.py start' in args[pid] or '/Backend/launch.py' in args[pid]):continue
  live.append(dict(pid=pid,command=args.get(pid,comm)))
 assert not live,live
idle();build=read(R/'build-audit1.json');assert build['passed']
checks={str(R/n):sha(R/n) for n in ['ordinary_worker1.py','ordinary-input1.json','lifecycle_worker1.py','lifecycle-input1.json','qualification-plan1.json','run_qualification1.py','build-audit1.json','source-bindings1.json']}
for mode,h in build['library_sha256'].items():
 p=R/('build-'+mode+'1/libgarnet.dylib');assert sha(p)==h;checks[str(p)]=h
for n,h in read(R/'source-bindings1.json')['files_sha256'].items():assert sha(R/'source1'/n)==h,n
for kind in ['target_model','draft_model']:
 model=Path(read(R/'ordinary-input1.json')[kind])
 for n in ['model.safetensors','config.json','tokenizer.json','tokenizer_config.json']:checks[str(model/n)]=sha(model/n)
checks[str(W/'out/build/macos-arm64-release/bin/xlang3')]=sha(W/'out/build/macos-arm64-release/bin/xlang3')
assert checks[read(R/'ordinary-input1.json')['target_model']+'/model.safetensors']=='d7182d886f93a9efa73d6d556ee33ffe839a4e09e15965eeef46f295ab5b503d'
save('qualification1-preflight.json',dict(passed=True,files_sha256=checks,collection_serial=True,performance_claim=False,no_comparator_execution=True))
for case in read(R/'qualification-plan1.json')['cases']:
 idle();name=case['name'];worker=case['worker'];out=R/(name+'-result1.json')
 cmd=[str(W/'out/build/macos-arm64-release/bin/xlang3'),'-B',str(R/(worker+'_worker1.py')),str(R/('build-'+case['build']+'1')),str(R/'source1'),str(R/(worker+'-input1.json')),str(out),case['configuration']]
 save(name+'-command1.json',cmd);start=time.time_ns()
 with (R/(name+'-stdout1.log')).open('x') as o,(R/(name+'-stderr1.log')).open('x') as e:
  p=subprocess.Popen(cmd,stdout=o,stderr=e,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',GARNET_METAL_PROFILE_OPS='0'));save(name+'-pid1.json',dict(pid=p.pid,started_ns=start));print(name,p.pid,'running',flush=True);rc=p.wait()
 save(name+'-status1.json',dict(pid=p.pid,status='terminal',returncode=rc,started_ns=start,finished_ns=time.time_ns()));print(name,rc,'terminal',flush=True)
 if rc:print((R/(name+'-stderr1.log')).read_text()[-3000:],flush=True)
 assert rc==0 and read(out)['completed'],(name,rc)
idle()
for n,h in checks.items():assert sha(Path(n))==h,n
for n,h in read(R/'source-bindings1.json')['files_sha256'].items():assert sha(R/'source1'/n)==h,n
save('qualification1-postflight.json',dict(passed=True,checked_files=len(checks),all_owned_processes_terminal=True,no_comparator_execution=True))
