"""Serial fresh-process helper B4 correctness collection; no performance claim."""
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
 lines=subprocess.check_output(['ps','-axo','pid=,comm='],text=True).splitlines()
 args={p[0]:p[1] for l in subprocess.check_output(['ps','-axww','-o','pid=,args='],text=True).splitlines() if len(p:=l.strip().split(None,1))==2}
 live=[]
 for line in lines:
  pid,comm=line.strip().split(None,1)
  if Path(comm).name!='xlang3':continue
  # Unrelated, pre-existing WorldSense service: verify its current command.
  if '/WorldSense/' in args.get(pid,'') and ("package_runtime.py start" in args[pid] or '/Backend/launch.py' in args[pid]):continue
  live.append(dict(pid=pid,command=args.get(pid,comm)))
 assert not live,live
idle();assert read(R/'build-audit1.json')['passed']
checks={str(R/n):sha(R/n) for n in ['helper_worker1.py','run_helper1.py','input1.json','build-audit1.json','source-bindings1.json']}
checks[str(R/'build-on1/libgarnet.dylib')]=read(R/'build-audit1.json')['library_sha256']
draft=Path(read(R/'input1.json')['draft_model'])
for n in ['model.safetensors','config.json','tokenizer.json','tokenizer_config.json']:checks[str(draft/n)]=sha(draft/n)
checks[str(W/'out/build/macos-arm64-release/bin/xlang3')]=sha(W/'out/build/macos-arm64-release/bin/xlang3')
for n,h in checks.items():assert sha(Path(n))==h,n
save('helper1-preflight.json',dict(passed=True,files_sha256=checks,collection_serial=True,performance_claim=False,no_comparator_execution=True))
for mode in ['serial','batch']:
 idle();cmd=[str(W/'out/build/macos-arm64-release/bin/xlang3'),'-B',str(R/'helper_worker1.py'),str(R/'build-on1'),str(R/'source1'),str(R/'input1.json'),str(R/(mode+'-result1.json')),mode]
 save(mode+'-command1.json',cmd);start=time.time_ns()
 with (R/(mode+'-stdout1.log')).open('x') as o,(R/(mode+'-stderr1.log')).open('x') as e:
  p=subprocess.Popen(cmd,stdout=o,stderr=e,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',GARNET_METAL_PROFILE_OPS='0'))
  save(mode+'-pid1.json',dict(pid=p.pid,started_ns=start));print(mode,p.pid,'running',flush=True);rc=p.wait()
 save(mode+'-status1.json',dict(pid=p.pid,status='terminal',returncode=rc,started_ns=start,finished_ns=time.time_ns()));print(mode,rc,'terminal',flush=True)
 assert rc==0 and read(R/(mode+'-result1.json'))['completed'],(mode,rc)
idle()
for n,h in checks.items():assert sha(Path(n))==h,n
save('helper1-postflight.json',dict(passed=True,checked_files=len(checks),all_owned_processes_terminal=True,no_comparator_execution=True))
