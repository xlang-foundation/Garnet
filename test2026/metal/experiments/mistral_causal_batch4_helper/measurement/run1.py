"""Four serial native engines, ordinary/productive/productive/ordinary; no comparator runs."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
R=Path(__file__).resolve().parent;M=R.parent;Q=M/'mistral-causal-batch4-helper1';W=Path('/Users/shawn/CantorAI')
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
 lines=subprocess.check_output(['ps','-axo','pid=,comm='],text=True).splitlines();args={p[0]:p[1] for l in subprocess.check_output(['ps','-axww','-o','pid=,args='],text=True).splitlines() if len(p:=l.strip().split(None,1))==2}
 live=[]
 for line in lines:
  pid,comm=line.strip().split(None,1)
  if Path(comm).name!='xlang3':continue
  if '/WorldSense/' in args.get(pid,'') and ('package_runtime.py start' in args[pid] or '/Backend/launch.py' in args[pid]):continue
  live.append(dict(pid=pid,command=args.get(pid,comm)))
 assert not live,live
idle();assert read(Q/'productive-audit1.json')['passed'] and read(R/'preparation1.json')['passed']
checks=read(Q/'productive1-preflight.json')['files_sha256']
for p in [R/n for n in ['prepare1.py','worker1.py','run1.py','input1.json','preparation1.json','measurement-instrumentation1.patch']]+[Q/'productive-audit1.json',Q/'helper-audit1.json',Q/'audit_productive1.py',Q/'audit_helper1.py']:
 checks[str(p)]=sha(p)
for n,h in checks.items():assert sha(Path(n))==h,n
for n,h in read(Q/'source-bindings1.json')['files_sha256'].items():assert sha(Q/'source1'/n)==h,n
sequence=['ordinary','productive','productive','ordinary'];save('plan1.json',dict(sequence=sequence,waves_per_engine=3,cold_wave=0,warm_waves=[1,2],input_tokens=[len(j['prompt_ids']) for j in read(R/'input1.json')['jobs']],output_limit=128,concurrency=4,scope='Native burst only; not HTTP TTFT or a newly matched vLLM/SGLang result',no_comparator_execution=True,source_hashing_during_collection=False))
checks[str(R/'plan1.json')]=sha(R/'plan1.json');save('preflight1.json',dict(passed=True,files_sha256=checks,collection_serial=True,no_comparator_execution=True))
for index,mode in enumerate(sequence):
 idle();label=f'engine{index+1}-{mode}';cmd=[str(W/'out/build/macos-arm64-release/bin/xlang3'),'-B',str(R/'worker1.py'),str(Q/'build-on1'),str(Q/'source1'),str(R/'input1.json'),str(R/(label+'-result1.json')),mode]
 save(label+'-command1.json',cmd);start=time.time_ns()
 with (R/(label+'-stdout1.log')).open('x') as o,(R/(label+'-stderr1.log')).open('x') as e:
  p=subprocess.Popen(cmd,stdout=o,stderr=e,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',GARNET_METAL_PROFILE_OPS='0'));save(label+'-pid1.json',dict(pid=p.pid,started_ns=start));print(label,p.pid,'running',flush=True);rc=p.wait()
 save(label+'-status1.json',dict(pid=p.pid,status='terminal',returncode=rc,started_ns=start,finished_ns=time.time_ns()));print(label,rc,'terminal',flush=True)
 assert rc==0 and read(R/(label+'-result1.json'))['completed'] and len(read(R/(label+'-result1.json'))['waves'])==3,(label,rc)
idle()
for n,h in checks.items():assert sha(Path(n))==h,n
save('postflight1.json',dict(passed=True,checked_files=len(checks),all_owned_processes_terminal=True,no_comparator_execution=True))
