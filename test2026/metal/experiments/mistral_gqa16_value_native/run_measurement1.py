"""Four fresh real-helper stages, OFF/ON then ON/OFF; no comparator reruns."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
R=Path(__file__).resolve().parent;M=R.parent;W=Path('/Users/shawn/CantorAI')
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(name,v):
 p=R/name;assert not p.exists(),name;p.write_text(json.dumps(v,indent=2)+'\n')
sys.path.insert(0,str(M/'mistral-fused-q4-decode-native1/tools1'));from run_local_coder_baseline import engine_processes
def idle():
 found=engine_processes()
 commands=subprocess.check_output(['ps','-axo','pid=,comm='],text=True)
 args=subprocess.check_output(['ps','-axww','-o','pid=,args='],text=True)
 argv={line.strip().split(None,1)[0]:line.strip().split(None,1)[1] for line in args.splitlines() if len(line.strip().split(None,1))==2}
 for line in commands.splitlines():
  fields=line.strip().split(None,1)
  if len(fields)==2 and Path(fields[1]).name=='xlang3' and '/WorldSense/' not in fields[1]+' '+argv.get(fields[0],''):found.append(line.strip())
 assert not found,found
idle();assert read(R/'qualification-audit1.json')['passed']
plan=read(R/'qualification-plan1.json');sequence=[[1,'off'],[1,'on'],[2,'on'],[2,'off']]
save('measurement-plan1.json',dict(sequence=sequence,profile=dict(batch=1,capacity=4096,target_prompt_tokens=2048,output_tokens=128,verification_depth=4,helper_context=512,helper_user_tail=192),minimum_native_decode_ratio=1.05,proposals_and_full_final_state_must_repeat=True,no_hash_build_during_collection=True,no_comparator_execution=True,serving_ttft_measured=False,goal_complete=False))
checks=read(R/'qualification1-preflight.json')['files_sha256'].copy()
for name in ['run_measurement1.py','audit_measurement1.py','measurement-plan1.json','qualification-audit1.json','qualification1-postflight.json']+[label+'-result1.json' for label,*rest in plan['sequence']]:checks[str(R/name)]=sha(R/name)
for name,h in checks.items():assert sha(Path(name))==h,name
save('measurement1-preflight.json',dict(passed=True,files_sha256=checks,collection_serial=True,hashes_during_collection=False,no_comparator_execution=True))
for cohort,mode in sequence:
 idle();label=f'cohort{cohort}-{mode}'
 cmd=[str(W/'out/build/macos-arm64-release/bin/xlang3'),'-B',str(R/'b1_worker1.py'),plan['libraries'][mode],plan['source'],plan['b1_input'],str(R/(label+'-result1.json')),'bounded'];save(label+'-command1.json',cmd);start=time.time_ns()
 with (R/(label+'-stdout1.log')).open('x') as out,(R/(label+'-stderr1.log')).open('x') as err:
  p=subprocess.Popen(cmd,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',GARNET_METAL_PROFILE_OPS='0'));save(label+'-pid1.json',dict(pid=p.pid,started_ns=start));print(json.dumps(dict(step=label,pid=p.pid,status='running')),flush=True);rc=p.wait()
 save(label+'-status1.json',dict(status='terminal',returncode=rc,pid=p.pid,started_ns=start,finished_ns=time.time_ns()));print(json.dumps(dict(step=label,status='terminal',returncode=rc)),flush=True)
 assert rc==0 and read(R/(label+'-result1.json'))['completed'],(label,rc)
idle()
for name,h in checks.items():assert sha(Path(name))==h,name
save('measurement1-postflight.json',dict(passed=True,checked_files=len(checks),all_owned_processes_terminal=True,no_comparator_execution=True))
print('Four fresh ABBA real-helper stages terminal.',flush=True)
