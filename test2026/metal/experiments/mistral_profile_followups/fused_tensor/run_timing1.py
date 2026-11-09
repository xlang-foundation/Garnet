"""Compile and collect one serial learned whole-MLP correctness pilot."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
R=Path(__file__).resolve().parent;M=R.parent;P=M/'mistral-tensor-view-prefill-screen1';Q=M/'mistral-causal-batch4-helper1';T=M/'mistral-fused-mlp-trained1'
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(n,d):
 p=R/n;assert not p.exists();p.write_text(json.dumps(d,indent=2)+'\n')
def run(name,cmd):
 save(name+'-command.json',cmd);start=time.time_ns()
 with (R/(name+'.stdout')).open('x') as o,(R/(name+'.stderr')).open('x') as e:
  p=subprocess.Popen(cmd,stdout=o,stderr=e,start_new_session=True);save(name+'-pid.json',dict(pid=p.pid,started_ns=start));print(name,p.pid,'running',flush=True);rc=p.wait()
 save(name+'-status.json',dict(status='terminal',returncode=rc,pid=p.pid,started_ns=start,finished_ns=time.time_ns()));print(name,rc,'terminal',flush=True)
 if rc:print((R/(name+'.stderr')).read_text()[:5000],flush=True)
 assert rc==0,name
cmd=read(R/'host-compile1-command.json');cmd=[str(R/'timing1.mm') if a==str(R/'pilot1.mm') else str(R/'timing1') if a==str(R/'pilot1') else a for a in cmd];run('timing-compile1',cmd)
sys.path.insert(0,str(M/'mistral-fused-q4-decode-native1/tools1'));from run_local_coder_baseline import engine_processes
assert not engine_processes(),engine_processes();assert read(R/'pilot-audit1.json')['passed'];checks=read(R/'pilot1-preflight.json')['files_sha256']
for n in ['run_timing1.py','prepare_timing1.py','timing1.mm','timing1','timing-plan1.json','timing-compile1-command.json','timing-compile1-status.json','pilot-audit1.json','audit_pilot1.py']:checks[str(R/n)]=sha(R/n)
for n,h in checks.items():assert sha(Path(n))==h,n
save('timing1-preflight.json',dict(passed=True,files_sha256=checks,collection_serial=True,model_execution=False,no_comparator_execution=True,source_or_fixture_hashes_during_timed_rounds=False))
for order in ['forward','reverse']:
 assert not engine_processes(),engine_processes()
 run('timing-'+order+'1',[str(R/'timing1'),str(R/'combined1.metal'),str(T/'ordinary1.metal'),str(T/'assets1'),str(T/'captures1'),str(R/('timing-'+order+'1')),order]);d=read(R/('timing-'+order+'1/result1.json'));assert d['completed'] and d['candidate_exact']
assert not engine_processes()
for n,h in checks.items():assert sha(Path(n))==h,n
save('timing1-postflight.json',dict(passed=True,checked_files=len(checks),all_owned_processes_terminal=True,no_comparator_execution=True))
