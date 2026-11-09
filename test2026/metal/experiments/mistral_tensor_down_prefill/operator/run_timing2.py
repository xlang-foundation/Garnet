"""Two serial operator cohorts with balanced seven-method ordering."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
R=Path(__file__).resolve().parent;M=R.parent
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(n,v):
 p=R/n;assert not p.exists(),n;p.write_text(json.dumps(v,indent=2)+'\n')
sys.path.insert(0,str(M/'mistral-fused-q4-decode-native1/tools1'));from run_local_coder_baseline import engine_processes
assert not engine_processes(),engine_processes();assert read(R/'pilot4-audit.json')['passed'] and read(R/'timing-compile2-status.json')['returncode']==0
checks=read(R/'pilot4-preflight.json')['files_sha256']
for name in ['run_timing2.py','timing2.mm','timing2','prepare_timing2.py','timing-preparation2.json','timing-instrumentation2.patch','pilot4-audit.json','audit_pilot4.py','timing-compile2-command.json','timing-compile2-status.json']:checks[str(R/name)]=sha(R/name)
assert read(R/'walk-source-audit1.json')['passed']
for name in ['walk-source-audit1.json','audit_walk_source1.py','walk4.metal','shared3.metal']:checks[str(R/name)]=sha(R/name)
for n,h in checks.items():assert sha(Path(n))==h,n
save('timing2-preflight.json',dict(passed=True,files_sha256=checks,collection_serial=True,model_execution=False,no_comparator_execution=True,source_or_fixture_hashes_during_timed_rounds=False))
plan=read(R/'plan4.json')
for order in ['forward','reverse']:
 assert not engine_processes(),engine_processes();label='timing-'+order+'2';cmd=[str(R/'timing2'),str(R/'combined4.metal'),plan['trained_assets'],plan['trained_captures'],str(R/label),order];save(label+'-command.json',cmd);started=time.time_ns()
 with (R/(label+'.stdout')).open('x') as out,(R/(label+'.stderr')).open('x') as err:
  p=subprocess.Popen(cmd,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ));save(label+'-pid.json',dict(pid=p.pid,started_ns=started));print(label,p.pid,'running',flush=True);rc=p.wait()
 save(label+'-status.json',dict(status='terminal',returncode=rc,pid=p.pid,started_ns=started,finished_ns=time.time_ns()));print(label,rc,'terminal',flush=True)
 if rc:print((R/(label+'.stderr')).read_text()[:6000])
 assert rc==0 and read(R/label/'result1.json')['completed']
assert not engine_processes(),engine_processes()
for n,h in checks.items():assert sha(Path(n))==h,n
save('timing2-postflight.json',dict(passed=True,checked_files=len(checks),all_owned_processes_terminal=True,no_comparator_execution=True))
