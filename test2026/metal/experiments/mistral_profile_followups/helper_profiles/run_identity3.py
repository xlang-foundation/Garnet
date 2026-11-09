"""One untimed fresh native correctness qualification of the corrected64-token cache identity."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
R=Path(__file__).resolve().parent;M=R.parent;W=Path('/Users/shawn/CantorAI');read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(n,d):
 p=R/n;assert not p.exists();p.write_text(json.dumps(d,indent=2)+'\n')
sys.path.insert(0,str(M/'mistral-fused-q4-decode-native1/tools1'));from run_local_coder_baseline import engine_processes
assert not engine_processes();checks=read(R/'qualification1-preflight.json')['files_sha256'].copy()
for n in ['run_identity3.py','b1_worker2.py','prepare_identity3.py','identity-preparation3.json','identity-change3.patch','qualification-audit2.json','cap64-result1.json']:checks[str(R/n)]=sha(R/n)
for n,h in checks.items():assert sha(Path(n))==h,n
save('identity3-preflight.json',dict(passed=True,files_sha256=checks,performance_admitted=False,no_comparator_execution=True));plan=read(R/'plan1.json');cmd=[str(W/'out/build/macos-arm64-release/bin/xlang3'),'-B',str(R/'b1_worker2.py'),plan['library'],plan['source'],str(R/'input1.json'),str(R/'identity64-result3.json'),'qualify','64'];save('identity64-command3.json',cmd);start=time.time_ns()
with (R/'identity64-stdout3.log').open('x') as o,(R/'identity64-stderr3.log').open('x') as e:
 p=subprocess.Popen(cmd,stdout=o,stderr=e,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',GARNET_METAL_PROFILE_OPS='0'));save('identity64-pid3.json',dict(pid=p.pid,started_ns=start));print('identity64',p.pid,'running',flush=True);rc=p.wait()
save('identity64-status3.json',dict(status='terminal',returncode=rc,pid=p.pid,started_ns=start,finished_ns=time.time_ns()));print('identity64',rc,'terminal',flush=True)
if rc:print((R/'identity64-stderr3.log').read_text()[-3000:],flush=True)
assert rc==0 and read(R/'identity64-result3.json')['completed'] and not engine_processes()
for n,h in checks.items():assert sha(Path(n))==h,n
save('identity3-postflight.json',dict(passed=True,all_owned_processes_terminal=True,no_comparator_execution=True))
