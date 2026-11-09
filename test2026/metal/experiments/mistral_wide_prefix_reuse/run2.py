"""Fresh serial control and retained-prefix arms; stop at the first state mismatch."""
from pathlib import Path
import ast,hashlib,json,os,subprocess,sys,time
R=Path(__file__).resolve().parent;M=R.parent;B=M/'mistral-wide-lifecycle1';W=M.parents[2]
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
sys.path.insert(0,str(M/'mistral-fused-q4-decode-native1/tools1'));from run_local_coder_baseline import engine_processes
assert not engine_processes(),engine_processes();assert read(M/'mistral-wide-lifecycle-checkpoint1.json')['passed'];assert read(R/'postflight1.json')['passed'];assert not (R/'preflight2.json').exists();ast.parse((R/'worker1.py').read_text())
checks=read(R/'preflight1.json')['files_sha256']
for p in [R/'worker1.py',R/'prepare1.py',R/'run2.py',R/'plan2.json',R/'input2.json',B/'build-on1/libgarnet.dylib']:checks[str(p)]=sha(p)
for n,h in checks.items():assert sha(Path(n))==h,n
save(R/'preflight2.json',dict(passed=True,files_sha256=checks,no_comparator_execution=True,performance_measured=False,source_reused_without_changes=str(B/'source1')))
sequence=read(R/'plan2.json')['arms'];save(R/'sequence2.json',sequence)
for mode in sequence:
 assert not engine_processes(),engine_processes()
 if mode!='ordinary-off':
  p=R/'ordinary-off-result1.json';assert read(p)['completed'];checks[str(p)]=sha(p);save(R/(mode+'-reference-binding2.json'),dict(path=str(p),sha256=sha(p),purpose='Fresh state/output control, not proposal oracle'))
 cmd=[str(W/'out/build/macos-arm64-release/bin/xlang3'),'-B',str(R/'worker1.py'),str(B/'build-on1'),str(B/'source1'),str(R/'input2.json'),str(R/(mode+'-result2.json')),mode];save(R/(mode+'-command2.json'),cmd)
 with (R/(mode+'-stdout2.log')).open('x') as out,(R/(mode+'-stderr2.log')).open('x') as err:
  p=subprocess.Popen(cmd,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',GARNET_METAL_PROFILE_OPS='0'));state=dict(status='running',pid=p.pid,started_ns=time.time_ns());save(R/(mode+'-status2.json'),state);print(json.dumps(dict(step=mode,**state)),flush=True);code=p.wait();state.update(status='terminal',returncode=code,finished_ns=time.time_ns());save(R/(mode+'-status2.json'),state)
 print(json.dumps(dict(step=mode,**state)),flush=True)
 if code!=0:
  save(R/'stopped2.json',dict(mode=mode,returncode=code,reason='Worker failed; preserve raw counterexample and stop later engines.',no_comparator_execution=True,goal_complete=False));raise SystemExit(code)
 assert read(R/(mode+'-result2.json'))['completed']
assert not engine_processes(),engine_processes()
for n,h in checks.items():assert sha(Path(n))==h,n
save(R/'postflight2.json',dict(passed=True,checked_files=len(checks),all_owned_engines_terminal=True,no_comparator_execution=True,performance_measured=False))
print('Prefix lineage arms terminal',flush=True)
