"""Fresh B1 causal draft and C4 lifecycle qualification, strictly serial."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
R=Path(__file__).resolve().parent;M=R.parent;W=Path('/Users/shawn/CantorAI');B=M/'mistral-bounded-draft-long1'
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(name,v):
 p=R/name;assert not p.exists(),name;p.write_text(json.dumps(v,indent=2)+'\n')
sys.path.insert(0,str(M/'mistral-fused-q4-decode-native1/tools1'));from run_local_coder_baseline import engine_processes
assert not engine_processes(),engine_processes();assert read(R/'build-audit1.json')['passed']
plan=dict(source=str(R/'source1'),libraries=dict(on=str(R/'build-on1'),off=str(R/'build-default-off1')),b1_input=str(B/'input1.json'),c4_input=str(R/'input2.json'),sequence=[['b1-on','b1_worker1.py','on',str(B/'input1.json'),'qualify'],['b1-off','b1_worker1.py','off',str(B/'input1.json'),'qualify']]+[[f'c4-{mode}','c4_worker1.py','on',str(R/'input2.json'),mode] for mode in ['ordinary','verify','reverse']],scope='Correctness qualification only; B1 uses actual causal helper; C4 zero proposals disclosed; no serving claim',no_comparator_execution=True,goal_complete=False)
save('qualification-plan1.json',plan)
checks=read(B/'qualification1-preflight.json')['files_sha256'].copy();checks.update(read(R/'build-preflight1.json')['files_sha256'])
for mode,lib in plan['libraries'].items():checks[lib+'/libgarnet.dylib']=read(R/'build-audit1.json')['library_sha256'][mode]
for name in ['run_qualification1.py','prepare_workers1.py','workers-bindings1.json','b1_worker1.py','c4_worker1.py','audit_owner1.py','audit_qualification1.py','input2.json','qualification-plan1.json','build-audit1.json']:checks[str(R/name)]=sha(R/name)
for p in [M/'mistral-wide-lifecycle1/ordinary-result2.json',M/'mistral-wide-lifecycle1/audit2.json',B/'input1.json',B/'qualification1-result.json',B/'audit_qualification1.py',B/'audit_bounded_common1.py']:checks[str(p)]=sha(p)
for name,h in checks.items():assert sha(Path(name))==h,name
save('qualification1-preflight.json',dict(passed=True,files_sha256=checks,collection_serial=True,hashes_during_collection=False,no_comparator_execution=True))
for label,worker,mode,fixture,experiment in plan['sequence']:
 assert not engine_processes(),engine_processes()
 cmd=[str(W/'out/build/macos-arm64-release/bin/xlang3'),'-B',str(R/worker),plan['libraries'][mode],plan['source'],fixture,str(R/(label+'-result1.json')),experiment]
 save(label+'-command1.json',cmd);start=time.time_ns()
 with (R/(label+'-stdout1.log')).open('x') as out,(R/(label+'-stderr1.log')).open('x') as err:
  p=subprocess.Popen(cmd,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',GARNET_METAL_PROFILE_OPS='0'));save(label+'-pid1.json',dict(pid=p.pid,started_ns=start));print(json.dumps(dict(step=label,pid=p.pid,status='running')),flush=True);rc=p.wait()
 save(label+'-status1.json',dict(status='terminal',returncode=rc,pid=p.pid,started_ns=start,finished_ns=time.time_ns()));print(json.dumps(dict(step=label,status='terminal',returncode=rc)),flush=True)
 assert rc==0 and read(R/(label+'-result1.json'))['completed'],(label,rc)
assert not engine_processes(),engine_processes()
for name,h in checks.items():assert sha(Path(name))==h,name
save('qualification1-postflight.json',dict(passed=True,checked_files=len(checks),all_owned_processes_terminal=True,no_comparator_execution=True))
print('All five full-model qualification engines terminal.',flush=True)
