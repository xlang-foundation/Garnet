"""Run two fresh isolated score screens serially; no model/comparator inference."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
R=Path(__file__).resolve().parent;M=R.parent;N=M/'mistral-fused-q4-decode-native1';W=M.parents[2]
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,d):
 t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
sys.path.insert(0,str(N/'tools1'));from run_local_coder_baseline import engine_processes
assert not engine_processes(),engine_processes()
assert not (R/'preflight1.json').exists()
assert read(R/'compile2-status.json')['returncode']==0
checks={}
def bind(p,h=None):
 v=sha(p)
 if h is not None:assert v==h,str(p)
 checks[str(p)]=v
for n,h in read(M/'mistral-half-attention-native1/source-audit7.json')['original_dirty_sha256'].items():bind(W/'Garnet'/n,h)
for n,h in read(R/'screen-plan1.json')['files_sha256'].items():bind(R/n,h)
bind(R/'screen2.mm',read(R/'screen-plan2.json')['host_sha256'])
p=read(R/'plan1.json');bind(Path(p['base_header']),p['base_header_sha256']);bind(R/'control1.metal',p['control_msl_sha256'])
for n in ['run1.py','screen2','compile2-command.json','compile2-status.json','screen-plan1.json','screen-plan2.json']:bind(R/n)
save(R/'preflight1.json',dict(passed=True,files_sha256=checks,collection_serial=True,no_comparator_execution=True,no_model_execution=True,hashes_during_collection=False))
for cohort in [1,2]:
 assert not engine_processes(),engine_processes()
 command=[str(R/'screen2'),str(R),str(R/('cohort'+str(cohort))),str(cohort)];save(R/('cohort'+str(cohort)+'-command1.json'),command)
 with (R/('cohort'+str(cohort)+'-stdout1.log')).open('x') as out,(R/('cohort'+str(cohort)+'-stderr1.log')).open('x') as err:
  p=subprocess.Popen(command,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
  state=dict(status='running',pid=p.pid,started_ns=time.time_ns());save(R/('cohort'+str(cohort)+'-status1.json'),state);print(json.dumps(dict(cohort=cohort,**state)),flush=True)
  code=p.wait();state.update(status='terminal',returncode=code,finished_ns=time.time_ns());save(R/('cohort'+str(cohort)+'-status1.json'),state)
 print(json.dumps(state),flush=True)
 assert code==0 and read(R/('cohort'+str(cohort)+'/result1.json'))['completed'],(cohort,code)
assert not engine_processes(),engine_processes()
for n,h in checks.items():assert sha(Path(n))==h,n
save(R/'postflight1.json',dict(passed=True,all_owned_processes_terminal=True,checked_files=len(checks),no_comparator_execution=True,no_model_execution=True));print('Both isolated screens terminal',flush=True)
