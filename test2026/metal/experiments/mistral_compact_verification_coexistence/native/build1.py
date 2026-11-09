"""Build prepared compact/verification coexistence ON/default-OFF with serial suites."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
R=Path(__file__).resolve().parent;M=R.parent;N=M/'mistral-scheduled-dense-verification1';V=M/'mistral-scheduled-c4-ownership2';S=R/'source1';W=M.parents[2]
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,d):
 t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
sys.path.insert(0,str(M/'mistral-fused-q4-decode-native1/tools1'));from run_local_coder_baseline import engine_processes
assert not engine_processes(),engine_processes()
assert not (R/'build-preflight1.json').exists()
assert read(V/'audit1.json')['passed']
bindings=read(R/'source-bindings1.json');checks={str(S/n):h for n,h in bindings['files_sha256'].items()}
for n,h in read(M/'mistral-half-attention-native1/source-audit7.json')['original_dirty_sha256'].items():checks[str(W/'Garnet'/n)]=h
for p in [R/'build1.py',R/'prepare1.py',R/'source-bindings1.json',V/'audit1.py',V/'audit1.json']:checks[str(p)]=sha(p)
for n,h in checks.items():assert sha(Path(n))==h,n
save(R/'build-preflight1.json',dict(passed=True,files_sha256=checks,no_comparator_execution=True,no_dirty_root_build=True))
def run(name,cmd,code=0):
 assert not (R/(name+'-status.json')).exists();save(R/(name+'-command.json'),cmd)
 with (R/(name+'.stdout')).open('x') as out,(R/(name+'.stderr')).open('x') as err:
  p=subprocess.Popen(cmd,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));state=dict(status='running',pid=p.pid,started_ns=time.time_ns());save(R/(name+'-status.json'),state);print(json.dumps(dict(step=name,**state)),flush=True);ret=p.wait();state.update(status='terminal',returncode=ret,finished_ns=time.time_ns());save(R/(name+'-status.json'),state)
 print(json.dumps(dict(step=name,**state)),flush=True);assert ret==code,(name,ret,code)
base=[str(S) if v==str(N/'source1') else str(R/'build-on1') if v==str(N/'build-on1') else v for v in read(N/'configure-on1-command.json')]
on=base+['-DGARNET_METAL_COMPACT_VERIFY_COEXISTENCE=ON']
invalid=[str(R/'invalid-layout-off1') if v==str(R/'build-on1') else '-DGARNET_METAL_DENSE_VERIFY_SCHEDULER=OFF' if v=='-DGARNET_METAL_DENSE_VERIFY_SCHEDULER=ON' else '-DGARNET_METAL_VERIFY_ATTENTION=OFF' if v=='-DGARNET_METAL_VERIFY_ATTENTION=ON' else v for v in on]
run('invalid-layout-off1',invalid,1);assert 'Compact verification coexistence requires' in (R/'invalid-layout-off1.stderr').read_text()
run('configure-on1',on);run('build-on1',['/opt/homebrew/bin/cmake','--build',str(R/'build-on1'),'--parallel','4'])
assert not engine_processes(),engine_processes();run('suite-on1',['/opt/homebrew/bin/ctest','--test-dir',str(R/'build-on1'),'--output-on-failure','--parallel','1'])
assert '100% tests passed' in (R/'suite-on1.stdout').read_text()
off=[str(R/'build-default-off1') if v==str(R/'build-on1') else v for v in base]
run('configure-default-off1',off);run('build-default-off1',['/opt/homebrew/bin/cmake','--build',str(R/'build-default-off1'),'--parallel','4'])
assert not engine_processes(),engine_processes();run('suite-default-off1',['/opt/homebrew/bin/ctest','--test-dir',str(R/'build-default-off1'),'--output-on-failure','--parallel','1'])
assert '100% tests passed' in (R/'suite-default-off1.stdout').read_text()
for n,h in checks.items():assert sha(Path(n))==h,n
assert not engine_processes(),engine_processes()
libs={mode:sha(R/build/'libgarnet.dylib') for mode,build in [('on','build-on1'),('off','build-default-off1')]}
save(R/'build-audit1.json',dict(passed=True,library_sha256=libs,source_bindings_sha256=sha(R/'source-bindings1.json'),all_owned_processes_terminal=True,default_option_on=False,no_comparator_execution=True,model_measured=False,serving_speed_claim=False,goal_complete=False));print('Native ON/OFF builds and serial suites passed',flush=True)
