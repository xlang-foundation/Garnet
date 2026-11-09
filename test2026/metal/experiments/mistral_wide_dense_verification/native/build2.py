"""Build prepared compact/verification coexistence ON/default-OFF with serial suites."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
R=Path(__file__).resolve().parent;M=R.parent;N=M/'mistral-compact-verification-coexist1';V=M/'mistral-scheduled-c4-ownership2';S=R/'source2';W=M.parents[2]
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
assert not (R/'build-preflight2.json').exists()
assert read(V/'audit1.json')['passed']
bindings=read(R/'source-bindings2.json');checks={str(S/n):h for n,h in bindings['files_sha256'].items()}
for n,h in read(M/'mistral-half-attention-native1/source-audit7.json')['original_dirty_sha256'].items():checks[str(W/'Garnet'/n)]=h
for p in [R/'build2.py',R/'prepare1.py',R/'source-bindings2.json',V/'audit1.py',V/'audit1.json']:checks[str(p)]=sha(p)
for n,h in checks.items():assert sha(Path(n))==h,n
save(R/'build-preflight2.json',dict(passed=True,files_sha256=checks,no_comparator_execution=True,no_dirty_root_build=True))
def run(name,cmd,code=0):
 assert not (R/(name+'-status.json')).exists();save(R/(name+'-command.json'),cmd)
 with (R/(name+'.stdout')).open('x') as out,(R/(name+'.stderr')).open('x') as err:
  p=subprocess.Popen(cmd,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));state=dict(status='running',pid=p.pid,started_ns=time.time_ns());save(R/(name+'-status.json'),state);print(json.dumps(dict(step=name,**state)),flush=True);ret=p.wait();state.update(status='terminal',returncode=ret,finished_ns=time.time_ns());save(R/(name+'-status.json'),state)
 print(json.dumps(dict(step=name,**state)),flush=True);assert ret==code,(name,ret,code)
base=[str(S) if v==str(N/'source1') else str(R/'build-on2') if v==str(N/'build-on1') else v for v in read(N/'configure-on1-command.json')]
on=base+['-DGARNET_METAL_WIDE_DENSE_VERIFICATION=ON']
invalid=[str(R/'invalid-layout-off2') if v==str(R/'build-on2') else '-DGARNET_METAL_DENSE_VERIFY_SCHEDULER=OFF' if v=='-DGARNET_METAL_DENSE_VERIFY_SCHEDULER=ON' else v for v in on]
run('invalid-layout-off2',invalid,1);assert 'Wide dense verification requires' in (R/'invalid-layout-off2.stderr').read_text()
run('configure-on2',on);run('build-on2',['/opt/homebrew/bin/cmake','--build',str(R/'build-on2'),'--parallel','4'])
assert not engine_processes(),engine_processes();run('suite-on2',['/opt/homebrew/bin/ctest','--test-dir',str(R/'build-on2'),'--output-on-failure','--parallel','1'])
assert '100% tests passed' in (R/'suite-on2.stdout').read_text()
off=[str(R/'build-default-off2') if v==str(R/'build-on2') else v for v in base]
run('configure-default-off2',off);run('build-default-off2',['/opt/homebrew/bin/cmake','--build',str(R/'build-default-off2'),'--parallel','4'])
assert not engine_processes(),engine_processes();run('suite-default-off2',['/opt/homebrew/bin/ctest','--test-dir',str(R/'build-default-off2'),'--output-on-failure','--parallel','1'])
assert '100% tests passed' in (R/'suite-default-off2.stdout').read_text()
for n,h in checks.items():assert sha(Path(n))==h,n
assert not engine_processes(),engine_processes()
libs={mode:sha(R/build/'libgarnet.dylib') for mode,build in [('on','build-on2'),('off','build-default-off2')]}
save(R/'build-audit2.json',dict(passed=True,library_sha256=libs,source_bindings_sha256=sha(R/'source-bindings2.json'),all_owned_processes_terminal=True,default_option_on=False,no_comparator_execution=True,model_measured=False,serving_speed_claim=False,goal_complete=False));print('Wide native ON/OFF builds and serial suites passed',flush=True)
