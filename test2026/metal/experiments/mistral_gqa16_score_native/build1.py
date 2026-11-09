"""Build GQA16 ON/defaultOFF isolated runtimes and run native suites serially."""
from pathlib import Path
import hashlib,importlib.util,json,os,subprocess,time
R=Path(__file__).resolve().parent;M=R.parent;B=M/'mistral-wide-lifecycle1';S=R/'source1';W=Path('/Users/shawn/CantorAI');G=W/'Garnet'
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(name,v):
 p=R/name;assert not p.exists(),name;p.write_text(json.dumps(v,indent=2)+'\n')
base=M/'mistral-fused-q4-decode-native1/tools1/run_local_coder_baseline.py';spec=importlib.util.spec_from_file_location('baseline',base);b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
assert not b.engine_processes(),b.engine_processes()
files={str(p.relative_to(S)):sha(p) for p in sorted(S.rglob('*')) if p.is_file()};previous=read(B/'source-bindings1.json')['files_sha256'];changed=[n for n,h in files.items() if previous.get(n)!=h]
assert set(previous)<=set(files)
assert all(n.startswith(('cmake/','src/','test2026/metal/')) for n in changed)
save('source-bindings1.json',dict(base_source=str(B/'source1'),files_sha256=files,changed_files=changed))
checks={str(S/n):h for n,h in files.items()}
for n,h in read(M/'mistral-half-attention-native1/source-audit7.json')['original_dirty_sha256'].items():checks[str(G/n)]=h
for p in [R/'prepare1.py',R/'prepare_test1.py',R/'build1.py',R/'source-bindings1.json']:checks[str(p)]=sha(p)
for n,h in checks.items():assert sha(Path(n))==h,n
save('build-preflight1.json',dict(passed=True,files_sha256=checks,original_dirty_unchanged=True,no_dirty_root_build=True,no_comparator_execution=True))
def run(name,argv,want=0):
 save(name+'-command.json',argv);start=time.time_ns()
 with (R/(name+'.stdout')).open('x') as out,(R/(name+'.stderr')).open('x') as err:
  p=subprocess.Popen(argv,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));save(name+'-pid.json',dict(pid=p.pid,started_ns=start,argv=argv));print(json.dumps(dict(step=name,pid=p.pid,status='running')),flush=True);rc=p.wait()
 save(name+'-status.json',dict(status='terminal',returncode=rc,pid=p.pid,started_ns=start,finished_ns=time.time_ns()));print(json.dumps(dict(step=name,returncode=rc,status='terminal')),flush=True);assert rc==want,(name,rc,want)
base=[str(S) if arg==str(B/'source1') else str(R/'build-on1') if arg==str(B/'build-on1') else arg for arg in read(B/'configure-on1-command.json')]
on=base+['-DGARNET_METAL_MISTRAL_VERIFY_GQA16_SCORES=ON']
invalid=[str(R/'invalid-layout-off1') if arg==str(R/'build-on1') else '-DGARNET_METAL_MISTRAL_VERIFY_LAYOUT=OFF' if arg=='-DGARNET_METAL_MISTRAL_VERIFY_LAYOUT=ON' else '-DGARNET_METAL_MISTRAL_VERIFY_SPLIT_SCORES=OFF' if arg=='-DGARNET_METAL_MISTRAL_VERIFY_SPLIT_SCORES=ON' else arg for arg in on]
run('invalid-layout-off1',invalid,1);assert 'GQA16 verification scores require' in (R/'invalid-layout-off1.stderr').read_text()
run('configure-on1',on);run('build-on1',['/opt/homebrew/bin/cmake','--build',str(R/'build-on1'),'--parallel','4'])
assert not b.engine_processes(),b.engine_processes();run('suite-on1',['/opt/homebrew/bin/ctest','--test-dir',str(R/'build-on1'),'--output-on-failure','--parallel','1']);assert '100% tests passed' in (R/'suite-on1.stdout').read_text()
off=[str(R/'build-default-off1') if arg==str(R/'build-on1') else arg for arg in base]
run('configure-default-off1',off);run('build-default-off1',['/opt/homebrew/bin/cmake','--build',str(R/'build-default-off1'),'--parallel','4'])
assert 'GARNET_METAL_MISTRAL_VERIFY_GQA16_SCORES:BOOL=OFF' in (R/'build-default-off1/CMakeCache.txt').read_text()
assert not b.engine_processes(),b.engine_processes();run('suite-default-off1',['/opt/homebrew/bin/ctest','--test-dir',str(R/'build-default-off1'),'--output-on-failure','--parallel','1']);assert '100% tests passed' in (R/'suite-default-off1.stdout').read_text()
for n,h in checks.items():assert sha(Path(n))==h,n
save('build-audit1.json',dict(passed=True,library_sha256={mode:sha(R/build/'libgarnet.dylib') for mode,build in [('on','build-on1'),('off','build-default-off1')]},source_bindings_sha256=sha(R/'source-bindings1.json'),default_option_on=False,original_dirty_unchanged=True,all_owned_processes_terminal=True,no_comparator_execution=True,model_measured=False,serving_speed_claim=False,goal_complete=False))
print('Native GQA16 ON/defaultOFF suites passed.',flush=True)
