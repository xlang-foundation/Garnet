"""Build shared-value ON/default-OFF runtimes and run all native tests serially."""
from pathlib import Path
import hashlib,importlib.util,json,os,subprocess,time
R=Path(__file__).resolve().parent;S=R/'source1';N=R.parent/'mistral-gqa16-score-native1';W=Path('/Users/shawn/CantorAI');G=W/'Garnet'
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(name,value):
 p=R/name;assert not p.exists(),name;p.write_text(json.dumps(value,indent=2)+'\n')
module=R.parent/'mistral-fused-q4-decode-native1/tools1/run_local_coder_baseline.py'
spec=importlib.util.spec_from_file_location('baseline',module);b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
def idle():
 assert not b.engine_processes(),b.engine_processes()
 lines=subprocess.check_output(['ps','-axww','-o','pid=,comm=,args='],text=True).splitlines()
 extra=[]
 for line in lines:
  f=line.strip().split(None,2)
  if len(f)==3 and Path(f[1]).name=='xlang3' and '/WorldSense/' not in f[1]+' '+f[2]:extra.append(line)
 assert not extra,extra
 return True
idle()
files={str(p.relative_to(S)):sha(p) for p in sorted(S.rglob('*')) if p.is_file()}
base=read(R/'upstream-source-bindings1.json')['files_sha256'];changed=[n for n,h in files.items() if base.get(n)!=h]
assert set(base)<=set(files) and len(changed)==10,changed
save('source-bindings1.json',dict(base_native=str(N/'source1'),upstream_source_bindings_sha256=sha(R/'upstream-source-bindings1.json'),files_sha256=files,changed_files=changed))
sync=W/'out/research/garnet-main-sync-20261108-1'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=G,text=True).strip()==read(sync/'complete1.json')['merge_commit']
dirty={n:sha(G/n) for n in read(sync/'before.json')['original_files']}
checks={str(S/n):h for n,h in files.items()};checks.update({str(G/n):h for n,h in dirty.items()})
for p in [R/'prepare1.py',R/'prepare_test1.py',R/'build1.py',R/'source-bindings1.json',R/'upstream1.json',R/'upstream-main1.patch']:checks[str(p)]=sha(p)
save('build-preflight1.json',dict(passed=True,files_sha256=checks,original_local_changes_preserved_after_main_merge=True,no_dirty_root_build=True,no_comparator_execution=True))
def run(name,argv,want=0):
 idle();save(name+'-command.json',argv);start=time.time_ns()
 with (R/(name+'.stdout')).open('x') as out,(R/(name+'.stderr')).open('x') as err:
  p=subprocess.Popen(argv,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));save(name+'-pid.json',dict(pid=p.pid,started_ns=start,argv=argv));print(json.dumps(dict(step=name,pid=p.pid,status='running')),flush=True);rc=p.wait()
 save(name+'-status.json',dict(status='terminal',returncode=rc,pid=p.pid,started_ns=start,finished_ns=time.time_ns()));print(json.dumps(dict(step=name,returncode=rc,status='terminal')),flush=True);assert rc==want,(name,rc,want)
base=[str(S) if a==str(N/'source1') else str(R/'build-on1') if a==str(N/'build-on1') else a for a in read(N/'configure-on1-command.json')]
on=base+['-DGARNET_METAL_MISTRAL_VERIFY_GQA16_VALUES=ON']
invalid=[str(R/'invalid-layout-off1') if a==str(R/'build-on1') else '-DGARNET_METAL_MISTRAL_VERIFY_LAYOUT=OFF' if a=='-DGARNET_METAL_MISTRAL_VERIFY_LAYOUT=ON' else '-DGARNET_METAL_MISTRAL_VERIFY_SPLIT_SCORES=OFF' if a=='-DGARNET_METAL_MISTRAL_VERIFY_SPLIT_SCORES=ON' else '-DGARNET_METAL_MISTRAL_VERIFY_GQA16_SCORES=OFF' if a=='-DGARNET_METAL_MISTRAL_VERIFY_GQA16_SCORES=ON' else a for a in on]
run('invalid-layout-off1',invalid,1);assert 'GQA16 verification values require' in (R/'invalid-layout-off1.stderr').read_text()
run('configure-on1',on);run('build-on1',['/opt/homebrew/bin/cmake','--build',str(R/'build-on1'),'--parallel','4'])
run('suite-on1',['/opt/homebrew/bin/ctest','--test-dir',str(R/'build-on1'),'--output-on-failure','--parallel','1']);assert '100% tests passed' in (R/'suite-on1.stdout').read_text()
off=[str(R/'build-default-off1') if a==str(R/'build-on1') else a for a in base]
run('configure-default-off1',off);run('build-default-off1',['/opt/homebrew/bin/cmake','--build',str(R/'build-default-off1'),'--parallel','4'])
assert 'GARNET_METAL_MISTRAL_VERIFY_GQA16_VALUES:BOOL=OFF' in (R/'build-default-off1/CMakeCache.txt').read_text()
run('suite-default-off1',['/opt/homebrew/bin/ctest','--test-dir',str(R/'build-default-off1'),'--output-on-failure','--parallel','1']);assert '100% tests passed' in (R/'suite-default-off1.stdout').read_text()
for n,h in checks.items():assert sha(Path(n))==h,n
save('build-audit1.json',dict(passed=True,library_sha256={mode:sha(R/build/'libgarnet.dylib') for mode,build in [('on','build-on1'),('off','build-default-off1')]},source_bindings_sha256=sha(R/'source-bindings1.json'),default_option_on=False,local_changes_preserved=True,all_owned_processes_terminal=True,no_comparator_execution=True,model_measured=False,serving_speed_claim=False,goal_complete=False))
print('Native shared-value ON/default-OFF suites passed.',flush=True)
