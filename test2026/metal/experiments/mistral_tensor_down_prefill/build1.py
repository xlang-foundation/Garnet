"""Build default-off and opt-in profiles, testing strictly serial native suites."""
from pathlib import Path
import hashlib,json,os,subprocess,time
R=Path(__file__).resolve().parent;S=R/'source1';Q=R.parent/'mistral-causal-batch4-helper1'
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def files(root):return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
def save(n,v):
 p=R/n;assert not p.exists(),n;p.write_text(json.dumps(v,indent=2)+'\n')
base=read(Q/'source-bindings1.json')['files_sha256'];target=files(S);changed=sorted(n for n,h in target.items() if base.get(n)!=h)
assert changed==read(R/'preparation1.json')['changed_files'] and len(changed)==7
assert set(base)-set(target)==set() and set(target)-set(base)=={'src/backends/metal/packed_down_tensor_shader.h'}
save('source-bindings1.json',dict(passed=True,files_sha256=target,changed_files=changed,base_source_bindings_sha256=sha(Q/'source-bindings1.json')))
def run(name,cmd):
 save(name+'-command.json',cmd);started=time.time_ns()
 with (R/(name+'.stdout')).open('x') as out,(R/(name+'.stderr')).open('x') as err:
  p=subprocess.Popen(cmd,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));save(name+'-pid.json',dict(pid=p.pid,started_ns=started));print(name,p.pid,'running',flush=True);rc=p.wait()
 save(name+'-status.json',dict(pid=p.pid,returncode=rc,status='terminal',started_ns=started,finished_ns=time.time_ns()));print(name,rc,'terminal',flush=True)
 if rc:print((R/(name+'.stderr')).read_text()[-7000:],flush=True)
 assert rc==0,(name,rc)
base_cmd=read(Q/'configure1-command.json');libraries={}
for mode in ['on','default-off']:
 build=R/('build-'+mode+'1');cmd=[str(S) if arg==str(Q/'source1') else str(build) if arg==str(Q/'build-on1') else arg for arg in base_cmd]
 assert '-DGARNET_METAL_MISTRAL_DOWN_TENSOR512=ON' not in cmd
 if mode=='on':cmd.append('-DGARNET_METAL_MISTRAL_DOWN_TENSOR512=ON')
 run('configure-'+mode+'1',cmd);run('build-'+mode+'1',['/opt/homebrew/bin/cmake','--build',str(build),'--parallel','4']);run('suite-'+mode+'1',['/opt/homebrew/bin/ctest','--test-dir',str(build),'--output-on-failure','--parallel','1'])
 assert '100% tests passed out of 56' in (R/('suite-'+mode+'1.stdout')).read_text()
 cache=(build/'CMakeCache.txt').read_text();assert ('GARNET_METAL_MISTRAL_DOWN_TENSOR512:BOOL='+('ON' if mode=='on' else 'OFF')) in cache
 libraries[mode]=sha(build/'libgarnet.dylib')
assert files(S)==target
save('build-audit1.json',dict(passed=True,native_tests_per_build=56,builds=['on','default-off'],library_sha256=libraries,source_bindings_sha256=sha(R/'source-bindings1.json'),source_files=len(target),new_option_default_off=True,no_dirty_root_build=True,no_model_execution=True,no_comparator_execution=True,goal_complete=False))
