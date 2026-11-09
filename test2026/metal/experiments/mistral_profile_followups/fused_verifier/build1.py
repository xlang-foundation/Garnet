"""Build and test the single changed native profile in an isolated source snapshot."""
from pathlib import Path
import hashlib,json,os,re,subprocess,time
R=Path(__file__).resolve().parent;Q=R.parent/'mistral-causal-batch4-helper1';read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(n,d):
 p=R/n;assert not p.exists();p.write_text(json.dumps(d,indent=2)+'\n')
def run(name,cmd):
 save(name+'-command.json',cmd);start=time.time_ns()
 with (R/(name+'.stdout')).open('x') as o,(R/(name+'.stderr')).open('x') as e:
  p=subprocess.Popen(cmd,stdout=o,stderr=e,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));save(name+'-pid.json',dict(pid=p.pid,started_ns=start));print(name,p.pid,'running',flush=True);rc=p.wait()
 save(name+'-status.json',dict(status='terminal',returncode=rc,pid=p.pid,started_ns=start,finished_ns=time.time_ns()));print(name,rc,'terminal',flush=True)
 if rc:print((R/(name+'.stderr')).read_text()[-5000:],flush=True)
 assert rc==0,name
cmd=read(Q/'configure1-command.json');assert cmd.count('-DGARNET_METAL_FUSED_Q4_DECODE=OFF')==1
cmd=[str(R/'source1') if a==str(Q/'source1') else str(R/'build-on1') if a==str(Q/'build-on1') else '-DGARNET_METAL_FUSED_Q4_DECODE=ON' if a=='-DGARNET_METAL_FUSED_Q4_DECODE=OFF' else a for a in cmd]
run('configure1',cmd);run('build1',['/opt/homebrew/bin/cmake','--build',str(R/'build-on1'),'--parallel','4']);run('suite1',['/opt/homebrew/bin/ctest','--test-dir',str(R/'build-on1'),'--output-on-failure','--parallel','1'])
s=(R/'suite1.stdout').read_text();need=re.search(r'100% tests passed, 0 tests failed out of (\d+)',s);assert need and int(need[1])>=56 and 'fused_q4_decode' in s
cache=(R/'build-on1/CMakeCache.txt').read_text()
for flag in ['GARNET_METAL_FUSED_Q4_DECODE','GARNET_METAL_MISTRAL_VERIFY_GQA16_SCORES','GARNET_METAL_MISTRAL_VERIFY_GQA16_VALUES']:assert flag+':BOOL=ON' in cache,flag
for n,h in read(R/'source-bindings1.json')['files_sha256'].items():assert sha(R/'source1'/n)==h,n
save('build-audit1.json',dict(passed=True,native_tests=int(need[1]),source_unchanged=True,changed_profile_options=['GARNET_METAL_FUSED_Q4_DECODE:OFF->ON'],candidate_library_sha256=sha(R/'build-on1/libgarnet.dylib'),control_library_sha256=sha(Q/'build-on1/libgarnet.dylib'),source_bindings_sha256=sha(R/'source-bindings1.json'),no_dirty_root_build=True,no_model_or_comparator_execution=True,goal_complete=False))
