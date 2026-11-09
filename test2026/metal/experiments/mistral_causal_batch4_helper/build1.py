"""Build the bounded B4 helper adapter/diagnostic increment in an isolated source."""
from pathlib import Path
import hashlib,json,os,subprocess,time
R=Path(__file__).resolve().parent;N=R.parent/'mistral-gqa16-value-native1';S=R/'source1'
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(n,d):
 p=R/n;assert not p.exists();p.write_text(json.dumps(d,indent=2)+'\n')
base=read(N/'source-bindings1.json')['files_sha256'];files={str(p.relative_to(S)):sha(p) for p in sorted(S.rglob('*')) if p.is_file()}
changed=[n for n,h in files.items() if base.get(n)!=h];assert changed==read(R/'source-plan1.json')['changed_files'] or set(changed)==set(read(R/'source-plan1.json')['changed_files'])
assert len(changed)==2 and set(files)==set(base)
save('source-bindings1.json',dict(passed=True,files_sha256=files,changed_files=changed,base_provenance_sha256=sha(N/'provenance1.json')))
def run(name,argv):
 save(name+'-command.json',argv);start=time.time_ns()
 with (R/(name+'.stdout')).open('x') as o,(R/(name+'.stderr')).open('x') as e:
  p=subprocess.Popen(argv,stdout=o,stderr=e,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));save(name+'-pid.json',dict(pid=p.pid,started_ns=start));print(name,p.pid,'running',flush=True);rc=p.wait()
 save(name+'-status.json',dict(pid=p.pid,status='terminal',returncode=rc,started_ns=start,finished_ns=time.time_ns()));print(name,rc,'terminal',flush=True);assert rc==0,(name,rc)
argv=[str(S) if a==str(N/'source1') else str(R/'build-on1') if a==str(N/'build-on1') else a for a in read(N/'configure-on1-command.json')]
run('configure1',argv);run('build1',['/opt/homebrew/bin/cmake','--build',str(R/'build-on1'),'--parallel','4'])
run('suite1',['/opt/homebrew/bin/ctest','--test-dir',str(R/'build-on1'),'--output-on-failure','--parallel','1'])
assert '100% tests passed out of 56' in (R/'suite1.stdout').read_text()
assert {str(p.relative_to(S)):sha(p) for p in sorted(S.rglob('*')) if p.is_file()}==files
save('build-audit1.json',dict(passed=True,native_tests=56,library_sha256=sha(R/'build-on1/libgarnet.dylib'),source_bindings_sha256=sha(R/'source-bindings1.json'),inference_math_changed=False,no_dirty_root_build=True,no_comparator_execution=True,goal_complete=False))
