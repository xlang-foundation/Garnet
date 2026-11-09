"""Reconstruct the two-file B4 adapter/diagnostic increment without GPU execution."""
from pathlib import Path
import difflib,hashlib,json,subprocess,sys
R=Path(__file__).resolve().parent;N=R.parent/'mistral-gqa16-value-native1';S=R/'source1'
assert len(sys.argv)==2,'FRESH_OUTPUT_DIR';O=Path(sys.argv[1]).resolve();assert not O.exists();O.mkdir(parents=True)
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def files(root):return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
assert sha(N/'provenance1.json')=='3caa7d472050aaf7f4d5ac353103854f57e495c95279e96d0d933a312a071e50'
base=read(N/'source-bindings1.json')['files_sha256'];target=read(R/'source-bindings1.json')['files_sha256']
assert files(N/'source1')==base and files(S)==target and set(base)==set(target)
changed=sorted(n for n in base if base[n]!=target[n]);assert set(changed)==set(read(R/'source-plan1.json')['changed_files']) and len(changed)==2
patch=''
for n in read(R/'source-plan1.json')['changed_files']:patch+=''.join(difflib.unified_diff((N/'source1'/n).read_text().splitlines(True),(S/n).read_text().splitlines(True),fromfile='a/'+n,tofile='b/'+n))
assert patch==(R/'helper-native1.patch').read_text()
T=O/'source1';subprocess.run(['/bin/cp','-cR',str(N/'source1'),str(T)],check=True)
for p in T.rglob('*'):
 if p.is_symlink():continue
 if p.is_file():p.chmod(p.stat().st_mode|0o200)
 elif p.is_dir():p.chmod(p.stat().st_mode|0o700)
T.chmod(0o755);subprocess.run(['/usr/bin/git','apply','--whitespace=error',str(R/'helper-native1.patch')],cwd=T,check=True)
assert files(T)==target
build=read(R/'build-audit1.json');assert build['passed'] and build['native_tests']==56 and sha(R/'build-on1/libgarnet.dylib')==build['library_sha256']
assert sha(R/'source-bindings1.json')==build['source_bindings_sha256']
cache=(R/'build-on1/CMakeCache.txt').read_text();assert 'CMAKE_HOME_DIRECTORY:INTERNAL='+str(S) in cache
assert 'GARNET_METAL_DENSE_PREFILL_MAX_TOKENS:STRING=512' in cache
assert '100% tests passed out of 56' in (R/'suite1.stdout').read_text()
for name in ['configure1','build1','suite1']:assert read(R/(name+'-status.json'))['returncode']==0
report=dict(passed=True,source_files=len(target),changed_files=changed,base_provenance_sha256=sha(N/'provenance1.json'),base_source_bindings_sha256=sha(N/'source-bindings1.json'),source_bindings_sha256=sha(R/'source-bindings1.json'),patch_sha256=sha(R/'helper-native1.patch'),library_sha256=build['library_sha256'],native_tests=56,fresh_source_subdirectory='source1',independent_source_reconstruction=True,xmodel_operator_changes=False,math_source_changes=False,no_model_execution=True,goal_complete=False)
(O/'source-audit1.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
