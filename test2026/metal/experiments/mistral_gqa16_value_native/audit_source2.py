"""Independently reconstruct both patches in a fresh directory, without GPU work."""
from pathlib import Path
import difflib,hashlib,json,subprocess,sys
R=Path(__file__).resolve().parent;M=R.parent;N=M/'mistral-gqa16-score-native1';P=M/'mistral-gqa16-value-screen1';S=R/'source1'
assert len(sys.argv)==2,'FRESH_OUTPUT_DIR'
O=Path(sys.argv[1]).resolve();assert not O.exists();O.mkdir(parents=True)
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def files(root):return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
target=read(R/'source-bindings1.json')['files_sha256'];base=read(R/'upstream-source-bindings1.json')['files_sha256'];original=read(R/'source-audit1.json')
assert original['passed'] and files(S)==target
assert sha(N/'provenance2.json')==read(R/'upstream1.json')['base_native_provenance_sha256']
assert files(N/'source1')==read(N/'source-bindings1.json')['files_sha256']
T=O/'source1';subprocess.run(['/bin/cp','-cR',str(N/'source1'),str(T)],check=True)
for p in T.rglob('*'):
 if p.is_symlink():continue
 if p.is_file():p.chmod(p.stat().st_mode|0o200)
 elif p.is_dir():p.chmod(p.stat().st_mode|0o700)
T.chmod(0o755)
subprocess.run(['/usr/bin/git','apply','--whitespace=nowarn',str(R/'upstream-main1.patch')],cwd=T,check=True)
assert files(T)==base
patch=''
for name in original['changed_files']:
 old=(T/name).read_text().splitlines(keepends=True) if name in base else []
 new=(S/name).read_text().splitlines(keepends=True)
 patch+=''.join(difflib.unified_diff(old,new,fromfile='a/'+name if name in base else '/dev/null',tofile='b/'+name,n=0))
assert patch==(R/'native1.patch').read_text()
subprocess.run(['/usr/bin/git','apply','--unidiff-zero',str(R/'native1.patch')],cwd=T,check=True)
assert files(T)==target and files(Path(original['source_replay']))==target
header='// SPDX-License-Identifier: Apache-2.0\n#pragma once\nnamespace Garnet {\ninline constexpr const char* kVerificationGqaValueShader=R"METAL(\n'+(P/'value1.metal').read_text()+')METAL";\n}\n'
assert (S/'src/backends/metal/verification_gqa_value_shader.h').read_text()==header
for mode in ['on','default-off']:
 cache=(R/f'build-{mode}1/CMakeCache.txt').read_text()
 assert ('GARNET_METAL_MISTRAL_VERIFY_GQA16_VALUES:BOOL='+('ON' if mode=='on' else 'OFF')) in cache
 assert 'GARNET_METAL_MISTRAL_VERIFY_GQA16_SCORES:BOOL=ON' in cache
 assert '100% tests passed out of 56' in (R/f'suite-{mode}1.stdout').read_text()
 assert 'Mistral GQA16 values PASS cases=62 fp32_words=1883915904' in (R/f'build-{mode}1/Testing/Temporary/LastTest.log').read_text()
assert 'GQA16 verification values require' in (R/'invalid-layout-off1.stderr').read_text()
result=dict(original,independent_fresh_source_reconstruction=True,fresh_source_subdirectory='source1',original_source_report_sha256=sha(R/'source-audit1.json'),original_source_auditor_sha256=sha(R/'audit_source1.py'),independent_source_auditor_sha256=sha(R/'audit_source2.py'))
(O/'source-audit2.json').write_text(json.dumps(result,indent=2)+'\n');print('Full source, patches, native route and test evidence independently reconstructed.')
