"""Reconstruct upstream plus the value patch and verify every native source file."""
from pathlib import Path
import difflib,hashlib,json,subprocess
R=Path(__file__).resolve().parent;M=R.parent;N=M/'mistral-gqa16-score-native1';P=M/'mistral-gqa16-value-screen1';S=R/'source1'
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def files(root):return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
bind=read(R/'source-bindings1.json');target=bind['files_sha256'];assert files(S)==target
assert sha(N/'provenance2.json')==read(R/'upstream1.json')['base_native_provenance_sha256']
base=read(R/'upstream-source-bindings1.json')['files_sha256']
replay=M/'mistral-gqa16-value-source-replay1';assert not replay.exists();replay.mkdir()
subprocess.run(['/bin/cp','-cR',str(N/'source1'),str(replay/'source1')],check=True)
T=replay/'source1'
for p in T.rglob('*'):
 if p.is_symlink():continue
 if p.is_file():p.chmod(p.stat().st_mode|0o200)
 elif p.is_dir():p.chmod(p.stat().st_mode|0o700)
T.chmod(0o755)
subprocess.run(['/usr/bin/git','apply','--whitespace=nowarn',str(R/'upstream-main1.patch')],cwd=T,check=True)
assert files(T)==base,'Upstream replay differs'
changed=bind['changed_files'];assert len(changed)==10 and not any(n.startswith('xModel/') for n in changed)
patch=''
for name in changed:
 old=(T/name).read_text().splitlines(keepends=True) if name in base else []
 new=(S/name).read_text().splitlines(keepends=True)
 patch+=''.join(difflib.unified_diff(old,new,fromfile='a/'+name if name in base else '/dev/null',tofile='b/'+name,n=0))
path=R/'native1.patch';assert not path.exists();path.write_text(patch)
subprocess.run(['/usr/bin/git','apply','--unidiff-zero',str(path)],cwd=T,check=True)
assert files(T)==target,'Native replay differs'
header='// SPDX-License-Identifier: Apache-2.0\n#pragma once\nnamespace Garnet {\ninline constexpr const char* kVerificationGqaValueShader=R"METAL(\n'+(P/'value1.metal').read_text()+')METAL";\n}\n'
assert (S/'src/backends/metal/verification_gqa_value_shader.h').read_text()==header
for mode in ['on','default-off']:
 cache=(R/f'build-{mode}1/CMakeCache.txt').read_text()
 assert ('GARNET_METAL_MISTRAL_VERIFY_GQA16_VALUES:BOOL='+('ON' if mode=='on' else 'OFF')) in cache
 assert 'GARNET_METAL_MISTRAL_VERIFY_GQA16_SCORES:BOOL=ON' in cache
 assert '100% tests passed out of 56' in (R/f'suite-{mode}1.stdout').read_text()
 log=(R/f'build-{mode}1/Testing/Temporary/LastTest.log').read_text()
 assert 'Mistral GQA16 values PASS cases=62 fp32_words=1883915904' in log
 assert 'probabilities_context_fullkv_bitwise=true abort_partialflush_ownership=true owners=0' in log
result=dict(passed=True,source_files=len(target),upstream_files=len(base),changed_files=changed,patch_sha256=sha(path),upstream_patch_sha256=sha(R/'upstream-main1.patch'),full_source_reconstructed_exact=True,screen_value_shader_identical=True,xmodel_operator_changes=False,global_workspace_bytes=0,prepared_tiles_by_batch={'1':64,'4':32},default_option_on=False,native_on_tests=56,native_default_off_tests=56,unit_cases_per_build=62,unit_fp32_words_per_build=1883915904,dependency_rejection_qualified=True,source_binding_sha256=sha(R/'source-bindings1.json'),screen_provenance_sha256=sha(P/'provenance1.json'),source_replay=str(T),no_comparator_execution=True,serving_speed_claim=False,goal_complete=False)
(R/'source-audit1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
