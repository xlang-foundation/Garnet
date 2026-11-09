"""Reconstruct every source file from base+patch and check screen math composition."""
from pathlib import Path
import difflib,hashlib,json,subprocess
R=Path(__file__).resolve().parent;M=R.parent;B=M/'mistral-wide-lifecycle1';P=M/'mistral-gqa16-score-screen1';S=R/'source1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
bind=read(R/'source-bindings1.json');base=read(B/'source-bindings1.json')['files_sha256'];target=bind['files_sha256']
assert {str(p.relative_to(S)):sha(p) for p in S.rglob('*') if p.is_file()}==target
assert {str(p.relative_to(B/'source1')):sha(p) for p in (B/'source1').rglob('*') if p.is_file()}==base
changed=bind['changed_files'];assert len(changed)==9 and not any(n.startswith('xModel/') for n in changed)
patch=''
for name in changed:
 old=(B/'source1'/name).read_text().splitlines(keepends=True) if name in base else []
 new=(S/name).read_text().splitlines(keepends=True)
 patch+=''.join(difflib.unified_diff(old,new,fromfile='a/'+name if name in base else '/dev/null',tofile='b/'+name,n=0))
patchPath=R/'native1.patch'
if patchPath.exists():assert patchPath.read_text()==patch
else:patchPath.write_text(patch)
# One separate complete source replay, never the dirty Garnet root.
replay=M/'mistral-gqa16-source-replay1'
if not replay.exists():
 replay.mkdir();subprocess.run(['/bin/cp','-cR',str(B/'source1'),str(replay/'source1')],check=True)
 for p in (replay/'source1').rglob('*'):
  if p.is_file():p.chmod(p.stat().st_mode|0o200)
  elif p.is_dir():p.chmod(p.stat().st_mode|0o700)
 (replay/'source1').chmod(0o755)
 subprocess.run(['/usr/bin/git','apply','--unidiff-zero',str(patchPath)],cwd=replay/'source1',check=True)
actual={str(p.relative_to(replay/'source1')):sha(p) for p in (replay/'source1').rglob('*') if p.is_file()}
assert actual==target
expectedDot=(P/'dots1.metal').read_text().replace(' const uint first=uint(slots[row*uint(p.sStride)]);\n','').replace(' threadgroup float keys[32*128];',' // All-inactive rows may have NaN slots. Never convert or address them.\n if(maximum==0)return;\n const uint first=uint(slots[row*uint(p.sStride)]);\n threadgroup float keys[32*128];')
expected='// SPDX-License-Identifier: Apache-2.0\n#pragma once\nnamespace Garnet {\ninline constexpr const char* kVerificationGqaScoreShader=R"METAL(\n'+expectedDot+(P/'softmax1.metal').read_text()+')METAL";\n}\n'
assert (S/'src/backends/metal/verification_gqa_score_shader.h').read_text()==expected
for mode in ['on','default-off']:
 cache=(R/f'build-{mode}1/CMakeCache.txt').read_text()
 assert 'GARNET_METAL_NATIVE_BFLOAT_BODY:BOOL=OFF' in cache
 assert ('GARNET_METAL_MISTRAL_VERIFY_GQA16_SCORES:BOOL='+('ON' if mode=='on' else 'OFF')) in cache
 assert '100% tests passed out of 55' in (R/f'suite-{mode}1.stdout').read_text()
result=dict(passed=True,source_files=len(target),base_files=len(base),changed_files=changed,patch_sha256=sha(patchPath),full_source_reconstructed_exact=True,screen_math_identical_except_uniform_inactive_row_guard=True,xmodel_operator_changes=False,default_option_on=False,native_on_tests=55,native_default_off_tests=55,dependency_rejection_qualified=True,source_binding_sha256=sha(R/'source-bindings1.json'),screen_provenance_sha256=sha(P/'provenance1.json'),source_replay=str(replay/'source1'),no_comparator_execution=True,serving_speed_claim=False,goal_complete=False)
(R/'source-audit1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
