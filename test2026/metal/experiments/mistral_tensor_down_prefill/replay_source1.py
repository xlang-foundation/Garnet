"""Reconstruct every native source file from the frozen base and deterministic preparation."""
from pathlib import Path
import difflib,hashlib,json,os,shutil,subprocess
R=Path(__file__).resolve().parent;M=R.parent;Q=M/'mistral-causal-batch4-helper1';F=M/'mistral-down-tensor-source-replay1'
assert not F.exists();F.mkdir();subprocess.run(['/bin/cp','-cR',str(Q/'source1'),str(F/'source1')],check=True)
for p in [F/'source1',*(F/'source1').rglob('*')]:
 if not p.is_symlink():p.chmod(p.stat().st_mode|0o200|(0o100 if p.is_dir() else 0))
shutil.copy2(R/'prepare1.py',F/'prepare1.py');(F/'prepare1.py').chmod(0o644)
cmd=['/Users/shawn/CantorAI/out/research/qwen-local-coder/asset-env/bin/python','-B',str(F/'prepare1.py')]
with (F/'stdout1.log').open('x') as o,(F/'stderr1.log').open('x') as e:p=subprocess.run(cmd,stdout=o,stderr=e)
assert p.returncode==0,(F/'stderr1.log').read_text()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();expected=json.loads((R/'source-bindings1.json').read_text())['files_sha256'];actual={str(p.relative_to(F/'source1')):sha(p) for p in sorted((F/'source1').rglob('*')) if p.is_file()};assert actual==expected
changed=json.loads((R/'preparation1.json').read_text())['changed_files'];patch=[];review=[]
for n in changed:
 a=(Q/'source1'/n).read_text().splitlines(True) if (Q/'source1'/n).exists() else [];b=(R/'source1'/n).read_text().splitlines(True)
 patch.extend(difflib.unified_diff(a,b,fromfile='a/'+n if a else '/dev/null',tofile='b/'+n));review.extend(difflib.unified_diff(a,b,fromfile='a/'+n if a else '/dev/null',tofile='b/'+n,n=0))
for n,s in [('native-source1.patch',''.join(patch)),('native-review1.patch',''.join(review))]:
 out=R/n;assert not out.exists();out.write_text(s)
report=dict(passed=True,reconstructed_source_files=len(actual),changed_files=changed,source_bindings_sha256=sha(R/'source-bindings1.json'),base_bindings_sha256=sha(Q/'source-bindings1.json'),preparation_script_sha256=sha(R/'prepare1.py'),patch_sha256={n:sha(R/n) for n in ['native-source1.patch','native-review1.patch']},no_build_or_model_execution=True,no_comparator_execution=True,goal_complete=False)
(F/'audit1.json').write_text(json.dumps(report,indent=2)+'\n');(R/'source-audit1.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
