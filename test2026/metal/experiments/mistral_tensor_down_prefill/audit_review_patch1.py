"""Apply the exact native review patch to fresh copies of every changed base file."""
from pathlib import Path
import hashlib,json,subprocess
R=Path(__file__).resolve().parent;Q=R.parent/'mistral-causal-batch4-helper1';F=R.parent/'mistral-down-tensor-patch-replay1'
assert not F.exists();S=F/'source1';S.mkdir(parents=True)
changed=json.loads((R/'preparation1.json').read_text())['changed_files']
for n in changed:
 p=Q/'source1'/n
 if p.exists():
  out=S/n;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(p.read_bytes())
cmd=['/usr/bin/git','apply','--unidiff-zero','--whitespace=error',str(R/'native-review1.patch')]
with (F/'stdout1.log').open('x') as o,(F/'stderr1.log').open('x') as e:r=subprocess.run(cmd,cwd=S,stdout=o,stderr=e)
assert r.returncode==0,(F/'stderr1.log').read_text()
assert all((S/n).read_bytes()==(R/'source1'/n).read_bytes() for n in changed)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report=dict(passed=True,changed_files=changed,source_edits_identical=True,zero_context_patch=True,patch_sha256=sha(R/'native-review1.patch'),source_bindings_sha256=sha(R/'source-bindings1.json'),no_model_or_gpu_execution=True,no_comparator_execution=True)
(F/'audit1.json').write_text(json.dumps(report,indent=2)+'\n');(R/'patch-audit1.json').write_text(json.dumps(report,indent=2)+'\n');print('Native review patch replay passed')
