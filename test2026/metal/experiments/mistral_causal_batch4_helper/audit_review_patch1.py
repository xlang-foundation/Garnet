"""Validate the review patch's zero-context encoding against both frozen sources."""
from pathlib import Path
import difflib,hashlib,json,subprocess,sys
assert len(sys.argv)==3,'RAW_QUALIFICATION_ROOT FRESH_OUTPUT_DIRECTORY'
R=Path(sys.argv[1]).resolve();N=R.parent/'mistral-gqa16-value-native1';P=Path(__file__).resolve().parent
O=Path(sys.argv[2]).resolve();assert not O.exists();O.mkdir(parents=True)
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
changed=read(R/'source-plan1.json')['changed_files'];patch=''
for name in changed:
 old=N/'source1'/name;new=R/'source1'/name
 patch+=''.join(difflib.unified_diff(old.read_text().splitlines(True),new.read_text().splitlines(True),fromfile='a/'+name,tofile='b/'+name,n=0))
 target=O/'source1'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(old.read_bytes())
assert patch==(P/'helper-native1.patch').read_text() and len(changed)==2
subprocess.run(['/usr/bin/git','apply','--unidiff-zero','--whitespace=error',str(P/'helper-native1.patch')],cwd=O/'source1',check=True)
assert all((O/'source1'/name).read_bytes()==(R/'source1'/name).read_bytes() for name in changed)
result=dict(passed=True,files=changed,source_edits_identical=True,zero_context_patch=True,raw_patch_sha256=sha(R/'helper-native1.patch'),review_patch_sha256=sha(P/'helper-native1.patch'),raw_provenance_sha256=sha(R/'provenance1.json'),no_model_execution=True)
(O/'package-patch-audit1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
