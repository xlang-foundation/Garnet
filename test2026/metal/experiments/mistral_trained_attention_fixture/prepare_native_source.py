"""Prepare clean native fixture-capture source; never build the dirty worktree.
Usage: python3 prepare_native_source.py GARNET_REPO ADMITTED_SOURCE ADMITTED_MANIFEST FRESH_OUTPUT
"""
from pathlib import Path
import hashlib,json,subprocess,sys
repo,admitted,manifest,out=[Path(x).resolve() for x in sys.argv[1:]];here=Path(__file__).resolve().parent
b=json.loads((here/'native-bindings2.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
helper=repo/'test2026/metal/experiments/mistral_kv_commit_hoist/prepare_native_source.py';assert sha(helper)==b['hoist_prepare_helper_sha256'] and not out.exists()
subprocess.run([sys.executable,'-B',str(helper),str(repo),str(admitted),str(manifest),str(out)],check=True)
for n,h in b['base_existing_files_sha256'].items():assert sha(out/n)==h,n
subprocess.run(['git','apply','--check',str(here/'native-candidate2.patch')],cwd=out,check=True);subprocess.run(['git','apply',str(here/'native-candidate2.patch')],cwd=out,check=True)
for n,h in b['candidate_owned_files_sha256'].items():assert sha(out/n)==h,n
(out.parent/(out.name+'-capture-preparation-audit.json')).write_text(json.dumps(dict(passed=True,candidate_owned_files_sha256=b['candidate_owned_files_sha256'],base_existing_files_sha256=b['base_existing_files_sha256'],default='OFF',test_only=True,no_xmodel_or_neural_math_change=True),indent=2)+'\n')
print('Verified native trained fixture source:',out)
