"""Prepare the clean bounded half-prefill candidate; never build a dirty tree.

Usage: python3 prepare_native_source.py REPOSITORY ADMITTED_SOURCE MANIFEST NEW_SOURCE
"""
from pathlib import Path
import hashlib,json,subprocess,sys

repo,admitted,manifest,out=[Path(x).resolve() for x in sys.argv[1:]]
here=Path(__file__).resolve().parent
binding=json.loads((here/'native-bindings7.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
helper=repo/'test2026/metal/experiments/mistral_committed_kv_attention/prepare_native_source.py'
assert sha(helper)==binding['committed_prepare_helper_sha256']
assert sha(manifest)==binding['admitted_manifest_sha256']
assert sha(here/'native-candidate7.patch')==binding['patch_sha256']
assert not out.exists()
subprocess.run([sys.executable,'-B',str(helper),str(repo),str(admitted),str(manifest),str(out)],check=True)
for name,h in binding['base_existing_files_sha256'].items():assert sha(out/name)==h,name
subprocess.run(['git','apply','--check',str(here/'native-candidate7.patch')],cwd=out,check=True)
subprocess.run(['git','apply',str(here/'native-candidate7.patch')],cwd=out,check=True)
for name,h in binding['candidate_owned_files_sha256'].items():assert sha(out/name)==h,name
audit=dict(passed=True,committed_prepare_helper_sha256=sha(helper),patch_sha256=binding['patch_sha256'],
    candidate_owned_files_sha256=binding['candidate_owned_files_sha256'],backend_tag=binding['backend_tag'],prefix_domain=binding['prefix_domain'],
    scratch_payload_and_guards_bytes=binding['scratch_payload_and_guards_bytes'],shared_status_requested_bytes=4,
    default_option='OFF',full_model_quality_qualified=False,prefix_reuse_qualified=False,serving_speed_claim=False)
(out.parent/(out.name+'-half-preparation-audit.json')).write_text(json.dumps(audit,indent=2)+'\n')
print('Verified bounded half-prefill native source:',out)
