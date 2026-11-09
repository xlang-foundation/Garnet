"""Build a clean verified source tree for the zero-copy committed-KV experiment.
Usage: python3 prepare_native_source.py GARNET_REPOSITORY ADMITTED_SOURCE ADMITTED_MANIFEST NEW_SOURCE
Delegates the admitted 28-overlay + 11-file hoist foundation, then applies only this candidate.
"""
from pathlib import Path
import hashlib,json,subprocess,sys
repo,admitted,manifest,out=[Path(v).resolve() for v in sys.argv[1:]]
here=Path(__file__).resolve().parent
bindings=json.loads((here/'native-bindings1.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
helper=repo/'test2026/metal/experiments/mistral_kv_commit_hoist/prepare_native_source.py'
assert sha(helper)==bindings['hoist_prepare_helper_sha256']
assert not out.exists()
subprocess.run([sys.executable,'-B',str(helper),str(repo),str(admitted),str(manifest),str(out)],check=True)
for name,h in bindings['base_existing_files_sha256'].items():assert sha(out/name)==h,name
subprocess.run(['git','apply','--check',str(here/'native-candidate1.patch')],cwd=out,check=True)
subprocess.run(['git','apply',str(here/'native-candidate1.patch')],cwd=out,check=True)
for name,h in bindings['candidate_owned_files_sha256'].items():assert sha(out/name)==h,name
(out.parent/(out.name+'-committed-preparation-audit.json')).write_text(json.dumps(dict(passed=True,hoist_prepare_helper_sha256=sha(helper),candidate_owned_files_sha256=bindings['candidate_owned_files_sha256'],base_existing_files_sha256=bindings['base_existing_files_sha256'],numerical_policy='committed_kv_mps_fp32_v1',prefix_domain='committed-mps-fp32-v1',default_option='OFF',extra_gpu_buffers=0),indent=2)+'\n')
print('Verified committed KV native source:',out)
