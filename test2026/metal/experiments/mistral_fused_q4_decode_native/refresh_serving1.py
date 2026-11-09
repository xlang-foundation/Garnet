"""Bind a fresh build/model/rollback qualification before any fresh serving run."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (R/'admission1-status.json').exists() and not (R/'admission1-preflight.json').exists()
for n in ['build-audit3.json','model-audit2.json','verification-audit3.json']:assert read(R/n)['passed'],n
engine=read(R/'engine-admission1.json');plan=read(R/'admission-plan1.json')
engine['native_library_sha256']=sha(R/'library3/libgarnet.dylib')
assert engine['native_library_sha256']==read(R/'build-audit3.json')['library_sha256']
engine['source_manifest_sha256']=sha(R/'source-bindings3.json')
engine['profile_sha256']=sha(R/'profile-admission1.json')
(R/'engine-admission1.json').write_text(json.dumps(engine,indent=2)+'\n')
plan.update(engine_sha256=sha(R/'engine-admission1.json'),profile_sha256=engine['profile_sha256'],worker_sha256=sha(R/'tools1/local_coder_batch_worker.py'),native_model_state_gate_sha256=sha(R/'model-audit2.json'))
(R/'admission-plan1.json').write_text(json.dumps(plan,indent=2)+'\n')
print('Fresh serving inputs bound; no inference executed')
