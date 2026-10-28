"""Freeze complete two-model capture audit before the first real model load."""
import datetime,hashlib,json,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();repo=root/'Garnet'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert read(root/'qwen35-mtp-stage-v2-controller-exit.json')['exit_code']==0
assert read(root/'qwen35-capture-topology-v1-parser-controls.json')['all_controls_passed']
prior=root/'qwen35-mtp-stage-v2-before-build-identity.json'
subprocess.run([sys.executable,str(repo/'test2026/cuda/validate_qwen35_execution_identity.py'),str(prior)],check=True)
identity=read(prior)
identity['capture_provenance']={'parent_manifest':prior.name,'parent_sha256':digest(prior),
    'scope':'Actual unchanged Garnet5EAD real-model capture trace. Inherited checkpoint full hashes/current metadata guards, no new learned checkpoint payload scan.'}
identity['captured_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
for name in ('profile_qwen35_capture_topology.py','profile_qwen35_capture_topology.x','qwen35_capture_dot.py',
             'test_qwen35_capture_dot.py','freeze_qwen35_capture_topology.py','summarize_qwen35_capture_topology.py'):
    identity['source_sha256']['test2026/cuda/'+name]=digest(repo/'test2026/cuda'/name)
for name in ('prepare-qwen35-capture-topology-v1.py','run-qwen35-capture-topology-v1.ps1',
             'qwen35-capture-topology-v1-prospective-design.json','qwen35-capture-topology-v1-parser-controls.json'):
    identity['orchestration_sha256'][name]=digest(root/name)
controls=read(root/'qwen35-capture-topology-v1-parser-controls.json')
identity['external_input_sha256'][controls['historical_format_fixture']]=digest(Path(controls['historical_format_fixture']))
output=root/'qwen35-capture-topology-v1-before-run-identity.json'
with output.open('x') as stream:json.dump(identity,stream,indent=2)
print('CAPTURE_TOPOLOGY_SOURCE_EPOCH_READY')
