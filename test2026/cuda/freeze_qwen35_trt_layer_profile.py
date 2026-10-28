"""Freeze sequential TensorRT-layer diagnostic models after the boundary probe."""
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
repo = root / 'Garnet'
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
output = root / 'qwen35-trt-layer-v1-before-run-identity.json'
assert not output.exists()
assert read(root / 'qwen35-first-token-probe-v2-controller-exit.json')['exit_code'] == 0
control = read(root / 'qwen35-first-token-probe-v2-summary.json')
assert control['all56_boundary_and_input_output_work_controls_passed'] and control['process_observation_qualified']
prior = root / 'qwen35-first-token-probe-v2-before-run-identity.json'
subprocess.run([sys.executable, str(repo / 'test2026/cuda/validate_qwen35_execution_identity.py'), str(prior)], check=True)
identity = read(prior)
identity['capture_provenance'] = {'parent_manifest': prior.name, 'parent_manifest_sha256': digest(prior),
    'parent_captured_at_utc': identity['captured_at_utc'],
    'checkpoint_full_hashes': 'Inherited full digests; unchanged size/mtime guards, no new payload scan.'}
identity['captured_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
for name in ('profile_qwen35_trt_layers.py', 'freeze_qwen35_trt_layer_profile.py', 'summarize_qwen35_trt_layers.py'):
    identity['source_sha256']['test2026/cuda/' + name] = digest(repo / 'test2026/cuda' / name)
for name in ('run-qwen35-trt-layer-v1.ps1', 'prepare-qwen35-trt-layer-v1.py', 'qwen35-trt-layer-v1-prospective-design.json',
             'qwen35-first-token-probe-v2-controller-exit.json', 'qwen35-first-token-probe-v2-summary.json'):
    identity['orchestration_sha256'][name] = digest(root / name)
identity['trt_layer_scope'] = 'One warmed original-MTP request per model, existing native IProfiler first execution per engine, no hardware counters or primary latency/memory acceptance.'
output.write_text(json.dumps(identity, indent=2) + '\n')
print('QWEN35_TRT_LAYER_SOURCE_EPOCH_READY')
