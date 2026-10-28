"""Freeze the complete two-model kernel-counter diagnostic queue."""
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
output = root / 'qwen35-ncu-v1-before-run-identity.json'
assert not output.exists()
assert read(root / 'qwen35-parent-mtp-trace-v2-controller-exit.json')['exit_code'] == 0
assert read(root / 'qwen35-parent-mtp-trace-v2-summary.json')['all_four_trace_controls_passed']
prior = root / 'qwen35-parent-mtp-trace-v2-before-run-identity.json'
subprocess.run([sys.executable, str(repo / 'test2026/cuda/validate_qwen35_execution_identity.py'), str(prior)], check=True)
identity = read(prior)
identity['capture_provenance'] = {'parent_manifest': prior.name, 'parent_manifest_sha256': digest(prior),
    'parent_captured_at_utc': identity['captured_at_utc'], 'checkpoint_full_hashes': 'Inherited full digests; stat guards revalidated without fresh model payload reads.'}
identity['captured_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
for name in ('freeze_qwen35_ncu.py', 'summarize_qwen35_ncu.py'):
    identity['source_sha256']['test2026/cuda/' + name] = digest(repo / 'test2026/cuda' / name)
for name in ('run-qwen35-ncu-v1.ps1', 'qwen35-ncu-v1-prospective-design.json', 'qwen35-ncu-kernel-plan-v2.json',
             'prepare-qwen35-ncu-inputs-v1.py',
             'observe-qwen35-cim-processes-v3.ps1', 'qwen35-ncu-v1-version.log',
             'qwen35-parent-mtp-trace-v2-controller-exit.json', 'qwen35-parent-mtp-trace-v2-summary.json'):
    identity['orchestration_sha256'][name] = digest(root / name)
tool = Path('C:/Program Files/NVIDIA Corporation/Nsight Compute 2026.1.1/target/windows-desktop-win7-x64/ncu.exe')
identity.setdefault('external_input_sha256', {})[str(tool)] = digest(tool)
identity['ncu_scope'] = 'Two original-model diagnostic processes; one sample per distinct hot-kernel launch configuration in warmed request8, replay allowed. No latency or memory acceptance.'
output.write_text(json.dumps(identity, indent=2) + '\n')
print('QWEN35_NCU_SOURCE_EPOCH_READY')
