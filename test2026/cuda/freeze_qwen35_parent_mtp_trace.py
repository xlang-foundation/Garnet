"""Capture a new diagnostic trace queue after six closed memory processes."""
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
stem = 'qwen35-parent-mtp-trace-v1'
output = root / (stem + '-before-run-identity.json')
assert not output.exists()
assert read(root / 'qwen35-parent-mtp-memory-v1-controller-exit.json')['exit_code'] == 0
assert read(root / 'qwen35-parent-mtp-memory-v1-summary.json')['all_six_memory_controls_passed']
prior = root / 'qwen35-parent-mtp-memory-v1-before-run-identity.json'
subprocess.run([sys.executable, str(repo / 'test2026/cuda/validate_qwen35_execution_identity.py'), str(prior)], check=True)
identity = read(prior)
identity['capture_provenance'] = {'parent_manifest': prior.name, 'parent_manifest_sha256': digest(prior),
    'parent_captured_at_utc': identity['captured_at_utc'],
    'checkpoint_full_hashes': 'Inherited full digests and revalidated stat guards; no fresh weight payload scan.'}
identity['captured_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
for name in ('profile_qwen35_parent_mtp.py', 'freeze_qwen35_parent_mtp_trace.py',
             'summarize_qwen35_parent_mtp_trace.py', 'summarize_qwen35_nsys.py', 'compare_qwen35_trace_requests.py'):
    identity['source_sha256']['test2026/cuda/' + name] = digest(repo / 'test2026/cuda' / name)
names = ['run-qwen35-parent-mtp-trace-v1.ps1', 'run-qwen35-after-parent-memory-v2.ps1', stem + '-prospective-design.json',
         stem + '-nsys-version.log',
         'qwen35-parent-mtp-memory-v1-controller-exit.json', 'qwen35-parent-mtp-memory-v1-summary.json']
for size, batch in [('2b', 1), ('4b', 8)]:
    names += [f'matched512-garnet-{size}-b{batch}-study-shared-mtp-v1-{role}.json' for role in ('a1', 'b1')]
for name in names:
    identity['orchestration_sha256'][name] = digest(root / name)
tool = Path('C:/Program Files/NVIDIA Corporation/Nsight Systems 2025.6.3/target-windows-x64/nsys.exe')
identity.setdefault('external_input_sha256', {})[str(tool)] = digest(tool)
output.write_text(json.dumps(identity, indent=2) + '\n')
print('PARENT_MTP_TRACE_SOURCE_EPOCH_READY')
