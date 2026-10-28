"""Capture the full six-process memory queue only after parent memcheck passes."""
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
stem = 'qwen35-parent-mtp-memory-v1'
output = root / (stem + '-before-run-identity.json')
assert not output.exists()
assert read(root / 'qwen35-parent-mtp-memcheck-v1-controller-exit.json')['exit_code'] == 0
assert read(root / 'qwen35-parent-mtp-memcheck-v1-summary.json')['all_memcheck_controls_passed']
prior = root / 'qwen35-parent-mtp-memcheck-v1-before-run-identity.json'
subprocess.run([sys.executable, str(repo / 'test2026/cuda/validate_qwen35_execution_identity.py'), str(prior)], check=True)
identity = read(prior)
identity['capture_provenance'] = {'parent_manifest': prior.name, 'parent_manifest_sha256': digest(prior),
    'parent_captured_at_utc': identity['captured_at_utc'],
    'checkpoint_full_hashes': 'Inherited full digests; unchanged size/mtime guards revalidated without fresh payload scan.'}
identity['captured_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
for name in ('run_qwen35_parent_mtp_memory.py', 'freeze_qwen35_parent_mtp_memory.py',
             'summarize_qwen35_parent_mtp_memory.py', 'capture_windows_gpu_memory.py',
             'summarize_windows_gpu_memory.py', 'compare_qwen35_memory_workloads.py'):
    identity['source_sha256']['test2026/cuda/' + name] = digest(repo / 'test2026/cuda' / name)
names = ['run-qwen35-parent-mtp-memory-v1.ps1', 'run-qwen35-after-parent-memcheck-v1.ps1', stem + '-prospective-design.json',
         'observe-benchmark-process-starts.ps1', 'set-qwen35-cohort-environment.ps1',
         'qwen35-parent-mtp-memcheck-v1-controller-exit.json', 'qwen35-parent-mtp-memcheck-v1-summary.json']
for size, batch in [('2b', 1), ('4b', 8)]:
    names += [f'matched512-garnet-{size}-b{batch}-study-compact-kv-' + ('retry-v2-b1.json' if size == '2b' else 'b1.json'),
              f'memory-lmstudio-{size}-b{batch}-512-partitioned-session.json']
for name in names:
    identity['orchestration_sha256'][name] = digest(root / name)
output.write_text(json.dumps(identity, indent=2) + '\n')
print('PARENT_MTP_MEMORY_SOURCE_EPOCH_READY')
