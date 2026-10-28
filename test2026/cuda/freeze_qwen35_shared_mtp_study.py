"""Freeze matched ABBA timing after both complete sanitizer cases pass."""
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
output = root / 'qwen35-shared-mtp-study-v1-before-run-identity.json'
assert not output.exists()
assert read(root / 'qwen35-shared-mtp-memcheck-v1-controller-exit.json')['exit_code'] == 0
assert read(root / 'qwen35-shared-mtp-memcheck-v1-summary.json')['all_memcheck_controls_passed']
prior = root / 'qwen35-shared-mtp-memcheck-v1-before-run-identity.json'
subprocess.run([sys.executable, str(repo / 'test2026/cuda/validate_qwen35_execution_identity.py'), str(prior)], check=True)
identity = read(prior)
identity['capture_provenance'] = {'parent_manifest': prior.name, 'parent_manifest_sha256': digest(prior),
    'parent_captured_at_utc': identity['captured_at_utc'],
    'checkpoint_full_hashes': 'Inherited full digests with current stat guards revalidated; no fresh payload scan.'}
identity['captured_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
for name in ('benchmark_qwen35_shared_mtp.py', 'freeze_qwen35_shared_mtp_study.py', 'summarize_qwen35_shared_mtp_study.py',
             'audit_qwen35_process_observations.py'):
    identity['source_sha256']['test2026/cuda/' + name] = digest(repo / 'test2026/cuda' / name)
names = ['run-qwen35-shared-mtp-study-v1.ps1', 'qwen35-shared-mtp-study-v1-prospective-design.json',
         'run-qwen35-after-mtp-memcheck-v1.ps1', 'qwen35-after-mtp-memcheck-v1-queued-identity.json',
         'observe-benchmark-process-starts.ps1', 'qwen35-shared-mtp-memcheck-v1-controller-exit.json',
         'qwen35-shared-mtp-memcheck-v1-summary.json']
for size, batch in [('2b', 1), ('4b', 8)]:
    for suffix in ('.json', '-exit.json', '.log'):
        names.append(f'qwen35-shared-mtp-{size}-b{batch}-memcheck-v1' + suffix)
for name in names:
    identity['orchestration_sha256'][name] = digest(root / name)
output.write_text(json.dumps(identity, indent=2) + '\n')
print('SHARED_MTP_ABBA_SOURCE_EPOCH_READY')
