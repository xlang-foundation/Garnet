"""New sanitizer epoch after complete parent-pool native plain validation."""
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
output = root / 'qwen35-parent-mtp-memcheck-v1-before-run-identity.json'
assert not output.exists()
assert read(root / 'qwen35-parent-mtp-v1-controller-exit.json')['exit_code'] == 0
assert read(root / 'qwen35-parent-mtp-v1-summary.json')['native_plain_controls_passed']
prior = root / 'qwen35-parent-mtp-v1-before-run-identity.json'
subprocess.run([sys.executable, str(repo / 'test2026/cuda/validate_qwen35_execution_identity.py'), str(prior)], check=True)
identity = read(prior)
identity['capture_provenance'] = {'parent_manifest': prior.name, 'parent_manifest_sha256': digest(prior),
    'parent_captured_at_utc': identity['captured_at_utc'],
    'checkpoint_full_hashes': 'Inherited validated full digests; current size/mtime guards revalidated. No fresh payload scan.'}
identity['captured_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
for name in ('freeze_qwen35_parent_mtp_memcheck.py', 'summarize_qwen35_parent_mtp_memcheck.py'):
    identity['source_sha256']['test2026/cuda/' + name] = digest(repo / 'test2026/cuda' / name)
names = ['run-qwen35-parent-mtp-memcheck-v1.ps1', 'qwen35-parent-mtp-memcheck-v1-prospective-design.json',
         'qwen35-parent-mtp-v1-controller-exit.json', 'qwen35-parent-mtp-v1-summary.json']
for model, batch in [('2b', 1), ('4b', 8)]:
    for suffix in ('.json', '.log', '-exit.json'):
        names.append(f'qwen35-parent-mtp-{model}-b{batch}-plain-v1' + suffix)
for name in names:
    identity['orchestration_sha256'][name] = digest(root / name)
output.write_text(json.dumps(identity, indent=2) + '\n')
print('PARENT_MTP_MEMCHECK_SOURCE_EPOCH_READY')
