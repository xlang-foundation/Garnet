"""Freeze the whole shared-MTP queue before either model is loaded."""
import ast
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
output = root / 'qwen35-shared-mtp-v1-before-run-identity.json'
assert not output.exists()
prior = root / 'cublaslt-cold-selection-v1-before-build-identity.json'
assert read(root / 'cublaslt-cold-selection-v1-controller-exit.json')['exit_code'] == 0
assert not read(root / 'cublaslt-cold-selection-v1-summary.json')['permits_only_further_component_validation']
subprocess.run([sys.executable, str(repo / 'test2026/cuda/validate_qwen35_execution_identity.py'), str(prior)], check=True)
identity = read(prior)
identity['capture_provenance'] = {'parent_manifest': prior.name, 'parent_manifest_sha256': digest(prior),
    'parent_captured_at_utc': identity['captured_at_utc'],
    'checkpoint_full_hashes': 'Inherited full digests with current size/mtime guards revalidated; no fresh payload scan.'}
identity['captured_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
for name in ('qwen35_shared_mtp_experiment.py', 'probe_qwen35_shared_mtp.py',
             'summarize_qwen35_shared_mtp.py', 'freeze_qwen35_shared_mtp.py'):
    relative = 'test2026/cuda/' + name
    ast.parse((repo / relative).read_text())
    identity['source_sha256'][relative] = digest(repo / relative)
names = ['run-qwen35-shared-mtp-v1.ps1', 'prepare-qwen35-shared-mtp-v1.py',
         'qwen35-shared-mtp-v1-prospective-design.json', 'audit-qwen35-mtp-work-v1.py', 'qwen35-mtp-work-v1.json',
         'cublaslt-cold-selection-v1-summary.json', 'cublaslt-cold-selection-v1-controller-exit.json']
design = read(root / 'qwen35-shared-mtp-v1-prospective-design.json')
for row in design['profiles']:
    path = root / row['request']
    assert digest(path) == row['sha256']
    names.append(row['request'])
for name in names:
    identity['orchestration_sha256'][name] = digest(root / name)
output.write_text(json.dumps(identity, indent=2) + '\n')
print('SHARED_MTP_SOURCE_EPOCH_READY', output)
