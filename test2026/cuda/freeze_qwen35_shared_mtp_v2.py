"""New execution epoch for a PowerShell7 relaunch; original startup retained."""
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
output = root / 'qwen35-shared-mtp-v2-before-run-identity.json'
assert not output.exists()
assert read(root / 'qwen35-shared-mtp-v1-controller-exit.json')['exit_code'] == 1
assert not (root / 'qwen35-shared-mtp-2b-b1-v1.json').exists()
prior = root / 'qwen35-shared-mtp-v1-before-run-identity.json'
subprocess.run([sys.executable, str(repo / 'test2026/cuda/validate_qwen35_execution_identity.py'), str(prior)], check=True)
identity = read(prior)
identity['capture_provenance'] = {'parent_manifest': prior.name, 'parent_manifest_sha256': digest(prior),
    'parent_captured_at_utc': identity['captured_at_utc'],
    'checkpoint_full_hashes': 'Inherited full digests; current size/mtime guards revalidated, no fresh payload scan.',
    'relaunch_reason': 'PowerShell5 Stop treated embedded-Python startup stderr as terminating before result creation. '
                       'Controller terminal1 and native handle missing; new epoch uses PowerShell7. No model-code change.'}
identity['captured_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
for name in ('freeze_qwen35_shared_mtp_v2.py', 'summarize_qwen35_shared_mtp_v2.py'):
    identity['source_sha256']['test2026/cuda/' + name] = digest(repo / 'test2026/cuda' / name)
for name in ('run-qwen35-shared-mtp-v2.ps1', 'qwen35-shared-mtp-v1-controller-exit.json', 'qwen35-shared-mtp-2b-b1-v1.log'):
    identity['orchestration_sha256'][name] = digest(root / name)
output.write_text(json.dumps(identity, indent=2) + '\n')
print('SHARED_MTP_V2_SOURCE_EPOCH_READY', output)
