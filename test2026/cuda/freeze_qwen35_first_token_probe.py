"""Freeze two sequential original-MTP boundary diagnostics after NCU closes."""
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
stem = 'qwen35-first-token-probe-v1'
output = root / (stem + '-before-run-identity.json')
assert not output.exists()
assert read(root / 'qwen35-ncu-v1-controller-exit.json')['exit_code'] == 1
assert 'ERR_NVGPUCTRPERM' in (root / 'profile-qwen35-ncu-2b-b1-v1.log').read_text()
assert not (root / 'profile-qwen35-ncu-4b-b8-v1-exit.json').exists()
prior = root / 'qwen35-ncu-v1-before-run-identity.json'
subprocess.run([sys.executable, str(repo / 'test2026/cuda/validate_qwen35_execution_identity.py'), str(prior)], check=True)
identity = read(prior)
identity['capture_provenance'] = {'parent_manifest': prior.name, 'parent_manifest_sha256': digest(prior),
    'parent_captured_at_utc': identity['captured_at_utc'],
    'checkpoint_full_hashes': 'Inherited full digests; unchanged size/mtime guards, no fresh payload scan.'}
identity['captured_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
assert read(root / 'qwen35-first-token-probe-v1-cpu-hooks.json')['all_hook_restoration_and_ordering_checks_passed']
for name in ('probe_qwen35_first_token.py', 'freeze_qwen35_first_token_probe.py',
             'summarize_qwen35_first_token_probe.py', 'test_qwen35_first_token_probe_hooks.py'):
    identity['source_sha256']['test2026/cuda/' + name] = digest(repo / 'test2026/cuda' / name)
for name in ('run-qwen35-first-token-probe-v1.ps1', 'qwen35-first-token-probe-v1-prospective-design.json',
             'qwen35-first-token-probe-v1-cpu-hooks.json',
             'qwen35-ncu-v1-controller-exit.json', 'profile-qwen35-ncu-2b-b1-v1-exit.json',
             'profile-qwen35-ncu-2b-b1-v1.log', 'profile-qwen35-ncu-2b-b1-v1-preservation.json'):
    identity['orchestration_sha256'][name] = digest(root / name)
identity['first_token_probe_scope'] = 'Original MTP only; Python timestamps preserve all28 input/output/control/work signatures. No client streaming or primary latency/memory promotion.'
output.write_text(json.dumps(identity, indent=2) + '\n')
print('QWEN35_FIRST_TOKEN_SOURCE_EPOCH_READY')
