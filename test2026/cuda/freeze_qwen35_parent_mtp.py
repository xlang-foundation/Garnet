"""Freeze an explicitly declared adapter-only change after completed timing."""
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
output = root / 'qwen35-parent-mtp-v1-before-run-identity.json'
assert not output.exists()
assert read(root / 'qwen35-shared-mtp-study-v1-controller-exit.json')['exit_code'] == 0
assert read(root / 'qwen35-after-mtp-memcheck-v1-exit.json')['exit_code'] == 0
study = read(root / 'qwen35-shared-mtp-study-v1-summary.json')
assert study['all_completed_raw_controls_passed'] and all(not m['experimental_speed_screen_passed'] for m in study['models'])
assert all(m['process_observation_qualified'] for m in study['models'])
preedit = read(root / 'qwen35-parent-mtp-v1-pre-edit.json')
prior = root / preedit['parent_manifest']
assert digest(prior) == preedit['parent_sha256']
identity = read(prior)
name = 'test2026/cuda/qwen35_shared_mtp_experiment.py'
assert identity['source_sha256'][name] == preedit['old_source_sha256'][name]
cpu = read(root / 'qwen35-parent-mtp-adapter-cpu-v1.json')
assert cpu['all_cpu_checks_passed'] and not cpu['gpu_executed']
assert digest(repo / name) == cpu['source_sha256']
selection = read(root / 'qwen35-mtp-parent-storage-cpu-v1.json')
assert selection['all_cpu_guards_passed'] and not selection['gpu_executed']
assert digest(repo / 'test2026/cuda/qwen35_mtp_parent_storage.py') == selection['source_sha256']
identity['capture_provenance'] = {'parent_manifest': prior.name, 'parent_manifest_sha256': digest(prior),
    'parent_captured_at_utc': identity['captured_at_utc'],
    'pre_edit_identity': preedit,
    'intentional_source_delta': {name: {'old_sha256': identity['source_sha256'][name], 'new_sha256': digest(repo / name)}},
    'checkpoint_full_hashes': 'Inherited validated full digests with current size/mtime guards revalidated; no fresh payload scan.'}
identity['captured_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
identity['source_sha256'][name] = digest(repo / name)
for name in ('qwen35_mtp_parent_storage.py', 'probe_qwen35_parent_mtp.py',
             'freeze_qwen35_parent_mtp.py', 'summarize_qwen35_parent_mtp.py'):
    identity['source_sha256']['test2026/cuda/' + name] = digest(repo / 'test2026/cuda' / name)
names = ['run-qwen35-parent-mtp-v1.ps1', 'qwen35-parent-mtp-v1-prospective-design.json',
         'qwen35-parent-mtp-v1-pre-edit.json', 'check-qwen35-parent-mtp-adapter-v1.py',
         'qwen35-parent-mtp-adapter-cpu-v1.json', 'check-qwen35-mtp-parent-storage-v1.py',
         'qwen35-mtp-parent-storage-cpu-v1.json', 'qwen35-mtp-parent-storage-review-v1.json',
         'qwen35-shared-mtp-study-v1-controller-exit.json', 'qwen35-after-mtp-memcheck-v1-exit.json',
         'qwen35-shared-mtp-study-v1-summary.json', 'qwen35-shared-mtp-stage-metrics-v1.json']
for name in names:
    identity['orchestration_sha256'][name] = digest(root / name)
output.write_text(json.dumps(identity, indent=2) + '\n')
# Every inherited source/input/runtime other than the one declared replacement
# must still match its old bytes. Native DLL and checkpoint guards stay fixed.
subprocess.run([sys.executable, str(repo / 'test2026/cuda/validate_qwen35_execution_identity.py'), str(output)], check=True)
print('PARENT_MTP_SOURCE_EPOCH_READY')
