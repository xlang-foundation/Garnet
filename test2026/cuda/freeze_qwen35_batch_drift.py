"""Freeze a separate ordinary-target diagnostic after the full batch queue."""
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
repo = root / 'Garnet'
prior_path = root / 'optimized-batches-v1-before-run-identity.json'
output = root / 'qwen35-4b-batch-drift-v1-before-run-identity.json'
assert not output.exists(), 'Preserve diagnostic epoch'
assert json.loads((root / 'optimized-batches-v1-controller-exit.json').read_text(encoding='utf-8-sig'))['exit_code'] == 0
assert json.loads((root / 'optimized-batches-v1-summary.json').read_text())['all_six_new_profiles_plain_and_memcheck_passed']
subprocess.run([sys.executable, str(repo / 'test2026/cuda/validate_qwen35_execution_identity.py'), str(prior_path)], check=True)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


identity = json.loads(prior_path.read_text())
identity['capture_provenance'] = {
    'parent_manifest': prior_path.name, 'parent_manifest_sha256': digest(prior_path),
    'parent_captured_at_utc': identity['captured_at_utc'],
    'checkpoint_full_hashes': 'Inherited parent full digests, current size/mtime guards revalidated. Not freshly hashed model payloads.'}
identity['captured_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
for name in ('probe_qwen35_batch_drift.py', 'freeze_qwen35_batch_drift.py', 'summarize_qwen35_batch_drift.py'):
    relative = 'test2026/cuda/' + name
    identity['source_sha256'][relative] = digest(repo / relative)
for name in ('run-qwen35-batch-drift.ps1', 'qwen35-4b-batch-drift-v1-prospective-design.json',
             'qwen35-4b-batch-drift-v1-preparation.json',
             'qwen35-4b-batch-drift-preparation-v2.json',
             'qwen35-batch-drift-auditor-selfcheck-v1.json',
             'optimized-batches-v1-controller-exit.json', 'optimized-batches-v1-summary.json',
             'audit-qwen35-batch-row-isolation.py', 'optimized-batch-row-isolation-v1.json',
             'optimized-batch-row-differences-v1.json', 'batch-invariant-source-review-v1.json'):
    identity['orchestration_sha256'][name] = digest(root / name)
checks = json.loads((root / 'qwen35-batch-drift-auditor-selfcheck-v1.json').read_text())
assert checks['all_nine_known_value_checks_passed']
for name, expected in checks['source_sha256'].items():
    assert digest(repo / 'test2026/cuda' / name) == expected, 'Numerical observer changed after golden checks'
for size, batches in (('2b', (2, 4, 8)), ('4b', (1, 2, 4))):
    for batch in batches:
        for mode in ('plain', 'memcheck'):
            for suffix in ('.json', '.log', '-exit.json'):
                name = f'optimized-batches-{size}-b{batch}-{mode}-v1' + suffix
                identity['orchestration_sha256'][name] = digest(root / name)
for path in sorted((root / 'models/Qwen3.5-4B').glob('*.json')):
    identity['orchestration_sha256'][path.relative_to(root).as_posix()] = digest(path)
audit = json.loads((root / 'optimized-batch-row-differences-v1.json').read_text())
four = next(item for item in audit['models'] if item['model'] == '4b')
assert [(row['batch'], row['first_differing_generated_index_zero_based']) for row in four['batches']] == [(1, None), (2, 26), (4, 26), (8, 26)]
output.write_text(json.dumps(identity, indent=2))
print('BATCH_DRIFT_SOURCE_EPOCH_READY', output)
