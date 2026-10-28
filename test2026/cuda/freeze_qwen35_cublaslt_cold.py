"""Extend a completed batch epoch for a separate, model-free cold GEMM probe."""
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
repo = root / 'Garnet'
prior_path = root / 'optimized-batches-v1-before-run-identity.json'
output = root / 'cublaslt-cold-selection-v1-before-build-identity.json'
assert not output.exists(), 'Preserve cold-search epoch'
assert json.loads((root / 'optimized-batches-v1-controller-exit.json').read_text(encoding='utf-8-sig'))['exit_code'] == 0
assert json.loads((root / 'optimized-batches-v1-summary.json').read_text())['all_six_new_profiles_plain_and_memcheck_passed']
assert json.loads((root / 'qwen35-4b-batch-drift-v1-controller-exit.json').read_text(encoding='utf-8-sig'))['exit_code'] == 0
assert json.loads((root / 'qwen35-4b-batch-drift-v1-summary.json').read_text())['diagnostic_evidence_complete']
assert json.loads((root / 'qwen35-layer0-stage-v1-controller-exit.json').read_text(encoding='utf-8-sig'))['exit_code'] == 0
stage = json.loads((root / 'qwen35-layer0-stage-v1-summary.json').read_text())
assert stage['all_original_output_and_state_words_exact_both_repeats'] and stage['sampled_original_fixed_bounds_passed']
attribution = json.loads((root / 'qwen35-layer0-stage-first-token-attribution-v1.json').read_text())
assert attribution['observed_first_token_down_projection_attribution_verified'] and attribution['all_residual_reconstructions_exact']
subprocess.run([sys.executable, str(repo / 'test2026/cuda/validate_qwen35_execution_identity.py'), str(prior_path)], check=True)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


identity = json.loads(prior_path.read_text())
identity['capture_provenance'] = {
    'parent_manifest': prior_path.name, 'parent_manifest_sha256': digest(prior_path),
    'parent_captured_at_utc': identity['captured_at_utc'],
    'checkpoint_full_hashes': 'Inherited from the complete parent epoch; current size/mtime guards validated. No fresh full-checkpoint hash claimed.'}
identity['captured_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
for name in ('probe_qwen35_cublaslt_cold.cu', 'freeze_qwen35_cublaslt_cold.py', 'summarize_qwen35_cublaslt_cold.py'):
    relative = 'test2026/cuda/' + name
    identity['source_sha256'][relative] = digest(repo / relative)
for name in ('run-cublaslt-cold-screen.ps1', 'build-cublaslt-cold-probe.cmd',
             'cublaslt-cold-selection-v1-prospective-design.json',
             'optimized-batches-v1-controller-exit.json', 'optimized-batches-v1-summary.json',
             'lossless-bf16-projection-v1-fixtures.json',
             'lossless-bf16-storage-paths-v1-rejection-summary.json', 'cublaslt-manual-v1-summary.json',
             'summarize-qwen35-batch-owners.py', 'optimized-batch-owned-memory-v1.json'):
    identity['orchestration_sha256'][name] = digest(root / name)
for name in ('qwen35-4b-batch-drift-v1-controller-exit.json', 'qwen35-4b-batch-drift-v1-summary.json',
             'qwen35-4b-batch-drift-v1-before-run-identity.json',
             'optimized-batch-row-isolation-v1.json', 'optimized-batch-row-differences-v1.json'):
    identity['orchestration_sha256'][name] = digest(root / name)
for batch in (1, 2):
    for suffix in ('.json', '.log', '-exit.json'):
        name = f'qwen35-4b-batch-drift-b{batch}-v1' + suffix
        identity['orchestration_sha256'][name] = digest(root / name)
for name in ('qwen35-prefill-state-v1-summary.json', 'qwen35-layer0-debug-v2-summary.json',
             'qwen35-layer0-debug-audit-v2-exit.json', 'qwen35-layer0-stage-v1-before-run-identity.json',
             'qwen35-layer0-stage-v1-controller-exit.json', 'qwen35-layer0-stage-v1-build-exit.json',
             'qwen35-layer0-stage-v1-summary.json', 'attribute-qwen35-layer0-stage-v1.py',
             'qwen35-layer0-stage-first-token-attribution-v1.json'):
    identity['orchestration_sha256'][name] = digest(root / name)
identity['completed_quality_context'] = {
    'observed_first_token_down_projection_attribution_verified': True,
    'actual_firstlayer_component_exact_original_controls_passed': True,
    'sampled_original_fixed_fp64_bounds_passed': True,
    'global_independent_model_quality_accepted': False,
    'scope': 'Same BF16 down operands/weights at rows64/128 explain four first-token hidden differences through exact original residual additions. This does not identify an internal algorithm or accept global model quality/batch invariance. Cold decode-row component results remain separate.'}
fixtures = json.loads((root / 'lossless-bf16-projection-v1-fixtures.json').read_text())
for fixture in fixtures['fixtures']:
    if not fixture['inner']:
        continue
    path = Path(fixture['raw_file'])
    assert digest(path) == fixture['raw_sha256']
    identity['orchestration_sha256'][path.relative_to(root).as_posix()] = fixture['raw_sha256']
for size, batches in (('2b', (2, 4, 8)), ('4b', (1, 2, 4))):
    for batch in batches:
        for mode in ('plain', 'memcheck'):
            for suffix in ('.json', '.log', '-exit.json'):
                name = f'optimized-batches-{size}-b{batch}-{mode}-v1' + suffix
                identity['orchestration_sha256'][name] = digest(root / name)
output.write_text(json.dumps(identity, indent=2))
print('CUBLASLT_COLD_SOURCE_EPOCH_READY', output)
