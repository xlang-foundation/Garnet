"""Audit four cold-ranked component anchors without promoting a serving policy."""
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
output = root / 'cublaslt-cold-selection-v1-summary.json'
assert not output.exists()
design = json.loads((root / 'cublaslt-cold-selection-v1-prospective-design.json').read_text())
identity_path = root / 'cublaslt-cold-selection-v1-before-build-identity.json'
identity_sha = hashlib.sha256(identity_path.read_bytes()).hexdigest()
binary_sha = hashlib.sha256((root / 'cublaslt-cold-probe-v1.exe').read_bytes()).hexdigest()
fixtures = json.loads((root / 'lossless-bf16-projection-v1-fixtures.json').read_text())['fixtures']
records = []
for anchor in design['anchors']:
    tag = 'cublaslt-cold-selection-v1-' + anchor['fixture']
    path, exit_path = root / (tag + '.jsonl'), root / (tag + '-exit.json')
    terminal = json.loads(exit_path.read_text(encoding='utf-8-sig'))
    assert terminal['exit_code'] == 0
    assert terminal['binary_sha256'].lower() == binary_sha
    assert terminal['execution_identity_sha256'].lower() == identity_sha
    assert terminal['gpu_uuid'] == design['gpu_uuid']
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]['event'] == 'metadata' and rows[-1]['event'] == 'complete'
    assert rows[0]['gpu_uuid'] == design['gpu_uuid'] and rows[0]['gpu'] == design['gpu_name']
    assert all(rows[0][key] == anchor[key] for key in ('rows', 'inner', 'columns'))
    fixture = next(row for row in fixtures if row['name'] == anchor['fixture'])
    assert rows[0]['fixture_raw_sha256'] == fixture['raw_sha256']
    frozen = [row for row in rows if row['event'] == 'selection_frozen']
    holdouts = [row for row in rows if row['event'] == 'holdout']
    assert len(frozen) == len(holdouts) == 1
    winner, holdout = frozen[0], holdouts[0]
    assert winner['config'] == holdout['config'] and not winner['other_finalists_allowed_for_rescue']
    assert rows.index(winner) < rows.index(next(row for row in rows if row['event'] == 'holdout_correctness'))
    assert holdout['all_output_words_bit_exact'] and rows[-1]['all_guards_inputs_weights_preserved']
    assert len(holdout['cold']['graph_us'][0]) == len(holdout['cold']['graph_us'][1]) == design['holdout_cold_pairs']
    assert len(holdout['warm']['graph_us'][0]) == len(holdout['warm']['graph_us'][1]) == design['holdout_warm_pairs']
    records.append({'anchor': anchor, 'selected': winner, 'holdout': holdout,
                    'sources_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (path, exit_path)}})
models = []
for size in ('2b', '4b'):
    selected = [row for row in records if row['anchor']['fixture'].startswith(size + '-')]
    assert len(selected) == 2
    ordinary = sum(row['anchor']['operator_count_per_target_pass'] * row['holdout']['cold']['baseline_median_us'] for row in selected)
    candidate = sum(row['anchor']['operator_count_per_target_pass'] * row['holdout']['cold']['candidate_median_us'] for row in selected)
    no_regression = all(row['holdout']['cold']['ordinary_over_candidate'] >= 1 for row in selected)
    models.append({'model': size, 'ordinary_weighted_us': ordinary, 'candidate_weighted_us': candidate,
                   'ratio': ordinary / candidate, 'no_selected_operator_regression': no_regression,
                   'passes_component_screen': ordinary / candidate >= 1.10 and no_regression})
result = {'records': records, 'models': models,
          'binary_sha256': binary_sha, 'execution_identity_sha256': identity_sha,
          'permits_only_further_component_validation': all(row['passes_component_screen'] for row in models),
          'sanitizers_completed': False, 'production_model_modified': False,
          'full_request_accepted': False, 'model_vram_saving_measured': False, 'goal_2x_achieved': False,
          'scope': design['scope']}
drift = json.loads((root / 'qwen35-4b-batch-drift-v1-summary.json').read_text())
result['quality_context'] = {'drift_diagnostic_evidence_complete': drift['diagnostic_evidence_complete'],
    'cross_batch_root_cause_resolved': drift['root_cause_resolved'],
    'component_agreement_does_not_resolve_model_batch_drift': True}
attribution = json.loads((root / 'qwen35-layer0-stage-first-token-attribution-v1.json').read_text())
result['quality_context']['sampled_first_token_down_projection_attribution_verified'] = attribution['observed_first_token_down_projection_attribution_verified']
result['quality_context']['global_model_quality_remains_unaccepted'] = True
output.write_text(json.dumps(result, indent=2))
print('CUBLASLT_COLD_COMPONENT_SUMMARY', result['permits_only_further_component_validation'])
