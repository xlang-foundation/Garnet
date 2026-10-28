"""Audit identical-prefix diagnostics; retain evidence without claiming a cause."""
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
output = root / 'qwen35-4b-batch-drift-v1-summary.json'
assert not output.exists()
identity_path = root / 'qwen35-4b-batch-drift-v1-before-run-identity.json'
identity_sha = hashlib.sha256(identity_path.read_bytes()).hexdigest()
identity = json.loads(identity_path.read_text())
design = json.loads((root / 'qwen35-4b-batch-drift-v1-prospective-design.json').read_text())
records, sources, head_checks = {}, {}, []


def f32_word(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def bf16_word(value):
    word = f32_word(value)
    return ((word + 0x7fff + ((word >> 16) & 1)) >> 16) & 65535


def bf16_float(value):
    return struct.unpack('<f', struct.pack('<I', bf16_word(value) << 16))[0]


def vector_difference(a, b):
    assert len(a) == len(b) == 2560
    assert all(math.isfinite(value) for value in a + b)
    squared = math.fsum((x - y) ** 2 for x, y in zip(a, b))
    denominator = math.fsum(x * x for x in a)
    return {'fp32_words_different': sum(f32_word(x) != f32_word(y) for x, y in zip(a, b)),
            'bf16_rounded_words_different': sum(bf16_word(x) != bf16_word(y) for x, y in zip(a, b)),
            'max_absolute_difference': max(abs(x - y) for x, y in zip(a, b)),
            'relative_l2': math.sqrt(squared / denominator) if denominator else None}


for batch in (1, 2):
    tag = f'qwen35-4b-batch-drift-b{batch}-v1'
    path, exit_path = root / (tag + '.json'), root / (tag + '-exit.json')
    data = json.loads(path.read_text())
    terminal = json.loads(exit_path.read_text(encoding='utf-8-sig'))
    assert terminal['exit_code'] == 0 and terminal['execution_identity_sha256'].lower() == identity_sha
    assert terminal['garnet_dll_sha256'].lower() == identity['garnet_binary_sha256'].lower()
    assert terminal['gpu_uuid'] == data['cuda_visible_devices'] == design['gpu_uuid']
    assert data['diagnostic_complete'] and not data['root_cause_resolved']
    assert len(data['trace']) == 27 and data['batch'] == batch
    source_path = root / f'optimized-batches-4b-b{batch}-plain-v1.json'
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == data['source_sha256']
    assert data['control']['generated_token_ids'] == data['instrumented']['generated_token_ids']
    assert [item['prediction_index_zero_based'] for item in data['trace']] == list(range(27))
    for item in data['trace']:
        assert len(item['raw_hidden_rows']) == len(item['normalized_hidden_rows']) == batch
        assert len(item['prediction']) == batch
        for row in item['raw_hidden_rows'] + item['normalized_hidden_rows']:
            assert len(row) == 2560 and all(math.isfinite(value) for value in row)
    head = data['head_isolation']
    query = [bf16_float(value) for value in head['normalized_row0']]
    assert query == head['bf16_query_row0']
    assert head['duplicate_m2_full_vocab_score_values_equal']
    largest_fraction = 0
    for sample in head['fp64_sampled_rows']:
        weight = sample['weight_fp32_words']
        assert len(weight) == len(query) == 2560
        assert all(math.isfinite(value) and bf16_float(value) == value for value in weight)
        terms = [x * y for x, y in zip(query, weight)]
        dot, absolute = math.fsum(terms), math.fsum(abs(value) for value in terms)
        gamma = len(query) * 2.0 ** -24 / (1 - len(query) * 2.0 ** -24)
        bound = gamma * absolute + (abs(dot) + gamma * absolute) / 256.0 + 2.0 ** -133
        assert math.isclose(dot, sample['fp64_dot'], rel_tol=2e-15, abs_tol=2e-15)
        assert math.isclose(bound, sample['original_fixed_fp32_accum_bf16_output_bound'], rel_tol=2e-15, abs_tol=2e-15)
        for name, observed in sample['observed_scores'].items():
            fraction = abs(observed - dot) / bound
            assert fraction <= 1, ('Original fixed head bound', batch, sample['token'], name)
            largest_fraction = max(largest_fraction, fraction)
    head_checks.append({'batch': batch, 'sampled_token_rows': len(head['fp64_sampled_rows']),
        'maximum_original_fixed_bound_fraction': largest_fraction,
        'same_query_m1_top1': head['same_query_m1'][0]['top8'][0],
        'same_query_m2_top1': head['same_query_m2'][0]['top8'][0],
        'same_query_m1_m2_decision_equal': head['same_query_m1'][0]['top8'][0]['id'] == head['same_query_m2'][0]['top8'][0]['id'],
        'watched_same_query_m1': head['same_query_m1'][0]['watched'],
        'watched_same_query_m2': head['same_query_m2'][0]['watched']})
    records[batch] = data
    for item in (path, exit_path):
        sources[item.name] = hashlib.sha256(item.read_bytes()).hexdigest()

one, two = records[1], records[2]
profile1, profile2 = dict(one['profile']), dict(two['profile'])
assert profile1.pop('batch') == 1 and profile2.pop('batch') == 2 and profile1 == profile2
assert one['image_sha256'][0] == two['image_sha256'][0]
ids1, ids2 = one['control']['generated_token_ids'][0], two['control']['generated_token_ids'][0]
assert ids1[:26] == ids2[:26] and (ids1[26], ids2[26]) == (22152, 21861)
inputs1, inputs2 = one['actual_prefill_inputs'], two['actual_prefill_inputs']
same_inputs = (all(value == inputs1['fp32_row_sha256'][0] for value in inputs2['fp32_row_sha256'])
    and inputs1['positions'][0] == inputs2['positions'][0]
    and inputs1['lengths'][0] == inputs2['lengths'][0]
    and inputs1['words_per_row'] == inputs2['words_per_row'])
differences = []
for a, b in zip(one['trace'], two['trace']):
    differences.append({'prediction_index_zero_based': a['prediction_index_zero_based'],
        'raw_hidden': vector_difference(a['raw_hidden_rows'][0], b['raw_hidden_rows'][0]),
        'normalized_hidden': vector_difference(a['normalized_hidden_rows'][0], b['normalized_hidden_rows'][0])})
first_raw = next((row['prediction_index_zero_based'] for row in differences if row['raw_hidden']['fp32_words_different']), None)
first_query = next((row['prediction_index_zero_based'] for row in differences if row['normalized_hidden']['bf16_rounded_words_different']), None)
result = {'diagnostic_evidence_complete': True, 'root_cause_resolved': False,
    'same_actual_prefill_inputs_and_positions': same_inputs,
    'common_generated_prefix_length': 26, 'divergence_generated_index_zero_based': 26,
    'batch1_divergent_token': ids1[26], 'batch2_divergent_token': ids2[26],
    'first_raw_hidden_fp32_difference_prediction': first_raw,
    'first_bf16_head_query_difference_prediction': first_query,
    'per_prediction_hidden_differences': differences, 'head_isolation_checks': head_checks,
    'sampled_head_fixed_bounds_passed': True, 'sources_sha256': sources,
    'execution_identity_sha256': identity_sha, 'garnet_dll_sha256': identity['garnet_binary_sha256'],
    'production_model_modified': False, 'independent_model_quality_accepted': False,
    'full_vocab_mathematical_argmax_proven': False, 'speed_or_memory_claim': False,
    'scope': 'Identical-prefix ordinary B1/B2 diagnosis only; actual input hashes, hidden differences, '
             'same-query head row-count effects and sampled FP64 score bounds. '
             'No automatic causal attribution, per-layer oracle, batch-invariant production policy, quality or2x admission.'}
output.write_text(json.dumps(result, indent=2))
print('BATCH_DRIFT_EVIDENCE_COMPLETE', 'same_inputs', same_inputs, 'first_raw', first_raw, 'first_bf16_query', first_query)
