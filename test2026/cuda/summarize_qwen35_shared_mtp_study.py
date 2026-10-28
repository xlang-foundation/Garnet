"""All20 ABBA timing plus separately identified historical equal32 subsets."""
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
sources = {}


def read(path):
    raw = path.read_bytes()
    sources[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw.decode('utf-8-sig'))


def key(row):
    return row['trial'], row['workload_group']


def median(values):
    return statistics.median(values)


def cohort(path):
    rows = read(path)
    assert len(rows) == 28 and len({key(r) for r in rows}) == 28
    assert sum(not r['warmup'] for r in rows) == 20
    assert len({r['process_id'] for r in rows}) == 1
    assert all(math.isfinite(r['wall_seconds']) and r['wall_seconds'] > 0 for r in rows)
    return {key(r): r for r in rows}


identity_path = root / 'qwen35-shared-mtp-study-v1-before-run-identity.json'
identity = read(identity_path)
epoch = sources[str(identity_path)]
design = read(root / 'qwen35-shared-mtp-study-v1-prospective-design.json')
result = {'all_completed_raw_controls_passed': False, 'models': [], 'execution_identity_sha256': epoch,
          'garnet_dll_sha256': identity['garnet_binary_sha256'], 'goal_2x_achieved': False,
          'independent_quality_accepted': False, 'process_memory_measured': False, 'automatic_promotion': False}
fields = ('profile', 'input_sha256', 'prompt_token_ids', 'generated_token_ids', 'sampled_tokens_including_stop',
          'finish_reasons', 'stop_token_ids', 'cache_policy', 'visual_tokens_per_row', 'native_build_policy',
          'request_output_limit', 'matched_batch_inputs', 'warmup', 'timing_scope')
for size, batch, old_study in [('2b', 1, 'compact-kv-retry-v2'), ('4b', 8, 'compact-kv')]:
    old_path = root / f'matched512-garnet-{size}-b{batch}-study-{old_study}-b1.json'
    reference = cohort(old_path)
    data = {}
    per_run = {}
    for role in ('a1', 'b1', 'b2', 'a2'):
        tag = f'matched512-garnet-{size}-b{batch}-study-shared-mtp-v1-{role}'
        terminal = read(root / (tag + '-exit.json'))
        assert terminal['exit_code'] == 0 and terminal['execution_identity_sha256'].lower() == epoch.lower()
        assert terminal['garnet_dll_sha256'].lower() == identity['garnet_binary_sha256'].lower()
        expected_mode = 'baseline' if role.startswith('a') else 'split'
        assert terminal['mode'] == expected_mode
        rows = cohort(root / (tag + '.json'))
        assert rows.keys() == reference.keys()
        for k, row in rows.items():
            assert all(row[f] == reference[k][f] for f in fields), (size, role, k, 'changed workload/target output')
            assert row['source_cohort_sha256'] == sources[str(old_path)]
            assert row['experimental_mtp_graphs']['mode'] == expected_mode
            assert row['experimental_mtp_graphs']['target_decode'] == 4 and row['experimental_mtp_graphs']['target_prefill'] == 64
            if not row['warmup']:
                assert all(g['contexts'] == 1 and g['graph_captures'] > 0 and g['graph_capture_failures'] == 0
                           for g in row['owned_memory']['decoder']['graphs'])
                if expected_mode == 'split':
                    graphs = row['owned_memory']['speculative']['mtp_experiment_graphs']
                    assert set(graphs) == {'1', '4', '16'}
                    assert all(g['external_bf16_weight_inputs'] == 8 and g['contexts'] == 1
                               and g['graph_capture_failures'] == 0 for g in graphs.values())
        data[role] = rows
        warm = [r for r in rows.values() if not r['warmup']]
        per_run[role] = {'median_group_ms': median(r['wall_seconds'] * 1000 for r in warm),
                         'median_prefill_ms': median(r['prefill_seconds'] * 1000 for r in warm),
                         'median_decode_ms': median(r['decode_seconds'] * 1000 for r in warm),
                         'owned_model_bytes_range': [min(r['owned_memory']['owned_model_bytes'] for r in warm),
                                                     max(r['owned_memory']['owned_model_bytes'] for r in warm)]}
    warm_keys = sorted(k for k, r in reference.items() if not r['warmup'])
    aggregate = {letter: {k: median(data[letter + n][k]['wall_seconds'] * 1000 for n in ('1', '2'))
                          for k in warm_keys} for letter in ('a', 'b')}
    values = {letter: median(rows.values()) for letter, rows in aggregate.items()}
    ratio = values['a'] / values['b']
    repeats_improve = all(per_run['a' + n]['median_group_ms'] > per_run['b' + n]['median_group_ms'] for n in ('1', '2'))
    observation_path = root / f'qwen35-shared-mtp-study-v1-{size}-b{batch}-process-audit.json'
    observation = read(observation_path)
    qualified = observation['all_four_roles_covered'] and observation['known_overlap_count'] == 0 and not observation['observer_metadata_errors']
    # Preserve exactly the already audited historical comparator subset. Source
    # bytes and reconstructed published medians must agree before reuse.
    previous_path = root / f'packed-study-{size}-b{batch}-compact-kv-all-baselines.json'
    previous = read(previous_path)
    common_keys = [tuple(k) for k in previous['group_keys']]
    assert len(common_keys) == previous['common_equal32_groups'] and len(set(common_keys)) == len(common_keys)
    assert all(k in warm_keys and all(n == 32 for n in reference[k]['sampled_tokens_including_stop']) for k in common_keys)
    competitor_values = {}
    comparator_sources = {}
    for name, entry in previous['sources'].items():
        if name in ('a1', 'a2', 'b1', 'b2', 'process_observation_audit'):
            continue
        path = root / entry['file']
        rows = read(path)
        assert sources[str(path)] == entry['sha256']
        indexed = {key(r): r for r in rows}
        assert len(rows) == len(indexed) == 28
        assert all(indexed[k]['input_sha256'] == reference[k]['input_sha256'] for k in common_keys)
        value = median(indexed[k]['wall_seconds'] * 1000 for k in common_keys)
        assert value == previous['median_group_ms'][name]
        competitor_values[name] = value
        comparator_sources[name] = {'file': entry['file'], 'sha256': entry['sha256']}
    common_values = {letter: median(aggregate[letter][k] for k in common_keys) for letter in ('a', 'b')}
    fastest = min(competitor_values, key=competitor_values.get)
    threshold = competitor_values[fastest] / 2
    result['models'].append({'model': size, 'batch': batch, 'warm_groups': 20, 'per_run': per_run,
        'all20_group_median_ms': values, 'all20_control_over_candidate_ratio': ratio,
        'both_repeat_medians_improve': repeats_improve, 'process_observation_qualified': qualified,
        'experimental_speed_screen_passed': ratio >= 1.05 and repeats_improve and qualified,
        'timing_provisional_due_to_process_observation': not qualified,
        'all20_groups': [{'trial': k[0], 'workload_group': k[1], 'control_ms': aggregate['a'][k], 'candidate_ms': aggregate['b'][k]} for k in warm_keys],
        'historical_common_equal32': {'group_keys': common_keys, 'groups': len(common_keys),
             'current_control_ms': common_values['a'], 'current_candidate_ms': common_values['b'],
             'saved_competitors_ms': competitor_values, 'saved_competitor_sources': comparator_sources,
             'fastest_tested_baseline': fastest, 'half_fastest_baseline_ms': threshold,
             'fastest_baseline_over_candidate_ratio': competitor_values[fastest] / common_values['b'],
             'latency_threshold_met': common_values['b'] <= threshold and qualified},
        'scope': 'Same28 signatures and original target profile. All20 paired repeats and historical15/14 equal32 subsets '
                 'are separate analyses. Observations are sampled; baselines not rerun. Intermediate arithmetic, API '
                 'and speculative settings differ. Owned allocation ranges are not process VRAM or lifetime peaks.'})
result['all_completed_raw_controls_passed'] = True
result['sources_sha256'] = sources
result['scope'] = 'Prospective ABBA step screen; no automatic production/default promotion, independent quality or full-goal acceptance.'
output = root / 'qwen35-shared-mtp-study-v1-summary.json'
assert not output.exists()
output.write_text(json.dumps(result, indent=2) + '\n')
for model in result['models']:
    print(model['model'], 'all20 ratio', model['all20_control_over_candidate_ratio'],
          'step_screen', model['experimental_speed_screen_passed'],
          'fastest_saved_ratio', model['historical_common_equal32']['fastest_baseline_over_candidate_ratio'])
