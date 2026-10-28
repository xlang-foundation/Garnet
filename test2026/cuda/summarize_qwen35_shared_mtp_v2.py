"""Audit raw shared-MTP controls after every resident model has closed."""
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
identity_path = root / 'qwen35-shared-mtp-v2-before-run-identity.json'
identity = read(identity_path)
epoch = digest(identity_path)
design = read(root / 'qwen35-shared-mtp-v1-prospective-design.json')
result = {'all_plain_controls_passed': False, 'profiles': [], 'execution_identity_sha256': epoch,
          'independent_quality_proven': False, 'performance_or_process_memory_measurement': False,
          'production_change_or_promotion': False}
keys = ('prompt_token_ids', 'generated_token_ids', 'sampled_tokens_including_stop', 'finish_reasons',
        'stop_token_ids', 'visual_tokens_per_row')
for profile in design['profiles']:
    stem = f"qwen35-shared-mtp-{profile['size']}-b{profile['batch']}-v2"
    path = root / (stem + '.json')
    terminal = read(root / (stem + '-exit.json'))
    assert terminal['exit_code'] == 0 and terminal['execution_identity_sha256'].lower() == epoch.lower()
    assert terminal['garnet_dll_sha256'].lower() == identity['garnet_binary_sha256'].lower()
    assert terminal['gpu_uuid'] == design['gpu_uuid']
    raw = read(path)
    assert raw['all_passed'] and not raw['performance_or_process_memory_measurement']
    assert [s['mode'] for s in raw['stages']] == ['baseline', 'pool', 'split', 'ordinary']
    base = raw['stages'][0]
    assert len(base['records']) == 8
    comparisons = 0
    for stage in raw['stages']:
        assert len(stage['records']) == 8
        for row in stage['records']:
            ref = next(x for x in base['records'] if (x['style'], x['trial']) == (row['style'], row['trial']))
            assert all(ref['result'][k] == row['result'][k] for k in keys)
            assert all(ref['state'][k] == row['state'][k] for k in ('last_hidden', 'canonical'))
            assert all(s['finite_fp32_words'] > 0 and len(s['sha256']) == 64 for s in row['state'].values())
            if stage['mode'] == 'pool':
                assert ref['state'] == row['state'] and ref['result']['speculative_metrics'] == row['result']['speculative_metrics']
            first = next(x for x in stage['records'] if x['style'] == row['style'] and x['trial'] == 0)
            assert first['state'] == row['state']
            if stage['mode'] != 'ordinary':
                assert first['result']['speculative_metrics'] == row['result']['speculative_metrics']
            comparisons += 1
    events = raw['events']
    for before, after in [('baseline', 'pool'), ('pool', 'split'), ('split', 'ordinary')]:
        assert next(e['unix_seconds'] for e in events if e['name'] == before + '_closed') <= next(e['unix_seconds'] for e in events if e['name'] == after + '_load_begin')
    budgets = []
    for stage in raw['stages'][1:3]:
        m = stage['memory']
        s = m['speculative']
        weight_bytes = s['mtp_shared_projection_bytes_in_parent_io_pool']
        assert m['decoder']['projection_weight_pool_bytes'] == base['memory']['decoder']['projection_weight_pool_bytes'] + weight_bytes
        assert m['decoder']['projection_weight_pool_entries'] == base['memory']['decoder']['projection_weight_pool_entries'] + 8
        graphs = s['mtp_experiment_graphs']
        assert set(graphs) == ({'4'} if stage['mode'] == 'pool' else {'1', '4', '16'})
        for g in graphs.values():
            assert g['external_bf16_weight_inputs'] == 8 and g['external_bf16_weight_bytes'] == weight_bytes
            assert g['external_output_storage'] and g['external_workspace_bound']
            assert g['owned_device_output_bytes'] == 0 and g['allocator_tracking_enabled']
            assert g['external_workspace_capacity_bytes'] == s['mtp_shared_activation_bytes']
        assert s['mtp_shared_activation_bytes'] == ((max(g['context_workspace_required_bytes'] for g in graphs.values()) + 3) // 4) * 4
        budgets.append({'mode': stage['mode'], 'owned_model_bytes': m['owned_model_bytes'],
                        'baseline_owned_model_bytes': base['memory']['owned_model_bytes'],
                        'owned_model_delta_bytes': m['owned_model_bytes'] - base['memory']['owned_model_bytes'],
                        'shared_mtp_projection_bytes_counted_once': weight_bytes,
                        'mtp_shared_activation_bytes': s['mtp_shared_activation_bytes'],
                        'mtp_shared_output_bytes': s['mtp_shared_output_bytes'],
                        'mtp_prefill_calls': [r['result']['speculative_metrics']['mtp_prefill_calls'] for r in stage['records']],
                        'target_calls': [r['result']['speculative_metrics']['target_verification_calls'] for r in stage['records']]})
    result['profiles'].append({'model': profile['size'], 'batch': profile['batch'], 'raw_sha256': digest(path),
                               'plain_signature_and_complete_target_state_comparisons': comparisons, 'budgets': budgets})
result['all_plain_controls_passed'] = True
result['scope'] = ('Matched plain real-image target-state/output/reset controls only. Pools are retained native accounting, '
                   'not WDDM/process peak. No independent MTP numerical suite, sanitizer, full-capacity, all-batch or speed acceptance.')
output = root / 'qwen35-shared-mtp-v2-summary.json'
assert not output.exists()
output.write_text(json.dumps(result, indent=2) + '\n')
print('SHARED_MTP_PLAIN_AUDIT_PASS', len(result['profiles']))
