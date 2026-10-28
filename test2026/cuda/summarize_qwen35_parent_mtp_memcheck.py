"""Audit unfiltered parent-pool integration memory checks and plain parity."""
import hashlib
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
identity_path = root / 'qwen35-parent-mtp-memcheck-v1-before-run-identity.json'
identity = read(identity_path)
epoch = digest(identity_path)
result = {'all_memcheck_controls_passed': False, 'profiles': [], 'execution_identity_sha256': epoch,
          'independent_quality_proven': False, 'process_memory_measured': False, 'latency_measured': False,
          'production_promoted': False}
keys = ('prompt_token_ids', 'generated_token_ids', 'sampled_tokens_including_stop', 'finish_reasons', 'stop_token_ids', 'visual_tokens_per_row')
for model, batch in [('2b', 1), ('4b', 8)]:
    tag = f'qwen35-parent-mtp-{model}-b{batch}-memcheck-v1'
    raw_path = root / (tag + '.json')
    terminal = read(root / (tag + '-exit.json'))
    assert terminal['exit_code'] == 0 and terminal['execution_identity_sha256'].lower() == epoch.lower()
    assert terminal['garnet_dll_sha256'].lower() == identity['garnet_binary_sha256'].lower()
    log_path = root / (tag + '.log')
    log = log_path.read_text(errors='replace')
    assert re.findall(r'ERROR SUMMARY:\s*(\d+)\s+errors?', log) == ['0']
    assert '========= ERROR: ' not in log
    raw = read(raw_path)
    plain = read(root / f'qwen35-parent-mtp-{model}-b{batch}-plain-v1.json')
    assert raw['all_passed'] and plain['all_passed'] and len(raw['records']) == len(plain['records']) == 8
    for row in raw['records']:
        ref = next(r for r in plain['records'] if (r['style'], r['trial']) == (row['style'], row['trial']))
        assert all(row['result'][k] == ref['result'][k] for k in keys)
        assert row['state'] == ref['state']
        assert row['result']['speculative_metrics'] == ref['result']['speculative_metrics']
    for name in ('session_poison_guard', 'target_poison_guard', 'pending_guard'):
        assert raw['lifecycle'][name]['full_finite_protected_words_unchanged']
        assert raw['lifecycle'][name]['mtp_graph_counts_and_work_unchanged']
    life = raw['lifecycle']
    assert life['same_python_parent_root_objects'] and life['rollback_restores_all_target_and_mtp_words_except_disposable_packed_output']
    assert life['pending_close_rolls_back_and_retained_output_provider_survives']['finite_fp32_words'] == plain['lifecycle']['pending_close_rolls_back_and_retained_output_provider_survives']['finite_fp32_words']
    assert life['closed_call_error']
    replay, ref = life['reset_after_rollback'], plain['lifecycle']['reset_after_rollback']
    assert all(replay['result'][k] == ref['result'][k] for k in keys)
    assert replay['state'] == ref['state'] and replay['result']['speculative_metrics'] == ref['result']['speculative_metrics']
    assert raw['memory']['owned_model_bytes'] == plain['memory']['owned_model_bytes']
    assert raw['removed_exclusive_pool_bytes'] == plain['removed_exclusive_pool_bytes']
    result['profiles'].append({'model': model, 'batch': batch, 'error_summary': 0,
        'full_plain_state_signature_work_matches': 8, 'lifecycle_gates_passed': True,
        'raw_sha256': digest(raw_path), 'log_sha256': digest(log_path),
        'removed_owned_pool_bytes_vs_validated_split': raw['removed_exclusive_pool_bytes']})
result['all_memcheck_controls_passed'] = True
result['scope'] = ('Unfiltered integration memcheck, exact plain controls and lifecycle guards only. No racecheck/synccheck '
                   'or concurrent-request, all-batch/capacity, independent quality, WDDM/residency/peak or latency acceptance.')
output = root / 'qwen35-parent-mtp-memcheck-v1-summary.json'
assert not output.exists()
output.write_text(json.dumps(result, indent=2) + '\n')
print('PARENT_MTP_MEMCHECK_AUDIT_PASS', len(result['profiles']))
