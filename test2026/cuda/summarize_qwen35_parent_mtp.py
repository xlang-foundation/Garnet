"""Audit native state/lifetime/owned-pool evidence after both models close."""
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
identity_path = root / 'qwen35-parent-mtp-v1-before-run-identity.json'
identity = read(identity_path)
epoch = digest(identity_path)
output = root / 'qwen35-parent-mtp-v1-summary.json'
assert not output.exists()
result = {'native_plain_controls_passed': False, 'profiles': [], 'execution_identity_sha256': epoch,
          'independent_quality_proven': False, 'process_memory_measured': False, 'latency_measured': False,
          'production_promoted': False}
keys = ('prompt_token_ids', 'generated_token_ids', 'sampled_tokens_including_stop', 'finish_reasons', 'stop_token_ids', 'visual_tokens_per_row')
for size, batch in [('2b', 1), ('4b', 8)]:
    tag = f'qwen35-parent-mtp-{size}-b{batch}-plain-v1'
    raw_path = root / (tag + '.json')
    terminal = read(root / (tag + '-exit.json'))
    assert terminal['exit_code'] == 0 and terminal['execution_identity_sha256'].lower() == epoch.lower()
    assert terminal['garnet_dll_sha256'].lower() == identity['garnet_binary_sha256'].lower()
    raw = read(raw_path)
    assert raw['all_passed'] and digest(Path(raw['plain_control'])) == raw['plain_sha256']
    old = read(Path(raw['plain_control']))
    control = next(s for s in old['stages'] if s['mode'] == 'split')
    ordinary = next(s for s in old['stages'] if s['mode'] == 'ordinary')
    assert len(raw['records']) == 8
    for row in raw['records']:
        ref = next(r for r in control['records'] if (r['style'], r['trial']) == (row['style'], row['trial']))
        target = next(r for r in ordinary['records'] if (r['style'], r['trial']) == (row['style'], row['trial']))
        assert all(row['result'][k] == ref['result'][k] == target['result'][k] for k in keys)
        assert row['state'] == ref['state']
        assert row['result']['speculative_metrics'] == ref['result']['speculative_metrics']
        assert all(row['state'][k] == target['state'][k] for k in ('last_hidden', 'canonical'))
    life = raw['lifecycle']
    for name in ('session_poison_guard', 'target_poison_guard', 'pending_guard'):
        assert life[name]['full_finite_protected_words_unchanged'] and life[name]['mtp_graph_counts_and_work_unchanged']
    assert life['same_python_parent_root_objects'] and life['rollback_restores_all_target_and_mtp_words_except_disposable_packed_output']
    assert life['pending_close_rolls_back_and_retained_output_provider_survives']['finite_fp32_words'] > 0
    assert life['closed_call_error']
    replay = life['reset_after_rollback']
    ref = control['records'][0]
    assert all(replay['result'][k] == ref['result'][k] for k in keys)
    assert replay['state'] == ref['state'] and replay['result']['speculative_metrics'] == ref['result']['speculative_metrics']
    memory, prior = raw['memory'], control['memory']
    decoder, mtp = memory['decoder'], memory['speculative']
    removed = prior['speculative']['mtp_shared_activation_bytes'] + prior['speculative']['mtp_shared_output_bytes']
    assert memory['owned_model_bytes'] == prior['owned_model_bytes'] - removed
    assert decoder['owned_decoder_bytes'] == prior['decoder']['owned_decoder_bytes']
    assert mtp['mtp_owned_activation_pool_bytes'] == mtp['mtp_owned_output_pool_bytes'] == 0
    assert mtp['mtp_borrowed_workspace_capacity_bytes'] == decoder['shared_activation_pool_bytes']
    assert mtp['mtp_borrowed_output_capacity_bytes'] == decoder['shared_output_pool_bytes']
    assert mtp['mtp_parent_storage'] and removed == raw['removed_exclusive_pool_bytes']
    result['profiles'].append({'model': size, 'batch': batch, 'full_request_state_work_matches': 8,
        'raw_sha256': digest(raw_path), 'removed_owned_pool_bytes_vs_validated_split': removed,
        'old_split_owned_model_bytes': prior['owned_model_bytes'], 'parent_backed_owned_model_bytes': memory['owned_model_bytes'],
        'owned_delta_vs_original_mtp_d4_bytes': memory['owned_model_bytes'] - old['stages'][0]['memory']['owned_model_bytes'],
        'lifecycle_gates_passed': True})
result['native_plain_controls_passed'] = True
result['scope'] = ('Plain native guards/rollback/close/provider/state and retained-owner ledger only. No sanitizer, '
                   'concurrent-request, all-batch/capacity, independent quality, WDDM/residency/peak or latency acceptance.')
output.write_text(json.dumps(result, indent=2) + '\n')
print('PARENT_MTP_PLAIN_AUDIT_PASS', len(result['profiles']))
