"""Audit explicit compact verifier/session profiles, without speed promotion."""
import argparse
import hashlib
import itertools
import json
import re
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('root', type=Path)
p.add_argument('identity', type=Path)
a = p.parse_args()
output = a.root / 'compact-kv-transaction-v1-summary.json'
assert not output.exists(), 'Preserve compact transaction evidence'
sources = {}


def read(path):
    raw = path.read_bytes()
    sources[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw.decode('utf-8-sig'))


identity = read(a.identity)
epoch = sources[str(a.identity)]
fixture = read(a.root / 'compact-kv-fixture-matrix-v1-summary.json')
assert fixture['all_16_fixture_profiles_passed'] and fixture['six_min_max_format_sanitizer_cases_passed']
assert fixture['garnet_dll_sha256'].lower() == identity['garnet_binary_sha256'].lower()
profiles = [('verifier', b, i, 64) for b, i in itertools.product((1, 2, 4, 8), (0, 1))]
profiles += [('session', b, i, t) for b, i, t in itertools.product((1, 2, 4, 8), (0, 1), (32, 64))]
profiles += [('errors', b, 1, 64) for b in (1, 8)]
sanitizer_profiles = []
for b, i in ((1, 0), (8, 1)):
    sanitizer_profiles += [('verifier', b, i, 64, mode) for mode in ('memcheck', 'racecheck', 'synccheck')]
    sanitizer_profiles += [('session', b, i, 64, 'memcheck'), ('errors', b, 1, 64, 'memcheck')]
records, sanitizers = [], []
expected_deferred = {
    'canonical_unchanged_until_commit', 'parent_reset_step_close_blocked_while_borrowed',
    'unsupported_verifier_rejected', 'recoverable_verify_and_partial_read_faults_rollback',
    'unknown_final_write_poisons_until_reset', 'unknown_partial_native_write_poisons_until_reset',
    'native_invalid_spans_and_layouts_rejected_before_mutation'}
expected_errors = {'gpu_active2_status1_no_write_poison', 'last_row_status_blocks_copy_commit',
                   'last_row_status_blocks_zero_prefix_replay', 'duplicate_active_slot_no_write'}
for kind, batch, internal, prefill, mode in [(*row, 'plain') for row in profiles] + sanitizer_profiles:
    tag = f'compact-kv-{kind}-b{batch}-i{internal}-p{prefill}-k1-{mode}-v1'
    data = read(a.root / (tag + '.json'))
    terminal = read(a.root / (tag + '-exit.json'))
    assert terminal['exit_code'] == 0 and data['all_passed'] and data['batch'] == batch
    assert terminal['execution_identity_sha256'].lower() == epoch
    assert terminal['garnet_dll_sha256'].lower() == identity['garnet_binary_sha256'].lower()
    assert (terminal['kind'], terminal['batch'], terminal['internal'], terminal['prefill'],
            terminal['compact'], terminal['mode']) == (kind, batch, internal, prefill, 1, mode)
    if kind == 'verifier':
        assert data['chunk'] == data['verification_width'] == 4 and len(data['checks']) == 6
        assert all(row['states_bitwise_equal'] and row['predictions_equal'] for row in data['checks'])
        assert set(data['deferred_checks']) == expected_deferred and all(data['deferred_checks'].values())
        assert data['invalid_prefix_token_control_capacity_and_close_rollback']
        native = data['memory']['native_prefix_commit']
        assert native['ready'] and native['compact_kv_outputs'] and not native['copy_only']
    elif kind == 'session':
        assert data['prefill_chunk'] == prefill and data['chunk'] == 4 and data['draft_depth'] == 3
        assert len(data['records']) == 24 and all(row['all_equal'] for row in data['records'])
        assert {row['proposal_mode'] for row in data['records']} == {'mtp', 'oracle_test_only'}
        assert {row['limit'] for row in data['records']} == {1, 2, 7, 8}
        assert data['target_memory']['compact_kv_outputs']
        native = data['memory']['target_verifier']['native_prefix_commit']
        assert native['ready'] and native['compact_kv_outputs'] and not native['copy_only']
    else:
        assert {row['case'] for row in data['checks']} == expected_errors
        assert data['published_invalid_status'] and set(data['published_invalid_status']) == {1.0}
        assert all('Compact KV control error' in row['message'] for row in data['checks']
                   if row['case'] != 'duplicate_active_slot_no_write')
    record = {'kind': kind, 'batch': batch, 'internal_stores': bool(internal),
              'prefill': prefill, 'mode': mode, 'all_original_checks_passed': True}
    if mode == 'plain':
        records.append(record)
    else:
        raw = (a.root / (tag + '.log')).read_bytes()
        sources[tag + '.log'] = hashlib.sha256(raw).hexdigest()
        pattern = (r'RACECHECK SUMMARY:\s+0 hazards displayed \(0 errors, 0 warnings\)'
                   if mode == 'racecheck' else r'ERROR SUMMARY:\s+0 errors')
        assert re.search(pattern, raw.decode('utf-8-sig'))
        sanitizers.append(record)
summary = {'sources_sha256': sources, 'garnet_dll_sha256': identity['garnet_binary_sha256'],
           'all_compact_verifier_session_profiles_passed': len(records) == 26,
           'exact_control_error_contracts_passed': True,
           'plain_profiles': records, 'ten_selected_unfiltered_sanitizer_cases_passed': len(sanitizers) == 10,
           'sanitizers': sanitizers, 'original_default_regressions_qualified': False,
           'pretrained_model_qualified': False, 'readonly_cross_device_contract_qualified': False,
           'primary_performance_eligible': False, 'process_memory_measured': False,
           'scope': '8 compact verifier and16 compact session profiles acrossB1/2/4/8 and ordinary/'
                    'internal output formats; sessions P32/P64, D4/MTP3. Original assertions retained '
                    'with explicit test-only oracle proposals for branch coverage. Two exact error '
                    'profiles. Selected10 unfiltered sanitizer cases, not every profile/tool. '
                    'Synthetic correctness does not prove independent quality, pretrained equivalence, '
                    'all ownership contracts, real-model memory, latency or the2x goal.'}
assert summary['all_compact_verifier_session_profiles_passed']
assert summary['ten_selected_unfiltered_sanitizer_cases_passed']
output.write_text(json.dumps(summary, indent=2), encoding='utf-8')
print('QWEN35_COMPACT_TRANSACTIONS_AUDIT_PASS', len(records), len(sanitizers))
