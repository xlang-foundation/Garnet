"""Audit the complete integrated fixture grid without model/speed promotion."""
import argparse
import hashlib
import itertools
import json
import re
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('root', type=Path)
a = p.parse_args()
output = a.root / 'compact-kv-fixture-matrix-v1-summary.json'
assert not output.exists(), 'Preserve integrated fixture proof'
sources = {}


def read(name):
    raw = (a.root / name).read_bytes()
    sources[name] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw.decode('utf-8-sig'))


epoch_names = ('compact-kv-fixture-v1-before-run-identity.json',
               'compact-kv-fixture-matrix-v1-before-run-identity.json')
epochs = [read(name) for name in epoch_names]
assert epochs[0]['garnet_binary_sha256'] == epochs[1]['garnet_binary_sha256']
assert epochs[0]['xlang_binary_sha256'] == epochs[1]['xlang_binary_sha256']
assert epochs[0]['xlang_runtime_dll_sha256'] == epochs[1]['xlang_runtime_dll_sha256']
for name, digest in epochs[0]['source_sha256'].items():
    assert epochs[1]['source_sha256'][name] == digest, 'Consumed core source changed across fixture cases'
phases = ['prefill', 'inactive_gaps'] + ['step' + str(i) for i in range(8)]
phases += ['reset_prefill', 'capacity_tail', 'inactive_at_capacity']
negative = {'gpu_invalid_control_blocks_commit_and_requires_reset',
            'past_capacity_rejected_without_state_write', 'schema_state_width', 'schema_packed_width',
            'schema_output_state_width', 'schema_status_offset', 'schema_output_interval', 'canonical_alias_rejected'}
records = []
sanitizers = []
for batch, capacity, internal in itertools.product((1, 2, 4, 8), (384, 512), (0, 1)):
    first = (batch, capacity, internal) == (1, 384, 0)
    attempt = 'v1' if first else 'matrix-v1'
    epoch_name = epoch_names[0 if first else 1]
    modes = ('plain', 'memcheck', 'racecheck', 'synccheck') if first or (batch, capacity, internal) == (8, 512, 1) else ('plain',)
    for mode in modes:
        tag = f'compact-kv-fixture-b{batch}-c{capacity}-i{internal}-{mode}-{attempt}'
        terminal = read(tag + '-exit.json')
        data = read(tag + '.json')
        assert terminal['exit_code'] == 0
        assert terminal['execution_identity_sha256'].lower() == sources[epoch_name]
        assert terminal['garnet_dll_sha256'].lower() == epochs[0]['garnet_binary_sha256'].lower()
        assert data['all_passed'] and data['batch'] == batch and data['capacity'] == capacity
        assert data['internal_stores'] == bool(internal) and len(data['runs']) == 2
        control, candidate = data['runs']
        assert control['compact'] is False and candidate['compact'] is True
        assert control['snapshots'] == candidate['snapshots']
        assert [row['phase'] for row in candidate['snapshots']] == phases
        assert control['closed_state_sha256'] == candidate['closed_state_sha256']
        assert control['closed_hidden_sha256'] == candidate['closed_hidden_sha256']
        assert set(candidate['negative_checks']) == negative
        old, new = control['owned_memory'], candidate['owned_memory']
        assert old['projection_weight_pool_bytes'] == new['projection_weight_pool_bytes']
        assert old['state_bytes'] == new['state_bytes']
        assert not old['compact_kv_outputs'] and new['compact_kv_outputs']
        assert len(new['compact_state_commit_helpers']) == 2
        assert all(row['ready'] and row['copy_only'] and row['compact_kv_outputs'] for row in new['compact_state_commit_helpers'])
        assert new['compact_state_commit_bytes'] > 0
        assert old['engine_output_bytes'] > new['engine_output_bytes']
        if mode == 'plain':
            records.append({'batch': batch, 'capacity': capacity, 'internal_stores': bool(internal),
                            'complete_snapshot_count': 13, 'all_finite_state_hidden_prediction_bytes_exact': True,
                            'candidate_negative_checks': candidate['negative_checks'],
                            'control_output_bytes': old['engine_output_bytes'],
                            'compact_output_bytes': new['engine_output_bytes'],
                            'compact_helper_bytes': new['compact_state_commit_bytes']})
        else:
            raw = (a.root / (tag + '.log')).read_bytes()
            sources[tag + '.log'] = hashlib.sha256(raw).hexdigest()
            text = raw.decode('utf-8-sig')
            pattern = (r'RACECHECK SUMMARY:\s+0 hazards displayed \(0 errors, 0 warnings\)'
                       if mode == 'racecheck' else r'ERROR SUMMARY:\s+0 errors')
            assert re.search(pattern, text)
            sanitizers.append({'batch': batch, 'capacity': capacity, 'internal_stores': bool(internal),
                               'mode': mode, 'unfiltered_zero_errors_or_hazards': True})
summary = {'sources_sha256': sources, 'garnet_dll_sha256': epochs[0]['garnet_binary_sha256'],
           'all_16_fixture_profiles_passed': len(records) == 16, 'profiles': records,
           'six_min_max_format_sanitizer_cases_passed': len(sanitizers) == 6, 'sanitizers': sanitizers,
           'speculative_prefix_rollback_qualified': False, 'pretrained_model_qualified': False,
           'readonly_cross_device_contract_qualified': False,
           'primary_performance_eligible': False, 'process_memory_measured': False,
           'scope': 'Sequential original/compact fixture models; full finite canonical/hidden bytes and '
                    'predictions, inactive gaps, reset, capacity tails, pending close, GPU control-error '
                    'guard/poison/reset and malformed schema/alias checks. All16 plain profiles plus '
                    'min/max/both-format memcheck/racecheck/synccheck. Synthetic fixture quality and '
                    'owned buffers do not establish pretrained/speculative correctness, every ownership '
                    'contract, process VRAM, latency or the 2x goal.'}
assert summary['all_16_fixture_profiles_passed'] and summary['six_min_max_format_sanitizer_cases_passed']
output.write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps({'all_16_fixture_profiles_passed': True, 'sanitizer_cases': len(sanitizers),
                  'pretrained_model_qualified': False, 'process_memory_measured': False}))
