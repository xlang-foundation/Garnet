"""Real compact state controls versus noncompact target and saved checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('root', type=Path)
p.add_argument('identity', type=Path)
a = p.parse_args()
output = a.root / 'compact-kv-pretrained-v1-summary.json'
assert not output.exists(), 'Preserve pretrained proof'
sources = {}


def read(name):
    raw = (a.root / name).read_bytes()
    sources[name] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw.decode('utf-8-sig'))


epoch = read(a.identity.name)
epoch_sha = sources[a.identity.name]
transactions = read('compact-kv-transaction-v1-summary.json')
assert transactions['all_compact_verifier_session_profiles_passed']
assert transactions['exact_control_error_contracts_passed']
assert transactions['garnet_dll_sha256'].lower() == epoch['garnet_binary_sha256'].lower()
cases = []
for size, batch in (('2b', 1), ('4b', 8)):
    prior = read(f'internal-store-pretrained-v1-{size}-b{batch}.json')
    for mode in ('plain', 'memcheck'):
        tag = f'compact-kv-pretrained-{size}-b{batch}-{mode}-v1'
        terminal = read(tag + '-exit.json')
        result = read(tag + '.json')
        assert terminal['exit_code'] == 0 and terminal['execution_identity_sha256'].lower() == epoch_sha
        assert terminal['garnet_dll_sha256'].lower() == epoch['garnet_binary_sha256'].lower()
        assert result['batch'] == batch and result['state_hash_method'] == 'bounded_native_finite_bytes'
        assert result['native_profile']['compactKVOutputs'] is True
        assert result['ordinary_profile']['compactKVOutputs'] is False
        assert result['native_profile']['maxInputTokens'] + result['native_profile']['maxOutputTokens'] == 512
        assert len(result['trials']) == len(result['state_hashes']) == 2
        assert result['all_state_hashes_equal'] and result['same_ordinary_outputs']
        assert result['same_recorded_sequences'] and result['same_all_recorded_stop_signatures']
        assert result['same_all_recorded_discrete_mtp_work']
        assert result['input_sha256'] == prior['input_sha256']
        assert all(row == result['ordinary_state_hashes'] for row in result['state_hashes'])
        assert result['ordinary_state_hashes'] == prior['ordinary_state_hashes']
        assert all(row == prior['ordinary_state_hashes'] for row in prior['state_hashes'])
        memory = result['native_memory']['decoder']
        assert memory['compact_kv_outputs'] and memory['compact_state_commit_bytes'] > 0
        assert all(not row['qwen35_whole_program_cuda'] for row in memory['graphs'])
        if mode == 'memcheck':
            raw = (a.root / (tag + '.log')).read_bytes()
            sources[tag + '.log'] = hashlib.sha256(raw).hexdigest()
            assert b'ERROR SUMMARY: 0 errors' in raw
        cases.append({'model': size, 'batch': batch, 'mode': mode,
                      'two_native_requests_then_separate_noncompact_target': True,
                      'complete_finite_state_hashes_equal_current_ordinary_and_saved_prior': True,
                      'saved_inputs_tokens_stops_and_discrete_mtp_work_equal': True,
                      'native_profile': result['native_profile'], 'ordinary_profile': result['ordinary_profile']})
summary = {'sources_sha256': sources, 'garnet_dll_sha256': epoch['garnet_binary_sha256'],
           'all_four_pretrained_controls_passed': len(cases) == 4, 'cases': cases,
           'full28_workload_qualified': False, 'independent_quality_qualified': False,
           'primary_performance_eligible': False, 'process_memory_measured': False,
           'scope': 'One real-image group per model, two reset compact native MTP requests then a separately '
                    'loaded noncompact ordinary target. Complete finite canonical/hidden hashes also '
                    'match the saved earlier ordinary checkpoint. Limits/stops/discrete work preserved. '
                    'This is not independent vision/heldout quality, all28 inputs, every batch/profile, '
                    'process-memory evidence, primary latency or the 2x goal.'}
assert summary['all_four_pretrained_controls_passed']
output.write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps({'all_four_pretrained_controls_passed': True, 'full28_workload_qualified': False,
                  'process_memory_measured': False, 'primary_performance_eligible': False}))
