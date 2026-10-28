"""Compare compact retained owners and separate warmed WDDM process counters."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('root', type=Path)
p.add_argument('--model', choices=('2b', '4b'), required=True)
p.add_argument('--batch', type=int, choices=(1, 8), required=True)
p.add_argument('--attempt', default='v1')
a = p.parse_args()
output = a.root / f'compact-kv-memory-{a.model}-b{a.batch}-pair-summary.json'
assert not output.exists(), 'Preserve compact memory evidence'
sources, data, binaries, epochs = {}, {}, set(), set()


def read(path):
    raw = path.read_bytes()
    sources[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw.decode('utf-8-sig'))


for mode in ('control', 'candidate'):
    prefix = f'memory-compact-kv-{a.model}-b{a.batch}-{mode}-{a.attempt}'

    def part(ending):
        return read(a.root / (prefix + ending))

    terminal, owner = part('-exit.json'), part('-owner.json')
    assert terminal['model_exit_code'] == terminal['monitor_exit_code'] == 0
    assert terminal['garnet_dll_sha256'].lower() == owner['garnet_dll_sha256'].lower()
    assert terminal['execution_identity_sha256'].lower() == owner['execution_identity_sha256'].lower()
    binaries.add(owner['garnet_dll_sha256'].lower())
    epochs.add(owner['execution_identity_sha256'].lower())
    audit = part('-workload-control.json')
    assert audit['all_six_input_hash_groups_exact'] and audit['same_garnet_profile']
    assert audit['all_six_garnet_output_stop_signatures_match_timing']
    for source in audit['sources']:
        assert hashlib.sha256((a.root / source['file']).read_bytes()).hexdigest() == source['sha256']
    events, summary = part('-session.json'), part('-summary.json')
    assert events[-1]['event'] == 'complete'
    assert {e['process_id'] for e in events} == {owner['process_id']} == {summary['process_id']}
    assert not summary['ambiguous_nonzero_duplicate_samples']
    assert all(summary['phases'][name]['samples'] >= 3
               for name in ('steady_before', 'steady_after', 'after_model_close'))
    steady = next(e for e in events if e['event'] == 'steady_after_begin')
    enabled = mode == 'candidate'
    assert steady['profile'].get('compactKVOutputs', False) == enabled
    assert steady['profile']['visionCudaGraph'] is True
    assert steady['profile'].get('internalPackedStores', False) == (a.model == '4b')
    memory = steady['owned_memory']
    assert memory['decoder']['compact_kv_outputs'] == enabled
    helper_bytes = memory['decoder']['compact_state_commit_bytes']
    assert (helper_bytes > 0) if enabled else (helper_bytes == 0)
    phase = summary['phases']['steady_after']
    data[mode] = {
        'profile': steady['profile'], 'native_build_policy': steady['native_build_policy'],
        'adapter': summary['adapter'], 'sample_count': phase['samples'],
        'warm_process_counter_bytes': {f: phase['bytes'][f]['median']
            for f in ('DedicatedUsage', 'SharedUsage', 'TotalCommitted')},
        'retained_owned_model_bytes': memory['owned_model_bytes'],
        'retained_engine_output_bytes': memory['decoder']['engine_output_bytes'],
        'copy_helper_bytes': memory['decoder']['compact_state_commit_bytes'],
        'all_phase_sampled_statistics': summary['phases'],
        'largest_observed_bytes_not_true_peak': summary['largest_observed_bytes'],
    }
assert len(binaries) == len(epochs) == 1
left, right = dict(data['control']['profile']), dict(data['candidate']['profile'])
assert left.pop('compactKVOutputs', False) is False and right.pop('compactKVOutputs', False) is True
assert left == right and data['control']['native_build_policy'] == data['candidate']['native_build_policy']
lm = read(a.root / f'memory-lmstudio-{a.model}-b{a.batch}-512-partitioned-summary.json')
assert data['control']['adapter']['uuid'] == data['candidate']['adapter']['uuid'] == lm['adapter']['uuid']
saved = read(a.root / f'compact-kv-{a.model}-b{a.batch}-a1-b1-workload-control.json')
assert saved['all_passed'] and saved['all_28_checked']
expected_owners = {(r['old_owned_model_bytes'], r['compact_owned_model_bytes']) for r in saved['ownership']}
assert expected_owners == {(data['control']['retained_owned_model_bytes'], data['candidate']['retained_owned_model_bytes'])}
deltas = {f: data['control']['warm_process_counter_bytes'][f] - data['candidate']['warm_process_counter_bytes'][f]
          for f in data['control']['warm_process_counter_bytes']}
owned_delta = data['control']['retained_owned_model_bytes'] - data['candidate']['retained_owned_model_bytes']
assert owned_delta > 0
result = {
    'model': a.model, 'batch': a.batch, 'attempt': a.attempt, 'garnet_dll_sha256': next(iter(binaries)),
    'execution_identity_sha256': next(iter(epochs)), 'same_gpu_uuid_as_saved_lmstudio': lm['adapter']['uuid'],
    'sources_sha256': sources, 'profiles': data, 'control_minus_candidate_counter_bytes': deltas,
    'control_minus_candidate_retained_owned_bytes': owned_delta,
    'candidate_committed_counter_not_greater': deltas['TotalCommitted'] >= 0,
    'primary_performance_eligible': False, 'physical_residency_measured': False,
    'scope': 'Sequential same-DLL compact-only profiles. Six exact input/output/stop/profile memory '
        'groups match separately completed full28 timing cohorts, and retained owners match their '
        'explicit allocation audits. WDDM dedicated/shared/committed phase medians are separate '
        'counters on the same GPU UUID as saved LM Studio. Largest observation is not a true '
        'peak or physical residency. Polling and phase holds invalidate latency comparisons. '
        'Foreign-process coverage is reported separately by the observed-memory parent.',
}
output.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({k: result[k] for k in ('model', 'batch', 'control_minus_candidate_counter_bytes',
                                      'control_minus_candidate_retained_owned_bytes')}))
