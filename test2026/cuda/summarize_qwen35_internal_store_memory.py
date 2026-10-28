"""Pair warmed Windows process counters for opt-in internal packed stores.

The model and monitor run separately from primary timing. These are sampled
committed process counters, not lifetime peaks or physical GPU residency.
"""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('root', type=Path)
p.add_argument('--model', choices=('2b', '4b'), required=True)
p.add_argument('--batch', type=int, choices=(1, 2, 4, 8), required=True)
a = p.parse_args()
output = a.root / f'internal-stores-memory-{a.model}-b{a.batch}-pair-summary.json'
assert not output.exists(), 'Preserve memory pair evidence'
sources = []
data = {}
binaries = set()


def read(path):
    blob = path.read_bytes()
    sources.append({'file': path.name, 'sha256': hashlib.sha256(blob).hexdigest()})
    return json.loads(blob)


for mode in ('control', 'candidate'):
    prefix = f'memory-internal-stores-{a.model}-b{a.batch}-{mode}'

    def part(ending):
        return read(a.root / (prefix + ending))

    terminal = part('-exit.json')
    assert terminal['model_exit_code'] == terminal['monitor_exit_code'] == 0
    owner = part('-owner.json')
    binaries.add(owner['garnet_dll_sha256'].lower())
    workload = part('-workload-control.json')
    assert workload['all_six_input_hash_groups_exact']
    assert workload['all_six_garnet_output_stop_signatures_match_timing']
    assert workload['same_garnet_profile']
    for source in workload['sources']:
        assert hashlib.sha256((a.root / source['file']).read_bytes()).hexdigest() == source['sha256']
    events = part('-session.json')
    summary = part('-summary.json')
    phase = summary['phases']['steady_after']
    assert phase['samples'] >= 3
    steady = next(row for row in events if row['event'] == 'steady_after_begin')
    memory = steady['owned_memory']['decoder']
    pool = memory['shared_activation_pool_bytes']
    bodies = memory['graphs'][:-1]
    layers = 24 if a.model == '2b' else 32
    internal = mode == 'candidate'
    assert steady['profile'].get('internalPackedStores', False) == internal
    assert steady['profile']['sharePackedActivationStorage'] is True
    assert pool > 0 and all(g['user_managed_workspace'] and g['external_workspace_bound']
                           and g['external_workspace_capacity_bytes'] == pool for g in bodies)
    assert sum(g['internal_packed_stores'] for g in bodies) == (layers * len(bodies) if internal else 0)
    assert sum(g['internal_packed_root_fills'] for g in bodies) == (len(bodies) if internal else 0)
    assert all(not g['network_output_aliases_input'] for g in memory['graphs'])
    assert any(row['event'] == 'complete' for row in events)
    keys = ('DedicatedUsage', 'SharedUsage', 'TotalCommitted')
    data[mode] = {
        'profile': dict(steady['profile']),
        'adapter': summary['adapter'],
        'native_build_policy': steady['native_build_policy'],
        'sample_count': phase['samples'],
        'warm_process_counter_bytes': {key: phase['bytes'][key]['median'] for key in keys},
        'retained_owned_model_bytes': steady['owned_memory']['owned_model_bytes'],
        'shared_activation_pool_bytes': pool,
        'steady_after_sampled_statistics': phase['bytes'],
        'all_phase_sampled_statistics': summary['phases'],
    }
assert len(binaries) == 1
left = dict(data['control']['profile'])
right = dict(data['candidate']['profile'])
assert left.pop('internalPackedStores', False) is False
assert right.pop('internalPackedStores', False) is True
assert left == right, 'Other memory profile fields changed'
assert data['control']['native_build_policy'] == data['candidate']['native_build_policy']
lm = read(a.root / f'memory-lmstudio-{a.model}-b{a.batch}-512-partitioned-summary.json')
assert data['control']['adapter']['uuid'] == data['candidate']['adapter']['uuid'] == lm['adapter']['uuid'], 'Memory counters use different GPU devices'
delta = {key: data['control']['warm_process_counter_bytes'][key]
         - data['candidate']['warm_process_counter_bytes'][key]
         for key in data['control']['warm_process_counter_bytes']}
result = {
    'model': a.model, 'batch': a.batch, 'garnet_dll_sha256': next(iter(binaries)),
    'sources': sources, 'profiles': data,
    'same_gpu_uuid_as_saved_lmstudio': lm['adapter']['uuid'],
    'control_minus_candidate_counter_bytes': delta,
    'control_minus_candidate_retained_owned_bytes': data['control']['retained_owned_model_bytes']
                                                  - data['candidate']['retained_owned_model_bytes'],
    'scope': 'Sequential same-build control/candidate memory processes, six matched real-image groups and '
             'exact timing profile/output/stop controls. Dedicated/shared/total have independent medians. '
             'All phase statistics retain observed maxima, not lifetime peak or physical residency. '
             'These diagnostics cannot establish primary latency, independent quality or causal significance.',
}
output.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({key: value for key, value in result.items() if key not in ('sources', 'profiles')}, indent=2))
