"""Same-profile warmed Windows counters for vision graph replay, diagnostic only."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('root', type=Path)
p.add_argument('--model', choices=('2b', '4b'), required=True)
p.add_argument('--batch', type=int, choices=(1, 2, 4, 8), required=True)
a = p.parse_args()
output = a.root / f'vision-graph-memory-{a.model}-b{a.batch}-pair-summary.json'
assert not output.exists(), 'Preserve vision memory evidence'
sources = []
data = {}
binaries = set()


def read(path):
    blob = path.read_bytes()
    sources.append({'file': path.name, 'sha256': hashlib.sha256(blob).hexdigest()})
    return json.loads(blob)


for mode in ('control', 'candidate'):
    prefix = f'memory-vision-graph-{a.model}-b{a.batch}-{mode}'

    def part(ending):
        return read(a.root / (prefix + ending))

    terminal = part('-exit.json')
    assert terminal['model_exit_code'] == terminal['monitor_exit_code'] == 0
    owner = part('-owner.json')
    binaries.add(owner['garnet_dll_sha256'].lower())
    workload = part('-workload-control.json')
    assert workload['all_six_input_hash_groups_exact']
    assert workload['all_six_garnet_output_stop_signatures_match_timing'] and workload['same_garnet_profile']
    for source in workload['sources']:
        assert hashlib.sha256((a.root / source['file']).read_bytes()).hexdigest() == source['sha256']
    events = part('-session.json')
    summary = part('-summary.json')
    phase = summary['phases']['steady_after']
    assert phase['samples'] >= 3 and events[-1]['event'] == 'complete'
    steady = next(row for row in events if row['event'] == 'steady_after_begin')
    enabled = mode == 'candidate'
    assert steady['profile'].get('visionCudaGraph', False) == enabled
    assert steady['profile'].get('internalPackedStores', False) == (a.model == '4b')
    vision = steady['owned_memory']['vision']
    assert vision['cuda_graph_requested'] == enabled
    assert steady['owned_memory']['vision_input_buffer_bytes'] == (1474560 if enabled else 0)
    if enabled:
        assert vision['serial_graph_reuse_requested'] and vision['contexts'] == 1
        assert vision['graph_captures'] == 1 and vision['graph_replays'] > 0 and vision['graph_capture_failures'] == 0
    else:
        assert vision['graph_captures'] == 0
    data[mode] = {
        'profile': dict(steady['profile']), 'native_build_policy': steady['native_build_policy'],
        'adapter': summary['adapter'], 'sample_count': phase['samples'],
        'warm_process_counter_bytes': {key: phase['bytes'][key]['median']
                                     for key in ('DedicatedUsage', 'SharedUsage', 'TotalCommitted')},
        'retained_owned_model_bytes': steady['owned_memory']['owned_model_bytes'],
        'vision_input_buffer_bytes': steady['owned_memory']['vision_input_buffer_bytes'],
        'vision_stats': vision, 'steady_after_sampled_statistics': phase['bytes'],
        'all_phase_sampled_statistics': summary['phases'],
    }
assert len(binaries) == 1
left, right = dict(data['control']['profile']), dict(data['candidate']['profile'])
assert left.pop('visionCudaGraph', False) is False and right.pop('visionCudaGraph', False) is True
assert left == right and data['control']['native_build_policy'] == data['candidate']['native_build_policy']
lm = read(a.root / f'memory-lmstudio-{a.model}-b{a.batch}-512-partitioned-summary.json')
assert data['control']['adapter']['uuid'] == data['candidate']['adapter']['uuid'] == lm['adapter']['uuid']
delta = {key: data['control']['warm_process_counter_bytes'][key] - data['candidate']['warm_process_counter_bytes'][key]
         for key in data['control']['warm_process_counter_bytes']}
result = {
    'model': a.model, 'batch': a.batch, 'garnet_dll_sha256': next(iter(binaries)),
    'same_gpu_uuid_as_saved_lmstudio': lm['adapter']['uuid'], 'sources': sources, 'profiles': data,
    'control_minus_candidate_counter_bytes': delta,
    'control_minus_candidate_retained_owned_bytes': data['control']['retained_owned_model_bytes'] - data['candidate']['retained_owned_model_bytes'],
    'scope': 'Sequential same-DLL vision flag control/candidate processes, six exact input/output/stop/profile '
             'controls against completed timing cohorts and same GPU UUID. Dedicated/shared/total medians '
             'are independent. All phase observed statistics are retained, not true lifetime peaks or physical '
             'residency. Owned staging, runtime accounting and process counters are distinct. These diagnostic '
             'holds/observations cannot establish primary latency, significance or independent quality.',
}
output.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({key: value for key, value in result.items() if key not in ('sources', 'profiles')}, indent=2))
