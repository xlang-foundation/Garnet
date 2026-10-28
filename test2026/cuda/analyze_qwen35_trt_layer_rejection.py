"""Audit layer identifiers/timings and full-cohort preservation, not speed."""
import csv
import datetime
import hashlib
import io
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
output = root / 'qwen35-trt-layer-v2-rejected-analysis.json'
assert not output.exists()
sources = {}
owner_rejections = []


def read(path):
    raw = path.read_bytes()
    sources[path.name] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw.decode('utf-8-sig'))


epoch_time = lambda s: datetime.datetime.fromisoformat(s).timestamp()
identity_path = root / 'qwen35-trt-layer-v2-before-run-identity.json'
identity = read(identity_path)
epoch = sources[identity_path.name]
observer = root / 'qwen35-trt-layer-v2-processes.jsonl'
blob = observer.read_bytes()
sources[observer.name] = hashlib.sha256(blob).hexdigest()
obs = [json.loads(line) for line in blob.decode('utf-8-sig').splitlines()]
assert obs[0]['event'] == 'observer_started' and obs[-1]['event'] == 'stopped'
assert obs[-1]['ticks'] > 0
errors = [r for r in obs if r['event'] in ('observer_failure', 'process_metadata_unavailable')]
processes = [r for r in obs if r['event'] == 'process_observed']
cases = []
previous_end = None
for size, batch in [('2b', 1), ('4b', 8)]:
    stem = f'profile-qwen35-trt-layers-{size}-b{batch}-v2'
    terminal = read(root / (stem + '-exit.json'))
    assert terminal['exit_code'] == 0
    assert terminal['execution_identity_sha256'].lower() == epoch
    assert terminal['garnet_dll_sha256'].lower() == identity['garnet_binary_sha256'].lower()
    assert terminal['gpu_uuid'] == 'GPU-472bb21a-0ca8-9d60-47a0-c527bb3d2d81'
    rows = read(root / (stem + '-requests.json'))
    ref = read(root / f'matched512-garnet-{size}-b{batch}-study-shared-mtp-v1-a1.json')
    assert len(rows) == len(ref) == 28
    layer_path = root / (stem + '-requests.layers.tsv')
    for index, (row, prior) in enumerate(zip(rows, ref)):
        for field in ('profile', 'native_build_policy', 'input_sha256', 'prompt_token_ids',
                      'generated_token_ids', 'sampled_tokens_including_stop', 'finish_reasons',
                      'stop_token_ids', 'cache_policy', 'speculative_metrics', 'visual_tokens_per_row'):
            assert row[field] == prior[field], (size, index, field)
        actual_owned = row['owned_memory']['owned_model_bytes']
        reference_owned = prior['owned_memory']['owned_model_bytes']
        if actual_owned != reference_owned:
            actual_spec = row['owned_memory']['speculative']
            prior_spec = prior['owned_memory']['speculative']
            delta = actual_owned - reference_owned
            assert delta == actual_spec['owned_speculative_bytes'] - prior_spec['owned_speculative_bytes']
            assert delta == actual_spec['mtp_graph']['trt_allocator_live_bytes'] - prior_spec['mtp_graph']['trt_allocator_live_bytes']
            owner_rejections.append({'model': size, 'batch': batch, 'request_index': index,
                'profiled_owned_model_bytes': actual_owned, 'reference_owned_model_bytes': reference_owned,
                'owned_delta_bytes': delta, 'mtp_contexts': actual_spec['mtp_graph']['contexts'],
                'reference_mtp_contexts': prior_spec['mtp_graph']['contexts'],
                'mtp_allocator_live_delta_bytes': delta})
        assert row['warmup'] == (index < 8) and row['vision_compile_ms'] == 0
        probe = row['startup_trt_layer_profile_diagnostic']
        assert probe['request_index'] == index and probe['native_profile_environment_fixed']
        assert 0 <= probe['profile_bytes_before'] <= probe['profile_bytes_after']
        assert probe['profile_bytes_after'] > 0
        if index:assert probe['profile_bytes_before'] == rows[index-1]['startup_trt_layer_profile_diagnostic']['profile_bytes_after']
        assert Path(probe['layer_file']).resolve() == layer_path
        assert not probe['primary_latency_accepted'] and not probe['memory_accepted']
    blob = layer_path.read_bytes()
    sources[layer_path.name] = hashlib.sha256(blob).hexdigest()
    engines = {}
    samples = []
    headers = 0
    for record in csv.reader(io.StringIO(blob.decode('utf-8-sig')), delimiter='\t'):
        if record == ['engine', 'layer', 'milliseconds']:
            headers += 1
            continue
        assert len(record) == 3, ('Malformed layer record', record)
        engine, layer, value = record
        assert engine and layer
        milliseconds = float(value)
        assert math.isfinite(milliseconds) and milliseconds >= 0
        sample = {'engine': engine, 'layer': layer, 'milliseconds': milliseconds}
        engines.setdefault(engine, []).append(sample)
        samples.append(sample)
    assert samples and len(engines) == headers, ('Missing/duplicate engine profile batches', headers, len(engines))
    lo, hi = map(epoch_time, (terminal['started_utc'], terminal['ended_utc']))
    assert previous_end is None or previous_end <= lo
    previous_end = hi
    assert epoch_time(obs[0]['utc']) <= lo < hi <= epoch_time(obs[-1]['utc'])
    own = next(r for r in processes if r['pid'] == rows[0]['process_id'] and lo <= epoch_time(r['observed_utc']) <= hi)
    foreign = []
    for snap in obs:
        if snap['event'] != 'snapshot' or max(lo, epoch_time(snap['begin_utc'])) > min(hi, epoch_time(snap['end_utc'])):
            continue
        other = [p for p in snap['processes'] if p['pid'] != own['pid']]
        if other:
            foreign.append({'begin_utc': snap['begin_utc'], 'end_utc': snap['end_utc'], 'processes': other})
    phase_errors = [r for r in errors if 'observed_utc' not in r or lo <= epoch_time(r['observed_utc']) <= hi]
    cases.append({'model': size, 'batch': batch, 'process_id': own['pid'], 'sample_count': len(samples),
        'engine_count': len(engines), 'engine_samples': engines,
        'request_indices_with_profile_file_growth': [i for i,r in enumerate(rows) if r['startup_trt_layer_profile_diagnostic']['profile_bytes_after'] > r['startup_trt_layer_profile_diagnostic']['profile_bytes_before']],
        'profile_bytes_before_first_request': rows[0]['startup_trt_layer_profile_diagnostic']['profile_bytes_before'],
        'largest_individual_samples': sorted(samples, key=lambda s: s['milliseconds'], reverse=True)[:25],
        'observed_foreign_process_snapshots': foreign, 'observer_metadata_errors': phase_errors,
        'process_observation_qualified': not foreign and not phase_errors})
result = {'all56_input_output_work_and_layer_metadata_checks_passed': True, 'cases': cases,
    'execution_identity_sha256': epoch, 'garnet_dll_sha256': identity['garnet_binary_sha256'],
    'sources_sha256': sources, 'maximum_snapshot_gap_seconds': obs[-1]['maximum_snapshot_gap_seconds'],
    'process_observation_qualified': all(c['process_observation_qualified'] for c in cases),
    'hardware_counters_collected': False, 'primary_latency_accepted': False, 'memory_accepted': False, 'production_promoted': False,
    'scope': 'Profiler flag fixed before process startup acrossall28 requests. Existing IProfiler samples first execution per engine, potentially cold, and temporarily '
             'bypasses graph replay; serial fast path is disabled by the fixed flag. All28 original-MTP signatures per model preserved. Samples are not representative '
             'averages, execution-frequency weighted shares, exact original kernel/grid attribution, hardware bandwidth '
             'or occupancy, causal speedups, primary latency/memory, full-state, quality or2x acceptance.'}
terminal = read(root / 'qwen35-trt-layer-v2-controller-exit.json')
assert terminal['exit_code'] == 1 and owner_rejections
result.update({'all_original_diagnostic_gates_passed': False, 'diagnostic_accepted': False,
    'owned_byte_preservation_passed': False, 'original_controller_exit_code': terminal['exit_code'],
    'ownership_rejections': owner_rejections,
    'rejection_scope': 'Original owned-byte assertion remains FAILED and original controller exit1 retained. Only that assertion is converted to explicit failure records; all other original assertions remain. Every owned delta is checked against the speculative MTP graph live allocator category. No primary latency/memory or corrected successful diagnostic gate.'})
result['sources_sha256'] = dict(sources)
output.write_text(json.dumps(result, indent=2) + '\n')
print('QWEN35_TRT_LAYER_OWNERSHIP_REJECTION_RETAINED', len(owner_rejections))
