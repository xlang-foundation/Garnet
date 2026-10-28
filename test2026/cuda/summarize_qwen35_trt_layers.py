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
output = root / 'qwen35-trt-layer-v1-summary.json'
assert not output.exists()
sources = {}


def read(path):
    raw = path.read_bytes()
    sources[path.name] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw.decode('utf-8-sig'))


epoch_time = lambda s: datetime.datetime.fromisoformat(s).timestamp()
identity_path = root / 'qwen35-trt-layer-v1-before-run-identity.json'
identity = read(identity_path)
epoch = sources[identity_path.name]
observer = root / 'qwen35-trt-layer-v1-processes.jsonl'
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
    stem = f'profile-qwen35-trt-layers-{size}-b{batch}-v1'
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
        assert row['owned_memory']['owned_model_bytes'] == prior['owned_memory']['owned_model_bytes']
        assert row['warmup'] == (index < 8) and row['vision_compile_ms'] == 0
        probe = row['trt_layer_profile_diagnostic']
        assert probe['request_index'] == index and probe['enabled'] == (index == 8)
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
        'largest_individual_samples': sorted(samples, key=lambda s: s['milliseconds'], reverse=True)[:25],
        'observed_foreign_process_snapshots': foreign, 'observer_metadata_errors': phase_errors,
        'process_observation_qualified': not foreign and not phase_errors})
result = {'all56_input_output_work_and_layer_records_passed': True, 'cases': cases,
    'execution_identity_sha256': epoch, 'garnet_dll_sha256': identity['garnet_binary_sha256'],
    'sources_sha256': sources, 'maximum_snapshot_gap_seconds': obs[-1]['maximum_snapshot_gap_seconds'],
    'process_observation_qualified': all(c['process_observation_qualified'] for c in cases),
    'hardware_counters_collected': False, 'primary_latency_accepted': False, 'memory_accepted': False, 'production_promoted': False,
    'scope': 'Existing native IProfiler enabled only in warmed request8, first execution per engine temporarily '
             'bypasses graph replay. All28 original-MTP signatures per model preserved. Samples are not representative '
             'averages, execution-frequency weighted shares, exact original kernel/grid attribution, hardware bandwidth '
             'or occupancy, causal speedups, primary latency/memory, full-state, quality or2x acceptance.'}
output.write_text(json.dumps(result, indent=2) + '\n')
print('QWEN35_TRT_LAYER_AUDIT_PASS', len(cases))
