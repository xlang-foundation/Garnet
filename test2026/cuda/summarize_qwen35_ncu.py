"""Audit raw kernel counters and preserved full-cohort behavior, not latency."""
import csv
import datetime
import hashlib
import io
import json
import math
import re
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
output = root / 'qwen35-ncu-v1-summary.json'
assert not output.exists()
sources = {}


def read(path):
    raw = path.read_bytes()
    sources[path.name] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw.decode('utf-8-sig'))


def number(value):
    try:
        result = float(value.replace(',', '').strip())
        return result if math.isfinite(result) else None
    except (ValueError, AttributeError):
        return None


def vector(value):
    parts = re.findall(r'\d+', value)
    assert len(parts) == 3, ('Missing3D launch metadata', value)
    return list(map(int, parts))


identity_path = root / 'qwen35-ncu-v1-before-run-identity.json'
identity = read(identity_path)
epoch = sources[identity_path.name]
plan = read(root / 'qwen35-ncu-kernel-plan-v2.json')
observation_path = root / 'qwen35-ncu-v1-processes.jsonl'
raw = observation_path.read_bytes()
sources[observation_path.name] = hashlib.sha256(raw).hexdigest()
observations = [json.loads(line) for line in raw.decode('utf-8-sig').splitlines()]
assert observations[0]['event'] == 'observer_started' and observations[-1]['event'] == 'stopped'
epoch_time = lambda s: datetime.datetime.fromisoformat(s).timestamp()
window = (epoch_time(observations[0]['utc']), epoch_time(observations[-1]['utc']))
errors = [r for r in observations if r['event'] in ('observer_failure', 'process_metadata_unavailable')]
processes = [r for r in observations if r['event'] == 'process_observed']
cases = []
for selection in plan['cases']:
    size, batch = selection['model'], selection['batch']
    stem = f'profile-qwen35-ncu-{size}-b{batch}-v1'
    terminal = read(root / (stem + '-exit.json'))
    assert terminal['exit_code'] == 0
    assert terminal['execution_identity_sha256'].lower() == epoch
    assert terminal['garnet_dll_sha256'].lower() == identity['garnet_binary_sha256'].lower()
    assert terminal['gpu_uuid'] == 'GPU-472bb21a-0ca8-9d60-47a0-c527bb3d2d81'
    rows = read(root / (stem + '-requests.json'))
    ref = read(root / f'matched512-garnet-{size}-b{batch}-study-shared-mtp-v1-a1.json')
    assert len(rows) == len(ref) == 28
    for index, (row, previous) in enumerate(zip(rows, ref)):
        for field in ('profile', 'native_build_policy', 'input_sha256', 'prompt_token_ids', 'generated_token_ids',
                      'sampled_tokens_including_stop', 'finish_reasons', 'stop_token_ids', 'cache_policy', 'speculative_metrics'):
            assert row[field] == previous[field], (size, index, field)
        assert row['gpu_trace_diagnostic']['capture'] == (index == 8)
    assert read(root / (stem + '-preservation.json'))['same_all28_input_output_stop_profile_speculative_counters']
    report = root / (stem + '.ncu-rep')
    assert report.stat().st_size > 0
    sources[report.name] = hashlib.sha256(report.read_bytes()).hexdigest()
    raw_path = root / (stem + '-raw.csv')
    blob = raw_path.read_bytes()
    sources[raw_path.name] = hashlib.sha256(blob).hexdigest()
    table = list(csv.reader(io.StringIO(blob.decode('utf-8-sig'))))
    canonical = lambda value: re.sub('[^a-z0-9]', '', value.lower())
    header_index = next(i for i, record in enumerate(table) if {'id', 'metricname', 'metricvalue'} <= {canonical(v) for v in record})
    headers = [canonical(v) for v in table[header_index]]
    actions = {}
    for record in table[header_index + 1:]:
        if len(record) != len(headers):
            continue
        row = dict(zip(headers, record))
        if not row['id'].isdigit():
            continue
        action = actions.setdefault(row['id'], {'kernel_name': row['kernelname'], 'grid': vector(row['gridsize']),
            'block': vector(row['blocksize']), 'metrics': {}})
        assert action['kernel_name'] == row['kernelname']
        action['metrics'][row['metricname']] = {'value': row['metricvalue'], 'unit': row.get('metricunit', ''), 'numeric': number(row['metricvalue'])}
    assert actions, 'No collected kernel counters'
    configurations = []
    representative = []
    for action_id, action in actions.items():
        assert 'cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8' in action['kernel_name']
        metrics = action['metrics']
        for required in ('launch__shared_mem_per_block_static', 'launch__shared_mem_per_block_dynamic',
                         'sm__throughput.avg.pct_of_peak_sustained_elapsed', 'dram__throughput.avg.pct_of_peak_sustained_elapsed',
                         'gpu__time_duration.sum'):
            assert required in metrics and metrics[required]['numeric'] is not None, (size, 'Missing counter', required, sorted(metrics))
        config = action['grid'] + action['block'] + [int(metrics['launch__shared_mem_per_block_static']['numeric']), int(metrics['launch__shared_mem_per_block_dynamic']['numeric'])]
        action['configuration'] = config
        configurations.append(tuple(config))
        if config == selection['selected_configuration']:
            representative.append(action_id)
    assert len(configurations) == len(set(configurations)), 'Expected one sample per launch configuration'
    assert representative, 'Collected metadata misses declared representative configuration; no attribution'
    lo, hi = map(epoch_time, (terminal['started_utc'], terminal['ended_utc']))
    assert window[0] <= lo < hi <= window[1]
    own = next(r for r in processes if r['pid'] == rows[0]['process_id'] and lo <= epoch_time(r['observed_utc']) <= hi)
    allowed = {own['pid']}
    parent = own['parent_pid']
    while parent not in allowed:
        ancestor = next((r for r in processes if r['pid'] == parent), None)
        if ancestor is None or ancestor['name'].lower().removesuffix('.exe') != 'ncu':
            break
        allowed.add(parent)
        parent = ancestor['parent_pid']
    foreign = []
    for snapshot in observations:
        if snapshot['event'] != 'snapshot' or max(lo, epoch_time(snapshot['begin_utc'])) > min(hi, epoch_time(snapshot['end_utc'])):
            continue
        other = [p for p in snapshot['processes'] if p['pid'] not in allowed]
        if other:
            foreign.append({'begin_utc': snapshot['begin_utc'], 'end_utc': snapshot['end_utc'], 'processes': other})
    phase_errors = [r for r in errors if 'observed_utc' not in r or lo <= epoch_time(r['observed_utc']) <= hi]
    cases.append({'model': size, 'batch': batch, 'process_id': own['pid'], 'actions': actions,
        'declared_representative_action_ids': representative, 'observed_foreign_process_snapshots': foreign,
        'observer_metadata_errors': phase_errors, 'process_observation_qualified': not foreign and not phase_errors})
result = {'all_counter_metadata_and_full28_controls_passed': True, 'cases': cases,
    'sources_sha256': sources, 'execution_identity_sha256': epoch, 'garnet_dll_sha256': identity['garnet_binary_sha256'],
    'process_observation_qualified': all(c['process_observation_qualified'] for c in cases),
    'primary_latency_accepted': False, 'process_memory_accepted': False, 'production_promoted': False,
    'scope': 'One kernel-replay counter sample per launch configuration of one dominant GEMM class in warmed request8; full28 signatures preserved. '
             'Declared configuration membership, not exact operator/operand identity, is established. Cache/clock controls explicitly none. '
             'Replay, instrumentation and sampled process observation prevent primary latency, memory, causal speedup or complete-isolation claims.'}
output.write_text(json.dumps(result, indent=2) + '\n')
print('QWEN35_NCU_COUNTER_AUDIT_PASS', len(cases))
