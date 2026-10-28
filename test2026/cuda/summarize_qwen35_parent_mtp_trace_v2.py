"""Independently audit current trace preservation, counts and interval measures."""
import collections
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
output = root / 'qwen35-parent-mtp-trace-v2-summary.json'
assert not output.exists()
sources = {}


def read(path):
    raw = path.read_bytes()
    sources[path.name] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw.decode('utf-8-sig'))


def union(intervals):
    merged = []
    for start, end in sorted(intervals):
        assert end >= start
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(e - s for s, e in merged)


def same_ms(left, right):
    assert abs(left - right) < 1e-7, (left, right)


identity_path = root / 'qwen35-parent-mtp-trace-v2-before-run-identity.json'
identity = read(identity_path)
epoch = sources[identity_path.name]
owners = read(root / 'qwen35-parent-mtp-v1-summary.json')
old = read(root / 'qwen35-shared-mtp-study-v1-summary.json')
assert owners['native_plain_controls_passed'] and old['all_completed_raw_controls_passed']
cases = []
for size, batch in [('2b', 1), ('4b', 8)]:
    for mode in ('original', 'parent'):
        stem = f'profile-parent-mtp-{size}-b{batch}-{mode}-v2'
        terminal = read(root / (stem + '-exit.json'))
        assert terminal['exit_code'] == 0
        assert terminal['execution_identity_sha256'].lower() == epoch
        assert terminal['garnet_dll_sha256'].lower() == identity['garnet_binary_sha256'].lower()
        assert terminal['gpu_uuid'] == 'GPU-472bb21a-0ca8-9d60-47a0-c527bb3d2d81'
        preservation = read(root / (stem + '-preservation.json'))
        assert preservation['same_all28_input_output_stop_profile_speculative_counters']
        rows = read(root / (stem + '-requests.json'))
        role = 'a1' if mode == 'original' else 'b1'
        reference = read(root / f'matched512-garnet-{size}-b{batch}-study-shared-mtp-v1-{role}.json')
        assert len(rows) == len(reference) == 28
        expected = (next(r for r in owners['profiles'] if r['model'] == size)['parent_backed_owned_model_bytes']
                    if mode == 'parent' else next(r for r in old['models'] if r['model'] == size)['per_run']['a1']['owned_model_bytes_range'][0])
        for index, (row, ref) in enumerate(zip(rows, reference)):
            for key in ('input_sha256', 'prompt_token_ids', 'generated_token_ids', 'sampled_tokens_including_stop',
                        'finish_reasons', 'stop_token_ids', 'profile', 'cache_policy', 'speculative_metrics', 'native_build_policy'):
                assert row[key] == ref[key], (size, mode, index, key)
            assert row['gpu_trace_diagnostic']['request_index'] == index
            assert row['gpu_trace_diagnostic']['storage_strategy'] == mode
            assert row['gpu_trace_diagnostic']['capture'] == (index == 8)
            assert row['owned_memory']['owned_model_bytes'] == expected
            if mode == 'parent':
                assert row['owned_memory']['speculative']['mtp_parent_storage']
        metrics = rows[8]['speculative_metrics']
        minimum_host_reads = metrics['mtp_head_calls'] + metrics['target_head_evaluations']
        database = root / (stem + '.sqlite')
        sources[database.name] = hashlib.sha256(database.read_bytes()).hexdigest()
        report = root / (stem + '.nsys-rep')
        assert report.stat().st_size > 0
        sources[report.name] = hashlib.sha256(report.read_bytes()).hexdigest()
        c = sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)
        names = dict(c.execute('select id,value from StringIds'))
        summary = read(root / (stem + '-summary.json'))
        activity = []
        for table in ('KERNEL', 'MEMCPY', 'MEMSET'):
            intervals = list(c.execute(f'select start,end from CUPTI_ACTIVITY_KIND_{table}'))
            activity += intervals
            assert len(intervals) == summary['groups'][table]['count']
            same_ms(sum(e - s for s, e in intervals) / 1e6, summary['groups'][table]['sum_ms'])
            same_ms(union(intervals) / 1e6, summary['groups'][table]['union_ms'])
        assert activity
        gpu_union = union(activity) / 1e6
        gpu_span = (max(e for _, e in activity) - min(s for s, _ in activity)) / 1e6
        same_ms(gpu_union, summary['gpu_activity_union_ms'])
        same_ms(gpu_span, summary['gpu_activity_span_ms'])
        runtime, api_intervals, setup_intervals = collections.defaultdict(lambda: [0, 0]), [], []
        for start, end, name in c.execute('select start,end,nameId from CUPTI_ACTIVITY_KIND_RUNTIME'):
            name = names[name]
            runtime[name][0] += 1
            runtime[name][1] += end - start
            (setup_intervals if name.startswith(('cudaProfilerStart', 'cudaProfilerStop')) else api_intervals).append((start, end))
        assert set(runtime) == {r['name'] for r in summary['runtime_api']}
        for row in summary['runtime_api']:
            assert row['count'] == runtime[row['name']][0]
            same_ms(row['sum_ms'], runtime[row['name']][1] / 1e6)
        sync_count = sum(v[0] for k, v in runtime.items() if k.startswith('cudaStreamSynchronize'))
        assert sync_count >= minimum_host_reads, 'Trace stream-sync count does not cover proven host prediction routes'
        copy_names = {row[0]: row[1] for row in c.execute('select * from ENUM_CUDA_MEMCPY_OPER')}
        copies = []
        for kind, count, size_bytes, duration in c.execute('select copyKind,count(*),sum(bytes),sum(end-start) from CUPTI_ACTIVITY_KIND_MEMCPY group by copyKind'):
            copies.append({'kind': copy_names[kind], 'count': count, 'bytes': size_bytes, 'summed_gpu_copy_ms': duration / 1e6})
        dtoh = sum(r['count'] for r in copies if r['kind'] == 'CUDA_MEMCPY_KIND_DTOH')
        assert dtoh >= 2 * minimum_host_reads, 'Trace D2H count does not cover both prediction payloads'
        c.close()
        cases.append({'process_id': rows[0]['process_id'], 'started_utc': terminal['started_utc'], 'ended_utc': terminal['ended_utc'], 'model': size, 'batch': batch, 'strategy': mode, 'retained_owned_model_bytes': expected,
            'captured_workload_key': [rows[8]['trial'], rows[8]['workload_group']],
            'proved_prediction_host_reads_lower_bound': minimum_host_reads,
            'observed_stream_sync_runtime_calls': sync_count, 'observed_gpu_copy_directions': copies,
            'gpu_activity_union_ms': gpu_union, 'gpu_activity_span_ms': gpu_span,
            'gpu_activity_uncovered_span_ms_not_removable_host_budget': gpu_span - gpu_union,
            'operational_runtime_api_union_ms_including_waited_gpu_work': union(api_intervals) / 1e6,
            'profiler_setup_stop_api_union_ms_separate': union(setup_intervals) / 1e6,
            'ranked_kernels': summary['ranked_kernels'], 'runtime_api': summary['runtime_api']})
observations_path = root / 'qwen35-parent-mtp-trace-v2-processes.jsonl'
raw = observations_path.read_bytes()
sources[observations_path.name] = hashlib.sha256(raw).hexdigest()
observations = [json.loads(line) for line in raw.decode('utf-8-sig').splitlines()]
import datetime
to_epoch = lambda s: datetime.datetime.fromisoformat(s).timestamp()
errors = [r for r in observations if r['event'] in ('observer_failure', 'process_metadata_unavailable')]
compiler_names = {'cl', 'nvcc', 'ptxas', 'ninja', 'cmake', 'msbuild', 'link'}
for case in cases:
    lo, hi = map(to_epoch, (case['started_utc'], case['ended_utc']))
    observed = [r for r in observations if r['event'] == 'process_observed']
    own = next(r for r in observed if r['pid'] == case['process_id'] and lo <= to_epoch(r['observed_utc']) <= hi)
    allowed = {case['process_id']}
    parent = own['parent_pid']
    while parent not in allowed:
        ancestor = next((r for r in observed if r['pid'] == parent), None)
        if ancestor is None or ancestor['name'].lower().removesuffix('.exe') != 'nsys':
            break
        allowed.add(parent)
        parent = ancestor['parent_pid']
    case['observed_external_model_or_diagnostic_snapshots'] = []
    for snapshot in observations:
        if snapshot['event'] != 'snapshot' or max(lo, to_epoch(snapshot['begin_utc'])) > min(hi, to_epoch(snapshot['end_utc'])):
            continue
        foreign = [p for p in snapshot['processes'] if p['name'].lower().removesuffix('.exe') in
                   {'xlang3', 'llama-server', 'compute-sanitizer', 'nsys'} and p['pid'] not in allowed]
        if foreign:
            case['observed_external_model_or_diagnostic_snapshots'].append({'begin_utc': snapshot['begin_utc'], 'end_utc': snapshot['end_utc'], 'processes': foreign})
    case['observed_external_compilers'] = [r for r in observations if r['event'] == 'process_observed'
        and r['name'].lower().removesuffix('.exe') in compiler_names
        and max(lo, to_epoch(r['started_utc'])) <= min(hi, to_epoch(r['observed_utc']))]
    case['observer_metadata_errors_in_phase'] = [r for r in errors if 'observed_utc' not in r or lo <= to_epoch(r['observed_utc']) <= hi]
    case['process_observation_qualified'] = not case['observed_external_compilers'] and not case['observer_metadata_errors_in_phase'] and not case['observed_external_model_or_diagnostic_snapshots']
memory_prerequisite = read(root / 'qwen35-parent-mtp-memory-v1-rejected-analysis.json')
assert not memory_prerequisite['isolation_accepted']
result = {'all_four_trace_controls_passed': True, 'cases': cases, 'sources_sha256': sources,
    'execution_identity_sha256': epoch, 'garnet_dll_sha256': identity['garnet_binary_sha256'],
    'original_memory_isolation_accepted': False, 'trace_process_observation_qualified': all(c['process_observation_qualified'] for c in cases), 'observer_errors': errors, 'primary_latency_accepted': False, 'production_promoted': False, 'two_x_achieved': False,
    'scope': 'Full28 real-image signature/control/work preservation and one warmed instrumented request per case. '
             'GPU activity spans, unions and API waits overlap; none is automatically recoverable overhead. '
             'One capture per strategy is diagnostic and cannot establish general bottlenecks or causal speedup. '
             'No occupancy, bandwidth, physical residency, independent quality, allbatch/capacity or complete host isolation proof.'}
output.write_text(json.dumps(result, indent=2) + '\n')
print('PARENT_MTP_TRACE_AUDIT_PASS', len(cases))
for case in cases:
    print(json.dumps({k: case[k] for k in ('model', 'strategy', 'gpu_activity_union_ms', 'gpu_activity_span_ms', 'observed_stream_sync_runtime_calls')}))
