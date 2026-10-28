"""Current-DLL node costs and API/idle overlap, with no speed forecast."""
import argparse
import collections
import hashlib
import json
import sqlite3
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('root', type=Path)
a = p.parse_args()
output = a.root / 'current-store-node-trace-costs.json'
assert not output.exists(), 'Preserve trace cost evidence'
sources = []
cases = []


def read(path):
    blob = path.read_bytes()
    sources.append({'file': path.name, 'sha256': hashlib.sha256(blob).hexdigest()})
    return json.loads(blob)


def merge(intervals):
    result = []
    for start, end in sorted(intervals):
        assert start > 0 and end >= start
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


for size, batch in (('2b', 1), ('4b', 8)):
    prefix = f'profile-current-store-{size}-b{batch}-node-v1'
    terminal = read(a.root / (prefix + '-exit.json'))
    assert terminal['exit_code'] == 0
    preservation = read(a.root / (prefix + '-preservation.json'))
    assert preservation['same_all28_input_output_stop_profile_speculative_counters']
    summary = read(a.root / (prefix + '-summary.json'))
    database = a.root / (prefix + '.sqlite')
    with database.open('rb') as stream:
        sources.append({'file': database.name, 'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()})
    c = sqlite3.connect(f'file:{database.as_posix()}?mode=ro', uri=True)
    tables = {row[0] for row in c.execute("select name from sqlite_master where type='table'")}
    names = dict(c.execute('select id,value from StringIds'))
    intervals = []
    for kind in ('KERNEL', 'MEMCPY', 'MEMSET'):
        table = 'CUPTI_ACTIVITY_KIND_' + kind
        if table in tables:
            intervals.extend(c.execute(f'select start,end from {table}'))
    activity = merge(intervals)
    assert activity
    first, last = activity[0][0], activity[-1][1]
    idle = [(activity[i][1], activity[i + 1][0]) for i in range(len(activity) - 1)]
    api = collections.defaultdict(lambda: [0, 0, 0])
    for start, end, name in c.execute('select start,end,nameId from CUPTI_ACTIVITY_KIND_RUNTIME'):
        start, end = max(first, start), min(last, end)
        if start >= end:
            continue
        values = api[names[name]]
        values[0] += 1
        values[1] += end - start
        values[2] += sum(max(0, min(end, y) - max(start, x)) for x, y in idle if x < end and y > start)
    c.close()
    groups = collections.defaultdict(lambda: {'count': 0, 'sum_ms': 0.0})
    for kernel in summary['ranked_kernels']:
        name = kernel['name'].lower()
        family = ('bf16_gemm_or_gemv' if ('gemm' in name or 'gemv' in name) and 'bf16' in name else
                  'indexed_proposal_head' if 'indexeddot' in name or 'indexedbest' in name else
                  'recurrent_register_state' if 'registerstate128kernel' in name else 'other')
        groups[family]['count'] += kernel['count']
        groups[family]['sum_ms'] += kernel['sum_ms']
    total_kernel = summary['groups']['KERNEL']['sum_ms']
    for group in groups.values():
        group['fraction_of_summed_kernel_time'] = group['sum_ms'] / total_kernel
    cases.append({
        'model': size, 'batch': batch, 'garnet_dll_sha256': terminal['garnet_dll_sha256'],
        'source_role': terminal['source_role'], 'captured_index': preservation['captured_index'],
        'gpu_activity_span_ms': (last - first) / 1e6,
        'gpu_activity_union_ms': sum(end - start for start, end in activity) / 1e6,
        'between_activity_idle_ms': sum(end - start for start, end in idle) / 1e6,
        'activity_groups': summary['groups'], 'kernel_families': dict(groups),
        'top_runtime_api_by_idle_overlap': [
            {'name': name, 'count': values[0], 'clipped_sum_ms': values[1] / 1e6,
             'idle_overlap_ms': values[2] / 1e6}
            for name, values in sorted(api.items(), key=lambda item: -item[1][2])[:20]
        ],
    })
result = {'sources': sources, 'cases': cases, 'primary_performance_eligible': False,
          'scope': 'One warmed index8 node-level diagnostic per current profile, all28 real request '
                   'input/output/stop/profile/discrete-MTP preservation. Kernel families sum named kernels; '
                   'GPU spans/unions and API intervals are different measures. API/idle overlap is not '
                   'causality or recoverable time; profiler overhead prevents direct speed forecasts. '
                   'No independent quality, significance, default promotion or2x claim.'}
output.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({'cases': cases}, indent=2))
