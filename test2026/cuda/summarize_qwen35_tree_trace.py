"""Attribute GPU events via CUDA API correlation and actual nested CPU NVTX."""
import collections
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from qwen35_nvtx_attribution import validate_ranges, correlated_path, phase

database, requests, output = map(Path, sys.argv[1:4])
assert not output.exists()
rows = json.loads(requests.read_text())
assert rows['all_exact_outputs_and_metrics'] and len(rows['records'])==3
assert [x['capture'] for x in rows['records']]==[False,False,True]
probe={'events':rows['nvtx_events']}
def phase(path):
    if path[-1]=='mtp_prefill_or_correction':return 'mtp_correction_execution' if 'mtp_correction' in path else 'mtp_prefill_execution'
    return path[-1]
c = sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True)
names = dict(c.execute('select id,value from StringIds'))
ranges = []
for start, end, event_type, text, text_id, tid in c.execute('select start,end,eventType,text,textId,globalTid from NVTX_EVENTS'):
    label = text if text is not None else names.get(text_id, '')
    if not label.startswith('garnet_tree:'):
        continue
    assert event_type == 59 and end is not None
    ranges.append({'start': start, 'end': end, 'tid': tid, 'label': label.removeprefix('garnet_tree:')})
roots = [r for r in ranges if r['label'] == 'request']
assert len(roots) == 1 and len({r['tid'] for r in ranges}) == 1
root = roots[0]
assert all(root['start'] <= r['start'] < r['end'] <= root['end'] for r in ranges)
ranges = validate_ranges(ranges)
counts=collections.Counter(x['label'] for x in ranges)
assert counts['request']==1
for label,value in rows['stages'].items():assert counts[label]==value['calls']
edges = []
for r in ranges:
    edges.extend([(r['start'], {'operation': 'push', 'label': r['label']}),
                  (r['end'], {'operation': 'pop', 'label': r['label']})])
assert len({t for t, _ in edges}) == len(edges), 'Ambiguous equal-timestamp marker edges'
assert [event for _, event in sorted(edges, key=lambda v: v[0])] == probe['events'], 'Dropped/reordered marker protocol'
apis = collections.defaultdict(list)
api_stages = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
for start, end, tid, correlation, name_id in c.execute('select start,end,globalTid,correlationId,nameId from CUPTI_ACTIVITY_KIND_RUNTIME'):
    row = {'start': start, 'end': end, 'tid': tid, 'name': names[name_id]}
    apis[correlation].append(row)
from qwen35_nvtx_attribution import api_path
for candidates in apis.values():
    for api in candidates:
        path = api_path(api, ranges)
        if path and path[0] == 'request':
            value = api_stages[phase(path)][api['name']]
            value[0] += 1;value[1] += api['end'] - api['start']


def union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:merged[-1][1] = max(end, merged[-1][1])
        else:merged.append([start, end])
    return sum(e-s for s, e in merged)


stages = collections.defaultdict(lambda: {'kernel': [], 'memcpy': [], 'memset': [], 'kernel_names': collections.defaultdict(lambda: [0, 0])})
totals = {}
unassigned = []
for table, kind in [('KERNEL', 'kernel'), ('MEMCPY', 'memcpy'), ('MEMSET', 'memset')]:
    selected = 'start,end,correlationId' + (',demangledName' if kind == 'kernel' else '')
    raw = list(c.execute(f'select {selected} from CUPTI_ACTIVITY_KIND_{table}'))
    totals[kind] = {'count': len(raw), 'sum_ns': sum(r[1]-r[0] for r in raw), 'union_ns': union([(r[0], r[1]) for r in raw])}
    for record in raw:
        start, end, correlation = record[:3]
        path, error = correlated_path({'start': start, 'end': end, 'correlation': correlation}, apis, ranges)
        group = phase(path) if path else 'UNASSIGNED'
        if error:
            unassigned.append({'kind': kind, 'start': start, 'end': end, 'correlation': correlation, 'reason': error})
        stages[group][kind].append((start, end))
        if kind == 'kernel':
            value = stages[group]['kernel_names'][names[record[3]]]
            value[0] += 1;value[1] += end-start
result_stages = {}
for group, data in stages.items():
    intervals = data['kernel'] + data['memcpy'] + data['memset']
    result_stages[group] = {kind: {'count': len(data[kind]), 'sum_ms': sum(e-s for s,e in data[kind])/1e6,
        'union_ms': union(data[kind])/1e6} for kind in ('kernel', 'memcpy', 'memset')}
    result_stages[group]['activity_union_ms'] = union(intervals)/1e6
    result_stages[group]['ranked_kernels'] = [{'name': n, 'count': v[0], 'sum_ms': v[1]/1e6}
        for n,v in sorted(data['kernel_names'].items(),key=lambda v:-v[1][1])]
    result_stages[group]['runtime_api'] = [{'name': n, 'count': v[0], 'sum_ms': v[1]/1e6}
        for n,v in sorted(api_stages[group].items(),key=lambda v:-v[1][1])]
for kind in totals:
    assert totals[kind]['count'] == sum(s[kind]['count'] for s in result_stages.values())
    assert abs(totals[kind]['sum_ns']/1e6 - sum(s[kind]['sum_ms'] for s in result_stages.values())) < 1e-7
unassigned_kernel_ns = sum(r['end']-r['start'] for r in unassigned if r['kind']=='kernel')
unassigned_kernel_count = sum(r['kind']=='kernel' for r in unassigned)
count_fraction = 1-unassigned_kernel_count/totals['kernel']['count']
time_fraction = 1-unassigned_kernel_ns/totals['kernel']['sum_ns']
result = {'schema_marker_protocol_passed': True, 'root_cpu_range': root, 'range_count': len(ranges),
    'marker_label_counts': dict(collections.Counter(r['label'] for r in ranges)), 'totals': totals,
    'stages': result_stages, 'unassigned_gpu_events': unassigned,
    'kernel_count_attributed_fraction': count_fraction, 'kernel_time_attributed_fraction': time_fraction,
    'predeclared_attribution_gate_passed': count_fraction >= .99 and time_fraction >= .99,
    'sources_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (database, requests)},
    'primary_latency_accepted': False, 'memory_accepted': False,
    'scope': 'One warmed instrumented request. Each GPU event is assigned only via one actual CUDA runtime '
             'correlation ID and its enclosing per-thread CPU NVTX chain. GPU timestamps need not be inside CPU '
             'ranges. Unassigned events retained; stage counts/sums reconcile to complete trace. Kernel sums, '
             'activity unions and overlapping API waits are distinct, not additive latency or removable overhead. '
             'No average hardware counters, causal speedup, primary latency/memory, quality or2x acceptance.'}
output.write_text(json.dumps(result, indent=2)+'\n')
print('QWEN35_NVTX_STAGE_ATTRIBUTION', count_fraction, time_fraction)
