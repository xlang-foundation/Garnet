"""Audit all compact ABBA roles; report API-return timing and retained owners."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('root', type=Path)
p.add_argument('identity', type=Path)
p.add_argument('size', choices=('2b', '4b'))
p.add_argument('batch', type=int, choices=(1, 8))
p.add_argument('--study', default='compact-kv')
a = p.parse_args()
output = a.root / f'packed-study-{a.size}-b{a.batch}-{a.study}-summary.json'
assert not output.exists(), 'Preserve compact study summary'
sources = {}


def read(path):
    raw = path.read_bytes()
    sources[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw.decode('utf-8-sig'))


epoch = read(a.identity)
epoch_sha = sources[str(a.identity)]
data = {}
for role in ('a1', 'b1', 'b2', 'a2'):
    tag = f'matched512-garnet-{a.size}-b{a.batch}-study-{a.study}-{role}'
    terminal = read(a.root / (tag + '-exit.json'))
    assert terminal['exit_code'] == 0 and terminal['role'] == role
    assert terminal['execution_identity_sha256'].lower() == epoch_sha
    assert terminal['garnet_dll_sha256'].lower() == epoch['garnet_binary_sha256'].lower()
    rows = read(a.root / (tag + '.json'))
    indexed = {(r['trial'], r['workload_group']): r for r in rows}
    assert len(rows) == len(indexed) == 28
    assert all(r['batch'] == a.batch for r in rows)
    data[role] = indexed
for left, right in (('a1', 'b1'), ('a1', 'b2'), ('a2', 'b2')):
    audit = read(a.root / f'{a.study}-{a.size}-b{a.batch}-{left}-{right}-workload-control.json')
    assert audit['all_passed'] and audit['all_28_checked'] and audit['prior_control_cohort_checked']
    for source in audit['sources']:
        assert hashlib.sha256(Path(source['file']).read_bytes()).hexdigest() == source['sha256']
keys = [key for key, row in data['a1'].items() if not row['warmup']]
assert len(keys) == 20
assert all(rows.keys() == data['a1'].keys() for rows in data.values())
for key in data['a1']:
    assert all(rows[key]['warmup'] == data['a1'][key]['warmup'] for rows in data.values())
group_ms = {letter: {key: statistics.median(data[letter + repeat][key]['wall_seconds']
    for repeat in ('1', '2')) * 1000 for key in keys} for letter in ('a', 'b')}
aggregate = {letter: statistics.median(groups.values()) for letter, groups in group_ms.items()}
per_run = {role: {
    'median_api_return_ms': statistics.median(rows[k]['wall_seconds'] for k in keys) * 1000,
    'median_full_request_output_tokens_per_second': statistics.median(
        sum(rows[k]['generated_tokens']) / rows[k]['wall_seconds'] for k in keys),
    'owned_model_bytes': sorted({rows[k]['owned_memory']['owned_model_bytes'] for k in keys}),
} for role, rows in data.items()}
result = {
    'sources_sha256': sources, 'garnet_dll_sha256': epoch['garnet_binary_sha256'],
    'execution_identity_sha256': epoch_sha, 'model': a.size, 'batch': a.batch, 'study': a.study,
    'order': ['a1', 'b1', 'b2', 'a2'], 'all28_workload_owner_controls_passed': True,
    'all20_warm_groups_included': True, 'per_run': per_run,
    'aggregate_group_median_api_return_ms': aggregate,
    'control_over_candidate_ratio': aggregate['a'] / aggregate['b'],
    'per_group_repeated_median_ms': {letter: [{'trial': k[0], 'workload_group': k[1], 'ms': ms}
        for k, ms in groups.items()] for letter, groups in group_ms.items()},
    'primary_performance_eligible': False, 'independent_quality_qualified': False,
    'process_memory_measured': False, 'goal_2x_achieved': False,
    'scope': 'Only compactKVOutputs changes. Same native DLL and frozen sources, fresh model '
        'processes in ABBA order, all28 real-image requests controlled. All20 warm groups '
        'retained; median two repeats per group, then median across groups. API-return '
        'latency can exclude queued GPU work, separately diagnosed. Two repeats do not '
        'establish significance. Owned allocations are not process VRAM or residency. '
        'No independent quality acceptance or new external-engine comparison is claimed.',
}
output.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({'aggregate_group_median_api_return_ms': aggregate,
                 'control_over_candidate_ratio': result['control_over_candidate_ratio']}))
