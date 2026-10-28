"""Audit sampled named-process presence across the four memory diagnostics."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('root', type=Path)
p.add_argument('observations', type=Path)
p.add_argument('output', type=Path)
p.add_argument('--study', choices=('internal-stores', 'vision-graph', 'compact-kv'), default='internal-stores')
p.add_argument('--attempt', default='v1')
a = p.parse_args()
assert not a.output.exists(), 'Preserve memory observer evidence'
sources = {}


def read(path, lines=False):
    blob = path.read_bytes()
    sources[path.name] = hashlib.sha256(blob).hexdigest()
    text = blob.decode('utf-8-sig')
    return [json.loads(line) for line in text.splitlines()] if lines else json.loads(text)


def epoch(value):
    return datetime.datetime.fromisoformat(value).timestamp()


observations = read(a.observations, lines=True)
started = next(row for row in observations if row['event'] == 'observer_started')
stopped = next(row for row in observations if row['event'] == 'stopped')
window = (epoch(started['utc']), epoch(stopped['utc']))
errors = [row for row in observations if row['event'] in ('observer_failure', 'process_metadata_unavailable')]
processes = [row for row in observations if row['event'] == 'process_observed']
compilers = {'cl', 'nvcc', 'ptxas', 'ninja', 'cmake', 'msbuild', 'link'}
results = []
for size, batch in (('2b', 1), ('4b', 8)):
    for mode in ('control', 'candidate'):
        stem = f'memory-{a.study}-{size}-b{batch}-{mode}' + ('-' + a.attempt if a.study == 'compact-kv' else '')
        owner = read(a.root / (stem + '-owner.json'))
        terminal = read(a.root / (stem + '-exit.json'))
        events = read(a.root / (stem + '-session.json'))
        assert terminal['model_exit_code'] == terminal['monitor_exit_code'] == 0
        own = owner['process_id']
        assert {row['process_id'] for row in events} == {own}
        assert events[-1]['event'] == 'complete'
        interval = (epoch(owner['started']), epoch(terminal['ended_utc']))
        own_seen = [row for row in processes if row['pid'] == own and
                    interval[0] <= epoch(row['observed_utc']) <= interval[1]]
        assert own_seen, 'Memory model was not sampled by observer'
        overlaps = []
        for process in processes:
            family = process['name'].lower().removesuffix('.exe')
            if family not in compilers | {'xlang3', 'llama-server', 'compute-sanitizer', 'nsys'}:
                continue
            if family == 'xlang3' and process['pid'] == own:
                continue
            presence = (epoch(process['started_utc']), epoch(process['observed_utc']))
            if max(interval[0], presence[0]) <= min(interval[1], presence[1]):
                overlaps.append({
                    'classification': ('external_compiler' if family in compilers else
                                       'external_diagnostic' if family in ('compute-sanitizer', 'nsys') else
                                       'unowned_llama_server' if family == 'llama-server' else
                                       'other_xlang_process'),
                    'process': process, 'confirmed_minimum_presence_unix': presence,
                })
        results.append({'model': size, 'batch': batch, 'mode': mode, 'process_id': own,
                        'model_and_monitor_interval_unix': interval,
                        'entire_interval_inside_observer_window': window[0] <= interval[0] <= interval[1] <= window[1],
                        'known_external_presence_overlap': overlaps})
result = {
    'study': a.study,
    'sources_sha256': sources, 'observer_window_unix': window, 'cases': results,
    'all_four_cases_covered': all(row['entire_interval_inside_observer_window'] for row in results),
    'known_overlap_count': sum(len(row['known_external_presence_overlap']) for row in results),
    'observer_metadata_errors': errors,
    'maximum_snapshot_gap_seconds': stopped['maximum_snapshot_gap_seconds'],
    'scope': 'Windows named-process snapshots establish creation-to-first-observation minimum presence only. '
             'Model and external monitor intervals are covered; no full process lifetimes, resource utilization, '
             'short-process exclusion or complete host/GPU isolation is established. Guest model entry/exit '
             'guards are separate from this Windows observer. Counter samples are diagnostic committed process '
             'memory, not physical residency, true lifetime peaks, primary latency or independent quality.',
}
a.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({key: result[key] for key in ('all_four_cases_covered', 'known_overlap_count', 'maximum_snapshot_gap_seconds')}, indent=2))
