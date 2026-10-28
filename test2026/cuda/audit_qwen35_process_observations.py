"""Audit observed external processes against saved model-role intervals.

Presence from creation until the first snapshot is a confirmed minimum lifetime.
No exit times are inferred. Snapshot gaps cannot exclude short-lived processes.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('root', type=Path)
parser.add_argument('observations', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--model', choices=('2b', '4b'), required=True)
parser.add_argument('--batch', type=int, required=True)
parser.add_argument('--study', default='activation-arena')
args = parser.parse_args()
assert not args.output.exists(), 'Preserve process audit evidence'
sources = {}


def read(path):
    blob = path.read_bytes()
    sources[path.name] = hashlib.sha256(blob).hexdigest()
    return blob.decode('utf-8-sig')


def epoch(value):
    return datetime.datetime.fromisoformat(value).timestamp()


observations = [json.loads(line) for line in read(args.observations).splitlines()]
started = next(row for row in observations if row['event'] == 'observer_started')
stopped = next(row for row in observations if row['event'] == 'stopped')
observed_window = (epoch(started['utc']), epoch(stopped['utc']))
metadata_errors = [row for row in observations if row['event'] in ('observer_failure', 'process_metadata_unavailable')]
processes = [row for row in observations if row['event'] == 'process_observed']
stages = json.loads(read(args.root / f'packed-study-{args.model}-b{args.batch}-{args.study}-stages.json'))
compilers = {'cl', 'nvcc', 'ptxas', 'ninja', 'cmake', 'msbuild', 'link'}
result = []
for role in ('a1', 'b1', 'b2', 'a2'):
    events = [row for row in stages if row['role'] == role]
    assert len(events) == 2 and events[0]['event'] == 'begin' and events[1]['event'] == 'complete'
    interval = (epoch(events[0]['utc']), epoch(events[1]['utc']))
    name = f'matched512-garnet-{args.model}-b{args.batch}-study-{args.study}-{role}.json'
    rows = json.loads(read(args.root / name))
    assert len(rows) == 28
    own = {row['process_id'] for row in rows}
    assert len(own) == 1
    overlap = []
    for process in processes:
        family = process['name'].lower().removesuffix('.exe')
        if family not in compilers | {'xlang3', 'llama-server', 'compute-sanitizer', 'nsys'}:
            continue
        if family == 'xlang3' and process['pid'] in own:
            continue
        minimum_presence = (epoch(process['started_utc']), epoch(process['observed_utc']))
        if max(interval[0], minimum_presence[0]) <= min(interval[1], minimum_presence[1]):
            classification = ('external_compiler' if family in compilers else
                              'external_diagnostic' if family in ('compute-sanitizer', 'nsys') else
                              'unowned_llama_server' if family == 'llama-server' else
                              'unowned_xlang_process')
            overlap.append({'classification': classification,
                            'process': process, 'confirmed_minimum_presence_unix': minimum_presence})
    covered = observed_window[0] <= interval[0] and interval[1] <= observed_window[1]
    result.append({'role': role, 'model_process_id': next(iter(own)), 'role_interval_unix': interval,
                   'entire_role_inside_observer_window': covered,
                   'known_external_presence_overlap': overlap})
output = {'model': args.model, 'batch': args.batch, 'sources_sha256': sources,
          'observer_window_unix': observed_window, 'roles': result,
          'all_four_roles_covered': all(row['entire_role_inside_observer_window'] for row in result),
          'known_overlap_count': sum(len(row['known_external_presence_overlap']) for row in result),
          'observer_metadata_errors': metadata_errors,
          'maximum_snapshot_gap_seconds': stopped['maximum_snapshot_gap_seconds'],
          'scope': 'Saved creation-to-first-observation intervals establish minimum process presence, not CPU/GPU '
                   'utilization or full lifetimes. Coverage is sampled; short processes can be missed. '
                   'Unowned XLang processes are not automatically classified as GPU models. '
                   'Earlier unobserved roles and other interference records remain separate. '
                   'No causal speed or clean-host guarantee.'}
args.output.write_text(json.dumps(output, indent=2), encoding='utf-8')
print(json.dumps({key: output[key] for key in ('all_four_roles_covered', 'known_overlap_count', 'maximum_snapshot_gap_seconds')}, indent=2))
