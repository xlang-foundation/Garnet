#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit saved direct Mistral prefix-API observations and cold/warm token parity.

Driver traces and successful native assertions are evidence here; they are not
independent GPU KV snapshots. No cached HTTP, arbitrary prefix truncation,
full4K request, coding-quality, peak-memory or speed qualification.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from audit_mistral_http_admission import OWNERS, require


def validate(fixture,controls,data,status):
    require(status['returncode'] == 0 and status['status'] == 'collection_complete_audit_pending'
            and data['completed'] is True, 'incomplete native collection')
    require('native_hybrid_token_prefix' in data['backend'] and 'affine_coefficient_dtype=fp16' in data['backend'],
            'incorrect native engine')
    profile = data['profile']
    require((profile['batch'],profile['capacity'],profile['maximum_chunk'],profile['host_cache_bytes'],profile['entries'])
            == (4,4096,128,96*1024*1024,3), 'changed prefix profile')
    coding = [c for c in fixture['cases'] if c['kind'] == 'coding']
    original = coding[0]['prompt_ids']
    gold = controls[coding[0]['id']]['generated_ids']
    appended = original+gold[:32]
    changed = list(original)
    changed[70] = (changed[70]+17)%32768
    labels = ['A-initial','E-replacement','A-exact-return','append-fresh',
              'append-restored','changed-fresh','changed-restored']
    budgets = [32,16,32,32,32,16,16]
    prompts = [original,coding[4]['prompt_ids'],original,appended,appended,changed,changed]
    records = data['records']
    require([r['label'] for r in records] == labels, 'missing/reordered cases')
    expected = [gold[:32],controls[coding[4]['id']]['generated_ids'][:16],gold[:32],None,
                records[3]['generated_ids'],None,records[5]['generated_ids']]
    hits = [0,0,1,1,2,2,3]
    captures = data['captures']
    require([c['boundary'] for c in captures] == [64,84,115], 'changed stored checkpoints')
    for i,capture in enumerate(captures):
        stats = capture['cache']
        require(capture['row'] == 0 and capture['epoch'] == data['leases'][0]['epoch']
                and stats['entries'] == i+1 and stats['hits'] == stats['misses'] == 0
                and sum(c['boundary'] for c in captures[:i+1])*262144 <= stats['accounted_host_bytes'] <= profile['host_cache_bytes'],
                'incorrect snapshot accounting/boundary')
    count = 0
    for i,(record,prompt,budget,wanted) in enumerate(zip(records,prompts,budgets,expected)):
        ids,stats = record['generated_ids'],record['cache']
        require(record['prompt_ids'] == prompt and len(ids) == budget and all(type(t) is int and 0 <= t < 32768 and t != 2 for t in ids),
                'changed input/output tokens')
        require(record['expected_ids'] == wanted and record['exact_control'] is (None if wanted is None else ids == wanted)
                and (wanted is None or ids == wanted), 'cold/warm/frozen output divergence')
        require(record['consumed_tokens'] == len(prompt)+budget-1
                and record['survivor_counts_before'] == [count]*3
                and record['survivor_counts_after'] == [count+budget]*3
                and math.isfinite(record['seconds']) and record['seconds'] > 0, 'incorrect progress')
        count += budget
        require(stats['hits'] == hits[i] and stats['misses'] == int(i>0)
                and stats['entries'] == (2 if i == 0 else 3)
                and stats['evictions'] == stats['replacements'] == 0
                and stats['accounted_host_bytes'] == captures[1 if i == 0 else 2]['cache']['accounted_host_bytes'],
                'incorrect hit/miss/checkpoint accounting')
    require(len(data['survivor_rows']) == 3 and count == 176, 'missing survivors')
    for row,case in zip(data['survivor_rows'],coding[1:4]):
        require(row['id'] == case['id'] and row['generated_ids'] == row['expected_ids']
                == controls[case['id']]['generated_ids'][:count], 'survivor isolation/control divergence')
    leases = data['leases']
    require([l['request'] for l in leases] == list(range(1,11))
            and [l['row'] for l in leases] == [0,1,2,3,0,0,0,0,0,0]
            and all(type(l['epoch']) is int and l['epoch'] > 0 for l in leases)
            and len({l['epoch'] for l in leases}) == 10, 'incorrect lease identities')
    a_leases = [l['epoch'] for l in leases if l['row'] == 0]
    require(a_leases == sorted(a_leases), 'stale row epoch')
    require([r['epoch'] for r in records] == a_leases, 'case epoch differs from lease')
    restored = [0,0,84,0,115,0,64]
    epoch_start = dict(zip(a_leases,restored))
    last_epochs = [l['epoch'] for l in leases[:4]]
    positions = [0]*4
    require(data['frames'], 'missing native execution frames')
    for frame in data['frames']:
        epochs = frame['epochs']
        require(len(epochs) == 4 and epochs[1:] == last_epochs[1:] and epochs[0] in epoch_start,
                'survivor epoch changed')
        if epochs[0] != last_epochs[0]:
            positions[0] = epoch_start[epochs[0]]
        require(frame['before'] == positions, 'incorrect fresh/restored position')
        if frame['stage'] == 'prefill':
            chunk = frame['chunk']
            increments = frame['valid']
            require(chunk in (2,4,8,16,32,64,128) and chunk*4 <= 512
                    and len(increments) == 4 and all(type(v) is int and 0 <= v <= chunk for v in increments), 'invalid prefill mask')
        else:
            require(frame['stage'] == 'decode', 'unknown frame stage')
            increments = frame['active']
            require(len(increments) == 4 and all(v in (0,1) for v in increments), 'invalid decode mask')
        positions = [p+n for p,n in zip(positions,increments)]
        require(frame['after'] == positions and all(0 <= p <= 4096 for p in positions), 'incorrect committed position')
        last_epochs = epochs
    require(positions == [len(changed)+15]+[len(c['prompt_ids'])+175 for c in coding[1:4]], 'incorrect final positions')
    require(data['rejections'] == ['stale-restore','stale-capture'], 'missing stale-epoch rejections')
    close = data['cleanup']
    require(all(close[k] == 0 for k in OWNERS) and 0 <= close['metal_allocated_bytes'] <= 4*1024*1024, 'native cleanup failed')
    require(0 <= data['prepared']['metal_allocated_bytes'] <= data['prepared']['metal_recommended_working_set_bytes']-256*1024*1024,
            'prepared working-set gate failed')
    return dict(frozen_control_token_observations=608,cold_warm_token_observations=96,
                survivor_token_observations=528,host_cache_accounted_bytes=captures[-1]['cache']['accounted_host_bytes'],
                checkpoint_boundaries=[64,84,115],asserted_restore_boundaries=[0,84,115,64],
                close_allocation_bytes=close['metal_allocated_bytes'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe',type=Path,required=True)
    parser.add_argument('--control-provenance',type=Path,required=True)
    parser.add_argument('--control-provenance-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    files = {}

    def read(path):
        data = path.read_bytes()
        files[str(path.resolve())] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    provenance = read(args.control_provenance)
    require(files[str(args.control_provenance.resolve())] == args.control_provenance_sha256, 'changed control provenance')
    data,status = read(args.probe/'result.json'),read(args.probe/'status.json')
    fixture_path = Path(data['fixture'])
    fixture = read(fixture_path)
    controls = {c['id']:read(Path(data['controls'])/c['id']/'request-0.result.json')
                for c in fixture['cases'] if c['kind'] == 'coding'}
    for path,digest in files.items():
        if path not in (str(args.control_provenance.resolve()),str((args.probe/'result.json').resolve()),str((args.probe/'status.json').resolve())):
            require(provenance['files'].get(path) == digest, 'unfrozen fixture/control: '+path)
    report = validate(fixture,controls,data,status)
    rejected = []
    for label in ('incomplete','wrong-output','changed-cold-control','lying-parity','stale-epoch','wrong-position',
                  'wrong-checkpoint','over-budget','spurious-hit','changed-survivor','live-owner','missing-case'):
        altered = copy.deepcopy(data)
        if label == 'incomplete': altered['completed'] = False
        elif label == 'wrong-output': altered['records'][0]['generated_ids'][0] ^= 1
        elif label == 'changed-cold-control': altered['records'][3]['generated_ids'][0] ^= 1
        elif label == 'lying-parity': altered['records'][4]['exact_control'] = False
        elif label == 'stale-epoch': altered['frames'][-1]['epochs'][1] += 10
        elif label == 'wrong-position': altered['frames'][-1]['after'][0] += 1
        elif label == 'wrong-checkpoint': altered['captures'][0]['boundary'] += 1
        elif label == 'over-budget': altered['captures'][2]['cache']['accounted_host_bytes'] = 97*1024*1024
        elif label == 'spurious-hit': altered['records'][1]['cache']['hits'] += 1
        elif label == 'changed-survivor': altered['survivor_rows'][0]['generated_ids'][0] ^= 1
        elif label == 'live-owner': altered['cleanup'][OWNERS[0]] = 1
        else: altered['records'].pop()
        try:
            validate(fixture,controls,altered,status)
        except (ValueError,KeyError,IndexError):
            rejected.append(label)
        else:
            raise AssertionError('Accepted corruption: '+label)
    for path,expected in [(args.probe/'run_mistral_prefix_probe.py',status['script_sha256']),
                          (Path(data['library']),status['library_sha256'])]:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        require(digest == expected, 'changed driver/library')
        files[str(path.resolve())] = digest
    for path in (Path(__file__),Path(__file__).with_name('audit_mistral_http_admission.py')):
        files[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    report.update(collection_integrity_passed=True,exact_native_prefix_gate_passed=True,
                  corruptions_rejected=rejected,scope=__doc__,files=files)
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2)
        stream.write('\n')
    print(json.dumps({k:report[k] for k in ('exact_native_prefix_gate_passed','frozen_control_token_observations',
                                           'cold_warm_token_observations','host_cache_accounted_bytes')}))


if __name__ == '__main__':
    main()
