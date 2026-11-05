#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit saved profile runs; distinguish collection integrity from exact parity.

The recorded masks/positions are driver traces, not independent observations of
GPU KV contents. Operator tests must separately establish masked state writes.
No inference engine is loaded by this auditor.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path


OWNER_KEYS = ('garnet_live_matrix_owners', 'garnet_live_affine_weight_owners',
              'garnet_live_bf16_weight_owners', 'garnet_live_pipeline_sets')


def require(value, message):
    if not value:
        raise ValueError(message)


def validate(fixture, status, control_statuses, controls):
    require(status['completed'] is True, 'collection incomplete')
    backend = {part.strip() for part in status['backend'].split(';')}
    require({'prepared_dense_prefill', 'dense_prefill_max_tokens=512',
             'affine_coefficient_dtype=fp16', 'compact_attention_index'} <= backend,
            'wrong native engine profile')
    batch, maximum = status['batch'], status['chunk']
    require(type(batch) is int and batch in (1, 2, 4), 'invalid batch')
    require(type(maximum) is int and maximum in (2, 4, 8, 16, 32, 64, 128, 256, 512)
            and batch*maximum <= 512 and status['capacity'] == 4096, 'invalid profile')
    require(status['kind'] in ('coding', 'context', 'all'), 'invalid case kind')
    require(status['policy'] in ('full', 'staggered')
            and (status['policy'] == 'full' or batch > 1), 'invalid budgets policy')
    require(all(status['cleanup'][key] == 0 for key in OWNER_KEYS), 'live native owner')
    for name in ('coding', 'context'):
        control = control_statuses[name]
        require(control['completed'] is True and control['batch'] == 1
                and control['capacity'] == 4096 and control['chunk'] == 16,
                'invalid frozen native control')
        require(all(control['cleanup'][key] == 0 for key in OWNER_KEYS), 'control cleanup failed')
    cases = [case for case in fixture['cases'] if status['kind'] == 'all' or case['kind'] == status['kind']]
    require(cases and len(status['records']) == (len(cases)+batch-1)//batch, 'missing groups')
    expected_profiles = [n for n in (2, 4, 8, 16, 32, 64, 128, 256, 512) if n <= maximum]
    graph_count = len(expected_profiles)*32
    require(status['prepared']['graph_count'] == graph_count, 'unprepared graph family')
    details = []
    for group_index, record in enumerate(status['records']):
        selected = cases[group_index*batch:(group_index+1)*batch]
        selected += [selected[-1]]*(batch-len(selected))
        require(record['group'] == group_index and len(record['rows']) == batch, 'invalid group identity')
        require(record['graph_count'] == graph_count, 'request compiled graphs')
        require(math.isfinite(record['elapsed_seconds']) and record['elapsed_seconds'] >= 0, 'invalid duration')
        lengths = [len(case['prompt_ids']) for case in selected]
        for case in selected:
            require(case['prompt_tokens'] == len(case['prompt_ids']) and
                    0 < case['prompt_tokens']+case['max_tokens'] <= 4096, 'fixture exceeds capacity')
        offset = 0
        for frame in record['prefill_frames']:
            require(offset < max(lengths), 'extra prefill frame')
            remaining = max(lengths)-offset
            chunk = maximum
            if maximum > 16 and remaining < maximum:
                chunk = max(2, 1 << (remaining-1).bit_length())
            valid = [min(chunk, max(0, length-offset)) for length in lengths]
            require(frame['offset'] == offset and frame['chunk'] == chunk
                    and chunk*batch <= 512 and frame['valid'] == valid, 'invalid prefill mask/budget')
            require(frame['slots_before'] == [min(offset, length) for length in lengths]
                    and frame['slots_after'] == [min(offset+chunk, length) for length in lengths],
                    'incorrect prefill row positions')
            require(frame['kv_inplace_updates'] == 64, 'incomplete prefill in-place coverage')
            offset += chunk
        require(offset >= max(lengths) and record['prefill_frames'], 'incomplete prefill')
        require(len(record['first_logits']) == batch*32768
                and all(type(v) in (int, float) and math.isfinite(v) for v in record['first_logits']),
                'invalid initial logits')
        for row_index, (case, row) in enumerate(zip(selected, record['rows'])):
            require(row['id'] == case['id'] and row['row'] == row_index, 'reordered/missing case')
            require(row['prompt_ids'] == case['prompt_ids'], 'changed prompt IDs')
            budget = min(8, case['max_tokens']) if status['policy'] == 'staggered' and row_index == 0 else case['max_tokens']
            require(row['budget'] == budget, 'changed generation budget')
            ids = row['generated_ids']
            require(0 < len(ids) <= budget and all(type(i) is int and 0 <= i < 32768 for i in ids), 'invalid output IDs')
            stopped = ids[-1] in fixture['stops']
            require(not any(i in fixture['stops'] for i in ids[:-1]), 'continued after EOS')
            require(row['finish_reason'] == ('stop' if stopped else 'length')
                    and (stopped or len(ids) == budget), 'wrong finish/count')
            require(row['slots'] == len(case['prompt_ids'])+len(ids)-1, 'incorrect final KV position')
            logits = record['first_logits'][row_index*32768:(row_index+1)*32768]
            require(max(range(32768), key=logits.__getitem__) == ids[0], 'first token/logits disagree')
            control = controls[case['id']]
            require(control['status'] == 'completed' and control['prompt_ids'] == case['prompt_ids'], 'changed control prompt')
            expected = control['generated_ids'][:budget]
            common = next((i for i, (a, b) in enumerate(zip(ids, expected)) if a != b), min(len(ids), len(expected)))
            exact = ids == expected
            require(row['exact_native_control'] is exact and row['matching_prefix_tokens'] == common, 'incorrect parity claim')
            if ids == control['generated_ids']:
                require(row['text'] == control['text'], 'equal IDs decoded differently')
            details.append(dict(group=group_index, row=row_index, id=case['id'], budget=budget,
                                output_tokens=len(ids), exact_native_control=exact, matching_prefix_tokens=common))
        output_lengths = [len(row['generated_ids']) for row in record['rows']]
        require(len(record['decode_frames']) == max(output_lengths)-1, 'incorrect decode frame count')
        for step, frame in enumerate(record['decode_frames']):
            active = [int(length > step+1) for length in output_lengths]
            before = [lengths[row]+min(step, length-1) for row, length in enumerate(output_lengths)]
            require(frame['active'] == active and frame['slots_before'] == before
                    and frame['slots_after'] == [before[row]+active[row] for row in range(batch)],
                    'incorrect active/inactive decode positions')
            require(frame['kv_inplace_updates'] == 64, 'incomplete decode in-place coverage')
    return details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--coding-control', type=Path, required=True)
    parser.add_argument('--context-control', type=Path, required=True)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--control-provenance', type=Path, required=True)
    parser.add_argument('--control-provenance-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    files = []

    def read(path):
        files.append(path)
        return json.loads(path.read_text())

    require(hashlib.sha256(args.control_provenance.read_bytes()).hexdigest() ==
            args.control_provenance_sha256, 'changed control provenance')
    provenance = read(args.control_provenance)
    fixture, status = read(args.fixture), read(args.native/'status.json')
    roots = dict(coding=args.coding_control, context=args.context_control)
    control_statuses = {name: read(root/'status.json') for name, root in roots.items()}
    controls = {case['id']: read(roots[case['kind']]/case['id']/'request-0.result.json') for case in fixture['cases']}
    # Bind the controls to the earlier immutable native collection, rather than
    # accepting newly generated controls or silently replacing a divergence.
    for path in files:
        if path not in (args.control_provenance, args.native/'status.json'):
            require(provenance['files'].get(str(path.resolve())) == hashlib.sha256(path.read_bytes()).hexdigest(),
                    'unfrozen/changed native control: '+str(path))
    details = validate(fixture, status, control_statuses, controls)
    rejected = []
    for label in ('incomplete', 'wrong-profile', 'missing-group', 'unprepared-graph', 'wrong-prompt',
                  'wrong-budget', 'wrong-slot', 'wrong-mask', 'nan-logit', 'false-parity',
                  'live-owner', 'failed-control', 'short-output', 'missing-inplace'):
        altered = copy.deepcopy(status)
        cs = copy.deepcopy(control_statuses)
        row = altered['records'][0]['rows'][0]
        if label == 'incomplete': altered['completed'] = False
        elif label == 'wrong-profile': altered['chunk'] = 1024
        elif label == 'missing-group': altered['records'].pop()
        elif label == 'unprepared-graph': altered['records'][0]['graph_count'] += 32
        elif label == 'wrong-prompt': row['prompt_ids'][0] ^= 1
        elif label == 'wrong-budget': row['budget'] += 1
        elif label == 'wrong-slot': row['slots'] += 1
        elif label == 'wrong-mask': altered['records'][0]['prefill_frames'][0]['valid'][0] -= 1
        elif label == 'nan-logit': altered['records'][0]['first_logits'][0] = math.nan
        elif label == 'false-parity': row['exact_native_control'] = not row['exact_native_control']
        elif label == 'live-owner': altered['cleanup'][OWNER_KEYS[0]] = 1
        elif label == 'failed-control': cs['coding']['completed'] = False
        elif label == 'missing-inplace': altered['records'][0]['prefill_frames'][0]['kv_inplace_updates'] = 0
        else: row['generated_ids'] = []
        try:
            validate(fixture, altered, cs, controls)
        except (ValueError, KeyError, IndexError):
            rejected.append(label)
        else:
            raise AssertionError('Accepted corruption: '+label)
    files.append(Path(__file__))
    close_bytes = status['cleanup']['metal_allocated_bytes']
    require(type(close_bytes) is int and close_bytes >= 0, 'invalid close allocation snapshot')
    snapshots = [status['prepared']['memory']]+[record['memory'] for record in status['records']]
    for snapshot in snapshots:
        require(type(snapshot['metal_allocated_bytes']) is int and snapshot['metal_allocated_bytes'] >= 0
                and type(snapshot['metal_recommended_working_set_bytes']) is int
                and snapshot['metal_recommended_working_set_bytes'] > 256*1024*1024,
                'invalid working-set snapshot')
    working_set_gate = all(snapshot['metal_allocated_bytes'] <= snapshot['metal_recommended_working_set_bytes']-256*1024*1024
                           for snapshot in snapshots)
    report = dict(collection_integrity_passed=True,
                  exact_native_control_gate_passed=all(row['exact_native_control'] for row in details),
                  sampled_working_set_gate_passed=working_set_gate,
                  close_allocation_gate_passed=close_bytes <= 4*1024*1024,
                  close_allocation_bytes=close_bytes, close_allocation_limit_bytes=4*1024*1024,
                  rows=details, corruptions_rejected=rejected,
                  scope='Saved driver trace/ID integrity and observed native-control parity; no independent GPU KV snapshot, quality or speed claim',
                  files={str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in files})
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({key: report[key] for key in ('collection_integrity_passed', 'exact_native_control_gate_passed')}))


if __name__ == '__main__':
    main()
