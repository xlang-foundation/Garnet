#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit causal single-path head proposals against frozen ordinary native outputs.

The batch1 prototype differs from the frozen B4 HTTP profile. Timings are one
run/mode/case with no warmups and trace overhead, not serving qualification.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re


def require(value, message):
    if not value:
        raise ValueError(message)


def state_hashes(record):
    value = record['state_sha256']
    return len(value) == 32 and all(len(parts) == 2 and all(re.fullmatch('[0-9a-f]{64}', h)
        for h in parts) for parts in value) and bool(re.fullmatch('[0-9a-f]{64}', record['hidden_sha256']))


def validate(status, fixture, controls):
    require(status['completed'] is True and status['profile'] == dict(batch=1, capacity=4096,
        prefill_chunk=128, verification_tokens=4, prefix_cache=False), 'incomplete/wrong prototype profile')
    require('native_greedy_dense_verification' in status['backend'] and 'medusa_residual_head_bf16' in status['backend'],
            'wrong native draft/verification extension')
    expected = {c['id']: c for c in fixture['cases']}
    require(len(status['records']) == 2*len(expected), 'missing/extra prototype cases')
    by_case = {}
    for record in status['records']:
        identity, mode = record['id'], record['mode']
        require(identity in expected and mode in ('ordinary', 'medusa') and (identity, mode) not in by_case,
                'unknown/duplicate prototype case')
        case, control = expected[identity], controls[identity]
        ids = record['generated_ids']
        require(control['status'] == 'completed' and record['prompt_ids'] == case['prompt_ids'] == control['prompt_ids']
                and record['budget'] == case['max_tokens'] and ids == control['generated_ids'], 'frozen native output/contract changed')
        require(record['slots'] == len(case['prompt_ids'])+len(ids)-1 and state_hashes(record), 'invalid cursor or state witness')
        require(record['finish_reason'] == ('stop' if ids[-1] in fixture['stops'] else 'length')
                and not any(t in fixture['stops'] for t in ids[:-1]), 'wrong stop behavior')
        require(all(type(record[key]) in (int, float) and math.isfinite(record[key]) and record[key] > 0
                    for key in ('prefill_seconds', 'decode_seconds', 'decode_tokens_per_second')), 'invalid native timing')
        require(abs(record['decode_tokens_per_second']-(len(ids)-1)/record['decode_seconds']) < 1e-9,
                'wrong decode token accounting')
        rounds, position, emitted = record['rounds'], len(case['prompt_ids']), 1
        if mode == 'ordinary':
            require(not rounds, 'ordinary mode contains draft rounds')
        else:
            require(bool(rounds) == (len(ids) > 1), 'missing trained-head rounds')
            for run in rounds:
                count, length = run['consumed_inputs'], run['active_tokens']
                require(run['slot'] == position and length == min(4, case['max_tokens']-emitted)
                        and 1 <= count <= length and len(run['submitted_ids']) == 4
                        and run['submitted_ids'][0] == ids[emitted-1]
                        and run['emitted_ids'] == ids[emitted:emitted+count]
                        and len(run['emitted_ids']) == count
                        and run['accepted_proposal_inputs'] == count-1, 'invalid causal proposal/commit trace')
                require(all(run['submitted_ids'][j] == run['emitted_ids'][j-1] for j in range(1, count)),
                        'committed proposal does not match target')
                if count < length and run['emitted_ids'][-1] not in fixture['stops']:
                    require(run['emitted_ids'][-1] != run['submitted_ids'][count], 'rejected an accepted proposal')
                stats = run['metrics']
                if 'mistral_decode_reuse_four_rows' in status['backend']:
                    require(stats['target_affine_decode_block_dispatches'] == (161 if length == 4 else 1),
                            'Mistral four-row projection reuse route differs')
                require(all(type(stats[key]) in (int, float) and math.isfinite(stats[key]) and stats[key] >= 0
                            for key in ('target_gpu_ms', 'commit_gpu_ms', 'sequential_gpu_duration_sum_ms'))
                        and stats['target_gpu_ms'] > 0
                        and abs(stats['sequential_gpu_duration_sum_ms']-stats['target_gpu_ms']-stats['commit_gpu_ms']) < 1e-6,
                        'invalid per-round sequential GPU durations')
                for key, value in [('target_affine_dispatches',225), ('target_affine_misses',0),
                    ('target_mpp_affine_dispatches',0), ('target_indexed_prefill_dispatches',0),
                    ('target_reuse64_prefill_dispatches',0), ('target_kv_backups',64), ('target_attention_dispatches',32),
                    ('commit_kv_restores',64 if count < length else 0)]:
                    require(stats[key] == value, 'wrong exact target/rollback route: '+key)
                require(stats['consumed_inputs'] == [count] and stats['emitted_tokens'] == [run['emitted_ids']],
                        'native result differs from prototype trace')
                position += count;emitted += count
            require(emitted == len(ids) and position == record['slots'], 'lost/extra committed tokens')
        require(0 <= record['memory']['metal_allocated_bytes'] <= record['memory']['metal_recommended_working_set_bytes']-256*1024*1024,
                'sampled native working set exceeded guard')
        by_case[identity, mode] = record
    comparisons = []
    for identity, case in expected.items():
        ordinary, medusa = by_case[identity, 'ordinary'], by_case[identity, 'medusa']
        require(ordinary['state_sha256'] == medusa['state_sha256'] and ordinary['hidden_sha256'] == medusa['hidden_sha256'],
                'complete ordinary/speculative state differs')
        rounds = medusa['rounds']
        inputs = sum(r['active_tokens']-1 for r in rounds)
        accepted = sum(r['accepted_proposal_inputs'] for r in rounds)
        target_gpu = sum(r['metrics']['target_gpu_ms']+r['metrics']['commit_gpu_ms'] for r in rounds)/1000
        require(math.isfinite(target_gpu) and 0 <= target_gpu <= medusa['decode_seconds'], 'invalid sequential GPU duration sum')
        comparisons.append(dict(id=identity, output_tokens=len(ordinary['generated_ids']), rounds=len(rounds),
            committed_inputs_per_round=(len(medusa['generated_ids'])-1)/len(rounds) if rounds else None,
            proposed_inputs=inputs, accepted_proposal_inputs=accepted, accepted_proposal_input_fraction=accepted/inputs if inputs else None,
            ordinary_decode_tps=ordinary['decode_tokens_per_second'], medusa_decode_tps=medusa['decode_tokens_per_second'],
            descriptive_decode_speedup=ordinary['decode_seconds']/medusa['decode_seconds'],
            target_and_commit_gpu_seconds_sum=target_gpu,
            draft_host_and_other_unattributed_seconds=medusa['decode_seconds']-target_gpu))
    for key in ('garnet_live_matrix_owners', 'garnet_live_affine_weight_owners',
                'garnet_live_bf16_weight_owners', 'garnet_live_pipeline_sets'):
        require(status['cleanup'][key] == 0, 'native owners retained')
    require(0 <= status['cleanup']['metal_allocated_bytes'] <= 4*1024*1024, 'close allocation guard exceeded')
    return comparisons


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--coding-control', type=Path, required=True)
    parser.add_argument('--context-control', type=Path, required=True)
    parser.add_argument('--collector-status', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    files = {}
    def read(path):
        data = path.read_bytes();files[str(path.resolve())] = hashlib.sha256(data).hexdigest()
        return json.loads(data)
    fixture = read(args.fixture)
    controls = {c['id']: read((args.coding_control if c['kind'] == 'coding' else args.context_control)
                            /c['id']/'request-0.result.json') for c in fixture['cases']}
    status = read(args.native/'status.json')
    collector = read(args.collector_status)
    require(collector['returncode'] == 0 and type(collector['pid']) is int and collector['pid'] > 0
            and collector['finished_ns'] > collector['started_ns'], 'native collector failed or not terminal')
    require(Path(collector['argv'][-1]).resolve() == args.native.resolve()
            and Path(collector['argv'][7]).resolve() == args.fixture.resolve()
            and Path(collector['argv'][8]).resolve() == args.coding_control.resolve()
            and Path(collector['argv'][9]).resolve() == args.context_control.resolve(), 'collector inputs/output differ')
    library = Path(collector['argv'][5])/'libgarnet.dylib'
    with library.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    require(digest == collector['library_sha256'] and status['library'] == str(library), 'native library changed/differs')
    files[str(library)] = digest
    comparisons = validate(status, fixture, controls)
    files[str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = dict(native_causal_generation_passed=True, comparisons=comparisons, files=files, scope=__doc__,
        full_B4_HTTP_serving_qualified=False, speed_targets_established=False,
        artifact_training_base_identity_verified=False)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2);stream.write('\n')
    print('native_causal_generation_passed', True)
