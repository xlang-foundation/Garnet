#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit native target-verification witnesses; oracle proposals are not draft performance."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def require(value, message):
    if not value:
        raise ValueError(message)


def validate(status, capacity):
    require(status['completed'] is True and status['capacity'] == capacity and status['batch'] == 1
            and status['verification_tokens'] == 4, 'incomplete/wrong target profile')
    backend = status['backend']
    require('native_greedy_dense_verification' in backend and 'affine_coefficient_dtype=fp16' in backend
            and 'linear_chunk_conv' not in backend and 'linear_chunk_delta' not in backend, 'wrong dense native extension policy')
    expected = {'prefix1', 'prefix2', 'prefix3', 'prefix4', 'partial1', 'partial2', 'partial3',
                'eos0', 'eos3', 'carried-first', 'carried-second'}
    if capacity == 4096:
        expected |= {'full-prompt-prefix2', 'full-prompt-prefix4'}
    require(len(status['records']) == len(expected) and {r['label'] for r in status['records']} == expected,
            'missing/duplicate target cases')
    for record in status['records']:
        length, consumed = record['active_tokens'], record['consumed']
        require(1 <= consumed <= length <= 4 and len(record['emitted']) == consumed
                and all(type(t) is int and 0 <= t < 32768 for t in record['emitted']), 'invalid accepted output')
        require(len(record['submitted_ids']) == len(record['slots']) == 4
                and record['slots'] == [record['slots'][0]+i for i in range(4)]
                and 0 <= record['slots'][0] and record['slots'][length-1] < capacity, 'invalid verifier slot span')
        state = record['state_sha256']
        require(len(state) == 32 and all(len(parts) == 2 and all(re.fullmatch('[0-9a-f]{64}', h) for h in parts)
                                       for parts in state) and state == record['reference_state_sha256'],
                'complete KV state differs')
        require(re.fullmatch('[0-9a-f]{64}', record['hidden_sha256'])
                and record['hidden_sha256'] == record['reference_hidden_sha256'], 'accepted hidden differs')
        stats = record['metrics']
        if 'mistral_decode_reuse_four_rows' in status['backend']:
            require(stats['target_affine_decode_block_dispatches'] == (161 if length == 4 else 1),
                    'Mistral four-row projection reuse route differs')
        for key, wanted in [('target_affine_dispatches',225), ('target_affine_misses',0), ('target_kv_backups',64),
                            ('target_attention_dispatches',32), ('target_conv_checkpoints',0), ('target_delta_checkpoints',0),
                            ('target_mpp_affine_dispatches',0), ('target_indexed_prefill_dispatches',0),
                            ('target_reuse64_prefill_dispatches',0), ('borrowed_s1_banks',0), ('commit_replay_layers',0),
                            ('commit_kv_restores',64 if consumed < length else 0)]:
            require(stats[key] == wanted, 'FP32 target/rollback route differs: '+key)
        if record['label'].startswith('prefix'):
            require(consumed == int(record['label'][-1]), 'accepted prefix depth differs')
        if record['label'].startswith('partial'):
            require(consumed == length == int(record['label'][-1]), 'partial span differs')
        if record['label'] == 'eos0':
            require(consumed == 1 and record['emitted'][-1] in record['stops'], 'first-position EOS missing')
        if record['label'] == 'eos3':
            require(consumed == 4 and record['emitted'][-1] in record['stops'], 'last-position EOS missing')
    by_label = {r['label']: r for r in status['records']}
    first, second = by_label['carried-first'], by_label['carried-second']
    require(second['slots'][0] == first['slots'][0]+first['consumed']
            and second['submitted_ids'][0] == first['emitted'][-1], 'continued accepted state was not carried')
    memory = status['prepared']
    require(0 <= memory['metal_allocated_bytes'] <= memory['metal_recommended_working_set_bytes']-256*1024*1024,
            'sampled working-set guard exceeded')
    for key in ('garnet_live_matrix_owners', 'garnet_live_affine_weight_owners',
                'garnet_live_bf16_weight_owners', 'garnet_live_pipeline_sets'):
        require(status['cleanup'][key] == 0, 'native owners retained')
    require(0 <= status['cleanup']['metal_allocated_bytes'] <= 4*1024*1024, 'close allocation guard exceeded')
    return dict(capacity=capacity, cases=len(expected), full_KV_and_hidden_parity=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('collection', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.collection.resolve()
    collection = json.loads((root/'collector-status.json').read_text())
    require(collection['completed'] is True and len(collection['runs']) == 2, 'incomplete serial target collection')
    previous, pids, reports, files = 0, set(), [], {}
    def bind(path):
        with path.open('rb') as stream:
            files[str(path)] = hashlib.file_digest(stream, 'sha256').hexdigest()
    for capacity, run in zip((256,4096), collection['runs']):
        require(run['capacity'] == capacity and run['returncode'] == 0 and run['pid'] not in pids
                and run['started_ns'] >= previous and run['finished_ns'] > run['started_ns'], 'failed/overlapping target process')
        previous = run['finished_ns'];pids.add(run['pid'])
        path = root/f'capacity{capacity}/status.json'
        status = json.loads(path.read_text())
        reports.append(validate(status, capacity));bind(path)
    bind(root/'collector-status.json');bind(Path(__file__).resolve())
    report = dict(native_target_verification_passed=True, profiles=reports, files=files,
        trained_draft_acceptance_established=False, coding_quality_established=False, speed_established=False,
        scope=__doc__)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2);stream.write('\n')
    print('native_target_verification_passed', True)
