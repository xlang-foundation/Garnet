#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Corrupt real completed target/generation evidence and require audit rejection.

CPU-only; takes retained local artifacts, never starts an inference engine.
This exercises structural validators, not their CLI provenance bindings.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from audit_mistral_greedy_verification import validate as target_validate
from audit_mistral_medusa_generation import validate as generation_validate


def check_rejected(original, validate, mutate):
    changed = copy.deepcopy(original)
    mutate(changed)
    try:
        validate(changed)
    except (ValueError, KeyError, TypeError, IndexError):
        return
    raise AssertionError('corrupted evidence was accepted')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--generation', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--coding-control', type=Path, required=True)
    parser.add_argument('--context-control', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    files, rejected = {}, []
    def read(path):
        data = path.read_bytes();files[str(path.resolve())] = hashlib.sha256(data).hexdigest()
        return json.loads(data)
    target = read(args.target/'capacity4096/status.json')
    target_check = lambda value: target_validate(value,4096)
    target_check(target)
    def target_case(value, label='prefix2'):
        return next(row for row in value['records'] if row['label'] == label)
    target_corruptions = {
        'incomplete': lambda s:s.update(completed=False),
        'missing-case': lambda s:s['records'].pop(),
        'duplicate-case': lambda s:s['records'].append(s['records'][0]),
        'changed-profile': lambda s:s.update(capacity=256),
        'wrong-extension': lambda s:s.update(backend='unknown'),
        'changed-kv': lambda s:target_case(s)['state_sha256'][17].__setitem__(1,'0'*64),
        'changed-hidden': lambda s:target_case(s).update(hidden_sha256='0'*64),
        'non-hash-witness': lambda s:target_case(s).update(hidden_sha256='x'),
        'wrong-prefix': lambda s:target_case(s).update(consumed=1),
        'noncontiguous-slot': lambda s:target_case(s)['slots'].__setitem__(2,0),
        'fp16-route': lambda s:target_case(s)['metrics'].update(target_mpp_affine_dispatches=1),
        'missing-rollback': lambda s:target_case(s)['metrics'].update(commit_kv_restores=0),
        'broken-carry': lambda s:target_case(s,'carried-second')['submitted_ids'].__setitem__(0,-1),
        'missing-eos': lambda s:target_case(s,'eos0').update(stops=[]),
        'live-owner': lambda s:s['cleanup'].update(garnet_live_matrix_owners=1),
        'allocation-after-close': lambda s:s['cleanup'].update(metal_allocated_bytes=5*1024*1024),
        'negative-memory': lambda s:s['prepared'].update(metal_allocated_bytes=-1),
        'working-set-overflow': lambda s:s['prepared'].update(metal_allocated_bytes=10**15),
    }
    for label, mutate in target_corruptions.items():
        check_rejected(target,target_check,mutate);rejected.append('target:'+label)
    fixture = read(args.fixture)
    controls = {c['id']: read((args.coding_control if c['kind'] == 'coding' else args.context_control)
                            /c['id']/'request-0.result.json') for c in fixture['cases']}
    generation = read(args.generation/'status.json')
    generation_check = lambda value:generation_validate(value,fixture,controls)
    generation_check(generation)
    def speculative(value):
        return next(row for row in value['records'] if row['mode'] == 'medusa')
    def round0(value):
        return speculative(value)['rounds'][0]
    generation_corruptions = {
        'incomplete': lambda s:s.update(completed=False),
        'missing-case': lambda s:s['records'].pop(),
        'duplicate-case': lambda s:s['records'].append(s['records'][0]),
        'wrong-profile': lambda s:s['profile'].update(batch=4),
        'changed-budget': lambda s:speculative(s).update(budget=1),
        'changed-output': lambda s:speculative(s)['generated_ids'].__setitem__(0,-1),
        'changed-kv': lambda s:speculative(s)['state_sha256'][17].__setitem__(1,'0'*64),
        'changed-hidden': lambda s:speculative(s).update(hidden_sha256='0'*64),
        'wrong-cursor': lambda s:speculative(s).update(slots=0),
        'causal-root': lambda s:round0(s)['submitted_ids'].__setitem__(0,-1),
        'lost-emission': lambda s:round0(s)['emitted_ids'].pop(),
        'wrong-prefix': lambda s:round0(s).update(consumed_inputs=0),
        'missing-round': lambda s:speculative(s)['rounds'].pop(),
        'fp16-route': lambda s:round0(s)['metrics'].update(target_mpp_affine_dispatches=1),
        'negative-gpu-time': lambda s:round0(s)['metrics'].update(target_gpu_ms=-1),
        'changed-gpu-time-sum': lambda s:round0(s)['metrics'].update(sequential_gpu_duration_sum_ms=0),
        'nonfinite-time': lambda s:speculative(s).update(decode_seconds=float('nan')),
        'wrong-token-rate': lambda s:speculative(s).update(decode_tokens_per_second=0.1),
        'negative-memory': lambda s:speculative(s)['memory'].update(metal_allocated_bytes=-1),
        'live-owner': lambda s:s['cleanup'].update(garnet_live_matrix_owners=1),
        'allocation-after-close': lambda s:s['cleanup'].update(metal_allocated_bytes=5*1024*1024),
    }
    for label, mutate in generation_corruptions.items():
        check_rejected(generation,generation_check,mutate);rejected.append('generation:'+label)
    for name in ('audit_mistral_greedy_verification.py','audit_mistral_medusa_generation.py',Path(__file__).name):
        path=Path(__file__).with_name(name);files[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    with args.output.open('x') as stream:
        json.dump(dict(completed=True,rejected=rejected,files=files,scope=__doc__),stream,indent=2);stream.write('\n')
    print('rejected_corruptions',len(rejected))
