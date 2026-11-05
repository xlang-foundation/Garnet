#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independently audit saved native cases without starting an inference engine."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def require(value, message):
    if not value:
        raise ValueError(message)


def validate(fixture, reference_status, status, rows, references, logits):
    require(reference_status['status'] == 'completed', 'reference incomplete')
    require(status['completed'] is True and status['batch'] == 1, 'native collection incomplete')
    require('affine_coefficient_dtype=fp16' in status['backend'], 'wrong coefficient engine')
    capacity, chunk = status['capacity'], status['chunk']
    require(capacity in (2048, 4096) and chunk in (0, 2, 4, 8, 16), 'wrong profile')
    if capacity == 4096 and chunk:
        require('compact_attention_index' in status['backend'], '4K parallel attention not enabled')
    owner_keys = ('garnet_live_matrix_owners', 'garnet_live_affine_weight_owners',
                  'garnet_live_bf16_weight_owners', 'garnet_live_pipeline_sets')
    require(all(status['cleanup'][key] == 0 for key in owner_keys), 'live native owner after close')
    identities = [record['id'] for record in status['records']]
    require(len(set(identities)) == len(identities) and identities, 'invalid native case list')
    kinds = {case['kind'] for case in fixture['cases'] if case['id'] in identities}
    require(len(kinds) == 1, 'mixed or missing case kinds')
    selected = [case for case in fixture['cases'] if case['kind'] in kinds]
    require(identities == [case['id'] for case in selected], 'missing/reordered native case')
    details = []
    for case, record in zip(selected, status['records']):
        identity = case['id']
        row, ref = rows[identity], references[identity]
        for result in (row, ref):
            require(result['status'] == 'completed', 'failed generation')
            require(result['prompt_ids'] == case['prompt_ids'], 'changed prompt IDs')
            ids = result['generated_ids']
            require(0 < len(ids) <= case['max_tokens'] and all(type(i) is int and 0 <= i < 32768 for i in ids), 'invalid token sequence')
            stopped = ids[-1] in fixture['stops']
            require(not any(i in fixture['stops'] for i in ids[:-1]), 'generation continued after EOS')
            require(result['finish_reason'] == ('stop' if stopped else 'length'), 'wrong finish reason')
            require(stopped or len(ids) == case['max_tokens'], 'short non-EOS output')
        ids, expected = row['generated_ids'], ref['generated_ids']
        require(case['prompt_tokens']+case['max_tokens'] <= capacity, 'profile capacity exceeded')
        require(row['slots'] == [len(case['prompt_ids'])+len(ids)-1], 'incorrect KV position')
        common = 0
        for a, b in zip(ids, expected):
            if a != b:
                break
            common += 1
        match = ids == expected
        if match:
            require(row['text'] == ref['text'], 'equal IDs decoded to different text')
        for value in (row, record):
            require(value['full_reference_token_match'] is match and value['matching_prefix_tokens'] == common, 'incorrect parity claim')
        require(record['prompt_tokens'] == case['prompt_tokens'] and record['output_tokens'] == len(ids)
                and record['finish_reason'] == row['finish_reason'], 'incorrect status summary')
        actual = np.asarray(row['first_logits'], dtype=np.float64)
        expected_logits = np.asarray(logits[identity], dtype=np.float64)
        # Native Tensor.tolist() exposes its flat storage; MLX preserves the
        # leading batch dimension. Require each representation explicitly.
        require(actual.shape == (32768,) and expected_logits.shape == (1, 32768), 'invalid first logits shape')
        expected_logits = expected_logits[0]
        require(np.isfinite(actual).all() and np.isfinite(expected_logits).all(), 'nonfinite first logits')
        require(int(actual.argmax()) == ids[0] and int(expected_logits.argmax()) == expected[0], 'first prediction/logits disagree')
        details.append(dict(id=identity, prompt_tokens=case['prompt_tokens'], output_tokens=len(ids),
                            reference_tokens=len(expected), exact_token_match=match, matching_prefix_tokens=common,
                            first_token_match=ids[0] == expected[0],
                            first_logits_max_abs=float(np.abs(actual-expected_logits).max()),
                            first_logits_mean_abs=float(np.abs(actual-expected_logits).mean())))
    return details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = [args.fixture, args.reference/'status.json', args.native/'status.json']
    fixture, reference_status, status = [json.loads(path.read_text()) for path in paths]
    require(reference_status['fixture_sha256'] == hashlib.sha256(args.fixture.read_bytes()).hexdigest(), 'reference fixture drift')
    rows, references, logits = {}, {}, {}
    for record in status['records']:
        identity = record['id']
        native = args.native/identity/'request-0.result.json'
        ref = args.reference/identity/'request-0.result.json'
        logit = args.reference/identity/'first-logits.npy'
        paths.extend((native, ref, logit))
        rows[identity], references[identity] = [json.loads(path.read_text()) for path in (native, ref)]
        logits[identity] = np.load(logit, allow_pickle=False)
    details = validate(fixture, reference_status, status, rows, references, logits)
    rejected = []
    identity = status['records'][0]['id']
    for label in ('wrong-token-claim', 'wrong-prompt', 'missing-case', 'live-owner', 'short-output',
                  'nonfinite-logits', 'wrong-slot', 'wrong-profile', 'failed-reference'):
        f, rs, s, r = [copy.deepcopy(value) for value in (fixture, reference_status, status, rows)]
        if label == 'wrong-token-claim':
            r[identity]['full_reference_token_match'] = not r[identity]['full_reference_token_match']
        elif label == 'wrong-prompt':
            r[identity]['prompt_ids'][0] ^= 1
        elif label == 'missing-case':
            s['records'].pop()
        elif label == 'live-owner':
            s['cleanup']['garnet_live_matrix_owners'] = 1
        elif label == 'short-output':
            r[identity]['generated_ids'] = r[identity]['generated_ids'][:1]
            r[identity]['finish_reason'] = 'length'
        elif label == 'nonfinite-logits':
            r[identity]['first_logits'][0] = math.nan
        elif label == 'wrong-slot':
            r[identity]['slots'][0] += 1
        elif label == 'wrong-profile':
            s['backend'] = 'BF16'
        else:
            rs['status'] = 'running'
        try:
            validate(f, rs, s, r, references, logits)
        except (ValueError, KeyError):
            rejected.append(label)
        else:
            raise AssertionError('Accepted corruption: '+label)
    report = dict(collection_audit_passed=True, cases=details, corruptions_rejected=rejected,
                  all_sequences_exact=all(row['exact_token_match'] for row in details),
                  scope='Collection integrity and observed parity; sandbox coding quality is separate',
                  files={str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
                  auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(audit_passed=True, cases=len(details),
                          exact_sequences=sum(row['exact_token_match'] for row in details),
                          corruptions_rejected=len(rejected))))


if __name__ == '__main__':
    main()
