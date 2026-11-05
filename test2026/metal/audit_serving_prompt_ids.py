#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Replay saved serving-tokenizer evidence against every frozen prompt ID."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re


def require(value, message):
    if not value:
        raise ValueError(message)


def validate(report, expected, raw):
    require(report['passed'] is True, 'proof did not complete')
    identities = [(row['id'], row['row']) for row in report['records']]
    require(identities == list(expected), 'missing, duplicated or reordered prompt proof')
    for row in report['records']:
        identity = row['id'], row['row']
        body, ids = expected[identity]
        data = raw[identity]
        value = json.loads(data)
        require(hashlib.sha256(data).hexdigest() == row['response_sha256'], 'changed raw response')
        require(row['request'] == body, 'changed chat request')
        require(value['tokens'] == ids and value['count'] == len(ids), 'serving prompt IDs/count differ')
        require(row['exact_ids_match'] is True and row['actual_tokens'] == len(ids)
                and row['expected_tokens'] == len(ids), 'incorrect proof summary')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proof', type=Path, required=True)
    parser.add_argument('--manifests', type=Path, nargs='+', required=True)
    parser.add_argument('--server-log', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    expected, paths = {}, [args.proof/'report.json', args.server_log]+args.manifests
    for path in args.manifests:
        manifest = json.loads(path.read_text())
        for plan in manifest['plans']:
            cases_path, workload_path = Path(plan['cases']), Path(plan['workload'])
            cases, workload = [json.loads(p.read_text()) for p in (cases_path, workload_path)]
            paths.extend((cases_path, workload_path))
            digest = hashlib.sha256(cases_path.read_bytes()).hexdigest()
            require(digest == plan['cases_sha256'] == workload['cases_sha256'], 'fixture drift')
            require(len(cases) == len(workload['requests']), 'request count drift')
            for index, (case, record) in enumerate(zip(cases, workload['requests'])):
                identity = plan['id'], index
                require(identity not in expected, 'duplicate fixture identity')
                body = {name: case[name] for name in
                        ('model', 'messages', 'tools', 'tool_choice', 'chat_template_kwargs') if name in case}
                expected[identity] = body, record['token_ids']
    report = json.loads((args.proof/'report.json').read_text())
    raw = {}
    for identity in expected:
        path = args.proof/(identity[0]+'-'+str(identity[1])+'.response.json')
        paths.append(path)
        raw[identity] = path.read_bytes()
    validate(report, expected, raw)
    logged = len(re.findall(r'"POST /tokenize HTTP/1\.1" 200', args.server_log.read_text()))
    require(logged >= len(expected), 'server lacks successful tokenization request evidence')
    rejected = []
    first = next(iter(expected))
    for label in ('changed-id', 'changed-count', 'missing-prompt', 'changed-request', 'failed-proof'):
        r, data = copy.deepcopy(report), copy.deepcopy(raw)
        if label in ('changed-id', 'changed-count'):
            value = json.loads(data[first])
            if label == 'changed-id':
                value['tokens'][0] ^= 1
            else:
                value['count'] += 1
            data[first] = json.dumps(value).encode()
            r['records'][0]['response_sha256'] = hashlib.sha256(data[first]).hexdigest()
        elif label == 'missing-prompt':
            r['records'].pop()
        elif label == 'changed-request':
            r['records'][0]['request']['messages'][0]['content'] += ' altered'
        else:
            r['passed'] = False
        try:
            validate(r, expected, data)
        except ValueError:
            rejected.append(label)
        else:
            raise AssertionError('Accepted corruption: '+label)
    result = dict(passed=True, exact_requests=len(expected),
                  compared_prompt_ids=sum(len(value[1]) for value in expected.values()),
                  logged_tokenize_requests=logged, corruptions_rejected=rejected,
                  files={str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
                  auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='Full serving chat prompt-ID equality only; not inference, quality or performance admission')
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(passed=True, exact_requests=len(expected), corruptions_rejected=len(rejected))))


if __name__ == '__main__':
    main()
