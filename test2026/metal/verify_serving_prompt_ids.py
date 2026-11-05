#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare real serving chat-tokenization IDs with frozen request fixtures.

Use during admission, never concurrently with a repeated speed campaign.
The /tokenize messages route follows the engine's chat-completion formatter.
"""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--engine', required=True)
    parser.add_argument('--manifests', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.url.startswith('http://127.0.0.1:'):
        raise ValueError('Loopback admission endpoint required')
    plans = []
    for manifest in args.manifests:
        plans.extend(json.loads(manifest.read_text())['plans'])
    identities = [row['id'] for row in plans]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate fixture identities')
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(passed=False, engine=args.engine, endpoint=args.url+'/tokenize',
                  manifest_sha256={str(path.resolve()): digest(path) for path in args.manifests},
                  verifier_sha256=digest(Path(__file__)), records=[],
                  scope='Actual serving chat formatter; full prompt IDs, not counts alone. Admission only, not a speed run.')
    for plan in plans:
        path, workload_path = Path(plan['cases']), Path(plan['workload'])
        cases, workload = [json.loads(p.read_text()) for p in (path, workload_path)]
        if digest(path) != plan['cases_sha256'] or digest(path) != workload['cases_sha256']:
            raise ValueError('Changed request fixture')
        if len(cases) != len(workload['requests']):
            raise ValueError('Changed request count')
        for index, (case, expected) in enumerate(zip(cases, workload['requests'])):
            body = {name: case[name] for name in
                    ('model', 'messages', 'tools', 'tool_choice', 'chat_template_kwargs') if name in case}
            request = Request(args.url+'/tokenize', data=json.dumps(body).encode(),
                              headers={'Content-Type': 'application/json'})
            with urlopen(request, timeout=15) as response:
                raw = response.read()
            value = json.loads(raw)
            ids = value['tokens']
            match = ids == expected['token_ids'] and value['count'] == len(ids)
            (args.output/(plan['id']+'-'+str(index)+'.response.json')).write_bytes(raw)
            report['records'].append(dict(id=plan['id'], row=index, request=body,
                                          exact_ids_match=match, actual_tokens=value['count'],
                                          expected_tokens=len(expected['token_ids']),
                                          cases_sha256=digest(path), workload_sha256=digest(workload_path),
                                          response_sha256=hashlib.sha256(raw).hexdigest()))
            (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
            if not match:
                raise ValueError('Serving chat token IDs differ: '+plan['id']+'/'+str(index))
    report['passed'] = bool(report['records'])
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(passed=report['passed'], exact_requests=len(report['records']))))


if __name__ == '__main__':
    main()
