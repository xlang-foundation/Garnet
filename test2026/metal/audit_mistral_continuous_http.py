#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit actual HTTP/native refill overlap and fixed control prefixes.

Uncached dense Mistral lifecycle evidence only: no prefix-hit, fused mixed
forward, independent GPU KV snapshot, full code-quality or speed claim.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from audit_mistral_http_admission import load_client, require, validate


def validate_lifecycle(events, clients):
    by_label = {c['label']:c for c in clients}
    require(set(by_label) == {'A','B','C','D','E','A-return'}, 'missing lifecycle identity')
    complete = {e['id']:(i,e) for i,e in enumerate(events) if e['event'] == 'completed'}
    results = {label:complete[c['id']] for label,c in by_label.items()}
    initial = {results[label][1]['native_request'] for label in ('A','B','C','D')}
    require(any(e['event'] == 'native_batch' and {r['id'] for r in e['requests']} == initial
                for e in events[:results['A'][0]]), 'initial four did not run together')
    starts = {}
    for label, (_, completed) in results.items():
        starts[label] = next(i for i,e in enumerate(events) if e['event'] == 'native_batch'
                             and e['stage'] == 'prefill'
                             and any(r['id'] == completed['native_request'] for r in e['requests']))
    require(results['A'][0] < starts['E'] < results['E'][0] < starts['A-return']
            < results['A-return'][0] < min(results[label][0] for label in ('B','C','D')),
            'refills did not overlap all three live survivors')
    a, e, returned = [results[label][1] for label in ('A','E','A-return')]
    require(a['native_row'] == e['native_row'] == returned['native_row']
            and a['native_epoch'] < e['native_epoch'] < returned['native_epoch'],
            'freed row not reused with fresh epochs')
    require(a['prompt_ids'] == returned['prompt_ids'] and a['prompt_ids'] != e['prompt_ids'],
            'returned/replacement prompt identity differs')
    for label in ('B','C','D'):
        identity = results[label][1]['native_request']
        frames = [i for i,event in enumerate(events) if event['event'] == 'native_batch'
                  and event['stage'] == 'decode' and any(r['id'] == identity for r in event['requests'])]
        require(any(i < starts['E'] for i in frames)
                and any(starts['E'] < i < results['E'][0] for i in frames)
                and any(starts['A-return'] < i < results['A-return'][0] for i in frames)
                and any(i > results['A-return'][0] for i in frames), 'survivor did not decode across both refills')
    for previous, arriving in (('A','E'), ('E','A-return')):
        require(by_label[previous]['result']['completed'] < by_label[arriving]['result']['submitted'],
                'client arrival preceded trigger completion')
    return dict(reused_row=a['native_row'], epochs=[a['native_epoch'],e['native_epoch'],returned['native_epoch']],
                lifecycle={label:dict(prefill_start_event=starts[label],completed_event=i,
                                      row=event['native_row'],epoch=event['native_epoch'])
                           for label,(i,event) in results.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--control-provenance',type=Path,required=True)
    parser.add_argument('--control-provenance-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    files = {}

    def raw(path):
        data = path.read_bytes()
        files[str(path.resolve())] = hashlib.sha256(data).hexdigest()
        return data

    def read(path):
        return json.loads(raw(path))

    provenance = read(args.control_provenance)
    require(files[str(args.control_provenance.resolve())] == args.control_provenance_sha256, 'changed control provenance')
    root = args.study
    manifest,status = read(root/'manifest.json'),read(root/'status.json')
    fixture = read(Path(manifest['fixture']))
    require(provenance['files'].get(str(Path(manifest['fixture']).resolve())) == files[str(Path(manifest['fixture']).resolve())],
            'unfrozen fixture')
    cases = {c['id']:c for c in fixture['cases']}
    controls = {}
    clients = []
    expected = [('A',0,8),('B',1,256),('C',2,320),('D',3,384),('E',4,16),('A-return',0,32)]
    coding = [c for c in fixture['cases'] if c['kind'] == 'coding']
    require(len(manifest['requests']) == 6, 'incorrect lifecycle count')
    for entry,(label,index,budget) in zip(manifest['requests'],expected):
        case = coding[index]
        require(entry['label'] == label and entry['case_id'] == case['id']
                and entry['body']['max_tokens'] == budget and entry['prompt_ids'] == case['prompt_ids'],
                'changed lifecycle contract')
        control_path = Path(manifest['coding_control'])/case['id']/'request-0.result.json'
        control = read(control_path)
        require(provenance['files'].get(str(control_path.resolve())) == files[str(control_path.resolve())], 'unfrozen control')
        controls[case['id']] = control
        require(entry['expected_ids'] == control['generated_ids'][:budget], 'changed expected tokens')
        client = load_client(root/label/'request-0',case['id'],entry['body'],case['prompt_ids'],read,raw)
        client['label'] = label
        clients.append(client)
    require(status['status'] == 'collected_pending_independent_audit'
            and status['shutdown']['graceful_native_closed'] is True
            and status['shutdown']['forced_kill'] is False, 'collection/shutdown failed')
    for path,digest in manifest['input_files'].items():
        require(hashlib.sha256(raw(Path(path))).hexdigest() == digest, 'changed study input: '+path)
    for name,digest in manifest['frozen_files'].items():
        require(hashlib.sha256(raw(root/'frozen'/name)).hexdigest() == digest, 'changed frozen driver/profile')
    profile = read(root/'frozen/profile.json')
    events = [json.loads(json.loads(line)['line']) for line in raw(root/'server/worker-stdout.jsonl').splitlines()]
    exit_status = read(root/'server/worker-exit.json')
    report = validate(events,clients,controls,profile,exit_status,expected_requests=6,control_prefixes=True)
    report.update(validate_lifecycle(events,clients))
    rejected = []
    # Exercise overlap independently of the shared generic trace validator.
    # A generic B4 run can be valid while a claimed refill occurs too late.
    for label in ('no-survivor-overlap','reused-epoch','wrong-row','wrong-client-trigger'):
        es,cs = copy.deepcopy(events),copy.deepcopy(clients)
        completed = {c['label']:next(e for e in es if e['event'] == 'completed' and e['id'] == c['id']) for c in cs}
        if label == 'no-survivor-overlap':
            event = completed['B'];es.remove(event)
            index = next(i for i,e in enumerate(es) if e['event'] == 'completed' and e['id'] == completed['A']['id'])
            es.insert(index+1,event)
        elif label == 'reused-epoch': completed['A-return']['native_epoch'] = completed['A']['native_epoch']
        elif label == 'wrong-row': completed['E']['native_row'] = (completed['A']['native_row']+1)%4
        else:
            next(c for c in cs if c['label'] == 'E')['result']['submitted'] = 0
        try:
            validate_lifecycle(es,cs)
        except (ValueError,KeyError,StopIteration):
            rejected.append(label)
        else:
            raise AssertionError('Accepted lifecycle corruption: '+label)
    raw(Path(__file__))
    raw(Path(__file__).with_name('audit_mistral_http_admission.py'))
    report.update(collection_integrity_passed=True,continuous_refill_gate_passed=True,
                  exact_native_control_prefix_tokens=sum(len(c['sse_ids']) for c in clients),
                  corruptions_rejected=rejected,scope=__doc__,files=files)
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2)
        stream.write('\n')
    print(json.dumps({k:report[k] for k in ('continuous_refill_gate_passed','exact_native_control_prefix_tokens',
                                           'reused_row','epochs','close_allocation_bytes')}))


if __name__ == '__main__':
    main()
