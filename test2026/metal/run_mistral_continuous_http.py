#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Actual-weight A/B/C/D -> E -> A-return HTTP refill study; no speed claim.

Uses a pinned, already-qualified uncached B4 engine. A/E/returned-A budgets
are8/16/32; B/C/D budgets256/320/384 keep survivors alive across both refills.
The independent auditor must prove actual overlap, row reuse and token parity.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from urllib.error import URLError
from urllib.request import urlopen

from run_local_coder_baseline import engine_processes, stop_owned
from local_coder_stream_benchmark import run_request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('engine-spec', 'fixture', 'coding-control', 'serving-manifest', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if engine_processes():
        raise RuntimeError('Inference engine already live: '+str(engine_processes()))
    spec = json.loads(args.engine_spec.read_text())
    profile_path = Path(spec['argv'][-1])
    profile = json.loads(profile_path.read_text())
    if profile['batch'] != 4 or profile['capacity'] != 4096 or profile['prefix_cache'] is not False:
        raise ValueError('Study requires qualified uncached B4/context4096 profile')
    if hashlib.sha256(profile_path.read_bytes()).hexdigest() != spec['profile_sha256']:
        raise ValueError('Changed engine profile')
    library = Path(spec['argv'][-4])/'libgarnet.dylib'
    if hashlib.sha256(library.read_bytes()).hexdigest() != spec['native_library_sha256']:
        raise ValueError('Changed native library')
    fixture = json.loads(args.fixture.read_text())
    cases = [c for c in fixture['cases'] if c['kind'] == 'coding']
    plans = {p['id']: p for p in json.loads(args.serving_manifest.read_text())['plans']}
    selected = [(label, cases[i], budget) for label, i, budget in
                [('A',0,8), ('B',1,256), ('C',2,320), ('D',3,384), ('E',4,16), ('A-return',0,32)]]
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = args.output/'frozen'
    frozen.mkdir()
    for p in (Path(__file__), Path(__file__).with_name('run_local_coder_baseline.py'),
              Path(__file__).with_name('local_coder_stream_benchmark.py')):
        shutil.copyfile(p, frozen/p.name)
    shutil.copyfile(profile_path, frozen/'profile.json')
    requests = []
    for label, case, budget in selected:
        source = Path(plans[case['id']]['cases'])
        if hashlib.sha256(source.read_bytes()).hexdigest() != plans[case['id']]['cases_sha256']:
            raise ValueError('Changed frozen request body')
        body = dict(json.loads(source.read_text())[0], max_tokens=budget)
        control_path = args.coding_control/case['id']/'request-0.result.json'
        control = json.loads(control_path.read_text())
        if control['prompt_ids'] != case['prompt_ids'] or len(control['generated_ids']) < budget:
            raise ValueError('Control does not cover lifecycle request')
        requests.append(dict(label=label, case_id=case['id'], body=body, prompt_ids=case['prompt_ids'],
                             expected_ids=control['generated_ids'][:budget], source_request=str(source),
                             source_control=str(control_path)))
        (args.output/label).mkdir()
    command = list(spec['argv'])
    command[command.index('--log-root')+1] = str(args.output/'server')
    command[-1] = str(frozen/'profile.json')
    manifest = dict(scope=__doc__, engine=spec, requests=requests, command=command,
                    fixture=str(args.fixture), coding_control=str(args.coding_control),
                    input_files={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (args.engine_spec,args.fixture,args.serving_manifest,profile_path,library)},
                    frozen_files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen.iterdir()})
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report = dict(status='starting', transitions=[], responses={}, scope=__doc__)

    def save():
        (args.output/'status.json').write_text(json.dumps(report,indent=2)+'\n')

    def call(entry, barrier):
        return run_request(spec['url'], entry['body'], args.output/entry['label']/'request-0',
                           barrier, 600, len(entry['prompt_ids']), False)

    save()
    with (args.output/'server.log').open('x') as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                  env=dict(os.environ, **spec.get('env',{})), start_new_session=True)
        report['server_pid'] = server.pid
        save()
        try:
            while True:
                if server.poll() is not None or (args.output/'server/worker-exit.json').exists():
                    raise RuntimeError('Owned bridge/native worker exited before readiness')
                try:
                    with urlopen(spec['url']+'/health',timeout=2) as response:
                        if response.status == 200:
                            report['health'] = json.load(response)
                            break
                except (URLError, TimeoutError):
                    pass
                time.sleep(.5)
            report['status'] = 'running_initial_four'
            save()
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
                barrier = threading.Barrier(4)
                jobs = {r['label']:pool.submit(call,r,barrier) for r in requests[:4]}
                for finished, arriving in [('A',requests[4]), ('E',requests[5])]:
                    result = jobs[finished].result()
                    if result['status'] != 'completed':
                        raise RuntimeError('Lifecycle trigger failed: '+finished)
                    report['responses'][finished] = result
                    report['transitions'].append(dict(completed=finished, submitted=arriving['label'],
                                                       monotonic=time.monotonic()))
                    report['status'] = 'running_'+arriving['label']
                    save()
                    jobs[arriving['label']] = pool.submit(call,arriving,threading.Barrier(1))
                for label, future in jobs.items():
                    report['responses'][label] = future.result()
                    if report['responses'][label]['status'] != 'completed':
                        raise RuntimeError('Lifecycle request failed: '+label)
            report['status'] = 'collected_pending_independent_audit'
        except BaseException as error:
            report.update(status='failed', error=repr(error))
            raise
        finally:
            report['shutdown'] = stop_owned(server, graceful_native=True)
            save()
    print(json.dumps(dict(status=report['status'], requests=len(report['responses']), shutdown=report['shutdown'])))


if __name__ == '__main__':
    main()
