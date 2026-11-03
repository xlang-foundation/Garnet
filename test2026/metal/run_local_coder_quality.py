"""Replay frozen coding tasks against one already-loaded engine.

Preserves every original request and response. This collects bounded regression
evidence, not a performance campaign. Evaluation is a separate sandboxed step.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request-timeout',type=float,default=600)
    parser.add_argument('--parallel-tasks',type=int,default=1,help='Bounded concurrent task clients; default serial replay')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--url', required=True)
    parser.add_argument('--engine-label', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 1<=args.request_timeout<=7200:parser.error('Request timeout must be1..7200seconds')
    if not 1<=args.parallel_tasks<=8:parser.error('Parallel task limit must be1..8')
    manifest = json.loads(args.manifest.read_text())
    benchmark = Path(__file__).with_name('local_coder_stream_benchmark.py')
    # Validate the entire plan before any model request or output mutation.
    plans, seen = [], set()
    for plan in manifest['plans']:
        identity = plan['id']
        if not identity or identity in seen or Path(identity).name != identity or identity in ('.', '..'):
            raise ValueError('Invalid or duplicate task identity')
        seen.add(identity)
        cases, workload = Path(plan['cases']), Path(plan['workload'])
        bodies = json.loads(cases.read_text())
        tokens = json.loads(workload.read_text())
        digest = hashlib.sha256(cases.read_bytes()).hexdigest()
        if len(bodies) != 1 or len(tokens['requests']) != 1 or digest != plan['cases_sha256'] or digest != tokens['cases_sha256']:
            raise ValueError('Quality tasks must be exact frozen single requests')
        plans.append((identity, cases, workload))
    if not plans:
        raise ValueError('Empty quality plan')
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = args.output/'frozen'
    frozen.mkdir()
    for source in (Path(__file__), benchmark):
        shutil.copy2(source, frozen/source.name)
    shutil.copy2(args.manifest, frozen/'manifest.json')
    status = dict(status='running', engine_label=args.engine_label, url=args.url,
                  manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  benchmark_sha256=hashlib.sha256(benchmark.read_bytes()).hexdigest(),
                  scope=manifest['scope'], tasks=[],parallel_tasks=args.parallel_tasks,request_timeout_seconds=args.request_timeout)
    def save():
        temporary = args.output/'status.tmp'
        temporary.write_text(json.dumps(status, indent=2)+'\n')
        temporary.replace(args.output/'status.json')
    save()
    if args.parallel_tasks>1:
        def collect(plan):
            identity,cases,workload=plan;task=frozen/identity;task.mkdir()
            shutil.copy2(cases,task/'cases.json');shutil.copy2(workload,task/'workload.json')
            with (args.output/(identity+'.client.log')).open('x') as log:
                result=subprocess.run([sys.executable,str(frozen/benchmark.name),'--url',args.url,
                    '--cases',str(task/'cases.json'),'--workload',str(task/'workload.json'),
                    '--metadata',str(frozen/'manifest.json'),'--timeout',str(args.request_timeout),'--output',str(args.output/identity)],stdout=log,stderr=subprocess.STDOUT)
            return dict(id=identity,client_exit_code=result.returncode)
        status['completion_order']=[]
        with concurrent.futures.ThreadPoolExecutor(args.parallel_tasks) as executor:
            pending=[executor.submit(collect,plan) for plan in plans]
            for future in concurrent.futures.as_completed(pending):
                row=future.result();status['tasks'].append(row);status['completion_order'].append(row['id']);save()
        rows={row['id']:row for row in status['tasks']};status['tasks']=[rows[plan[0]] for plan in plans]
        if any(row['client_exit_code'] for row in status['tasks']):
            status['status']='stopped_after_generation_failure';save();return 1
    else:
        for identity, cases, workload in plans:
            task = frozen/identity
            task.mkdir()
            shutil.copy2(cases, task/'cases.json')
            shutil.copy2(workload, task/'workload.json')
            status['current_task'] = identity
            save()
            with (args.output/(identity+'.client.log')).open('x') as log:
                completed = subprocess.run([sys.executable, str(frozen/benchmark.name),
                    '--url', args.url, '--cases', str(task/'cases.json'),
                    '--workload', str(task/'workload.json'), '--metadata', str(frozen/'manifest.json'), '--timeout',str(args.request_timeout),
                    '--output', str(args.output/identity)], stdout=log, stderr=subprocess.STDOUT)
            status['tasks'].append(dict(id=identity, client_exit_code=completed.returncode))
            save()
            if completed.returncode:
                status['status'] = 'stopped_after_generation_failure'
                save()
                return 1
    status['status'] = 'collected_pending_sandbox_evaluation'
    save()
    return 0


if __name__ == '__main__':
    sys.exit(main())
