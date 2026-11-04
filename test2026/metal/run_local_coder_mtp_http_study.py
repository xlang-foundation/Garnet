"""Freeze one qualified B1 MTP server and replay coding plus canonical HTTP cells.

Finite serving admission only: eight coding tasks, six canonical cells once,
an invalid-context request and a valid request after rejection. No timing claim.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from run_local_coder_baseline import engine_processes, stop_owned


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runtime', 'build', 'asset-manifest', 'auxiliary', 'qualified', 'quality-manifest', 'reference-campaign', 'experiment-root', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--port', type=int, default=8129)
    args = parser.parse_args()
    if engine_processes():
        raise RuntimeError('A competing inference engine is live')
    repo = Path(__file__).resolve().parents[2]
    source = Path(__file__).parent
    qualified = json.loads((args.qualified/'qualification-provenance.json').read_text())
    if qualified['coding']['passed'] != 8 or qualified['coding']['full_token_sequences_equal_control'] != 8:
        raise ValueError('Complete coding qualification required')
    archive = Path(qualified['source_archive'])
    if digest(archive/'manifest.json') != qualified['source_manifest_sha256']:
        raise ValueError('Qualified source archive changed')
    archive_manifest = json.loads((archive/'manifest.json').read_text())
    for name, wanted in archive_manifest['sha256'].items():
        if name.startswith(('src/', 'xModel/')) and Path(name).suffix in ('.py', '.cpp', '.h', '.mm', '.json') and digest(repo/name) != wanted:
            raise ValueError('Qualified inference source changed: '+name)
    library_sha = qualified['builds']['native-mtp-profiles-build']['libgarnet.dylib']
    if digest(args.build/'libgarnet.dylib') != library_sha:
        raise ValueError('Qualified native library changed')
    if digest(args.auxiliary) != 'dec6630788d1ba4557230ca02b5df2db239f2b2bbcfff1c6742776e231e7540b':
        raise ValueError('Auxiliary checkpoint changed')
    if digest(args.asset_manifest) != 'db7aecc1a6bd8911e5737c0dc3f13089f048638669177f8c8dc331df0a6ad198':
        raise ValueError('Pinned target asset manifest changed')
    assets = json.loads(args.asset_manifest.read_text())
    checkpoint = Path(assets['root'])
    for entry in assets['files']:
        path = checkpoint/entry['name']
        if path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
            raise ValueError('Pinned target checkpoint changed: '+entry['name'])
    reference = args.reference_campaign
    benchmark = json.loads((reference/'sources/benchmark-manifest.json').read_text())
    if {p['id'] for p in benchmark['plans']} != {f'p{p}-c{c}-o128' for p in (256, 2048) for c in (1, 2, 4)}:
        raise ValueError('All six canonical reference cells required')
    reference_ids = {}
    reference_log = reference/'server/cohort-1/worker-stdout.jsonl'
    for line in reference_log.read_text().splitlines():
        try:
            event = json.loads(json.loads(line)['line'])
        except (ValueError, KeyError):
            continue
        if isinstance(event, dict) and event.get('event') == 'completed':
            key = tuple(event['prompt_ids'])
            if key in reference_ids and reference_ids[key] != event['generated_ids']:
                raise ValueError('Canonical reference output is not deterministic')
            reference_ids[key] = event['generated_ids']
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    frozen = out/'sources'
    frozen.mkdir()
    for path in [Path(__file__), source/'local_coder_mtp_worker.py', source/'local_coder_batch_http.py',
                 source/'local_coder_batch_transport.py', source/'run_local_coder_baseline.py',
                 source/'run_local_coder_quality.py', source/'local_coder_stream_benchmark.py',
                 repo/'test2026/cuda/qwen38_5090/native_http_bridge.py']:
        shutil.copy2(path, frozen/path.name)
    shutil.copy2(args.build/'libgarnet.dylib', frozen/'libgarnet.dylib')
    shutil.copy2(args.build/'CMakeCache.txt', frozen/'CMakeCache.txt')
    shutil.copy2(args.quality_manifest, frozen/'quality-manifest.json')
    profile = dict(profile_id='native27b-mtp-b1-c4096-prefill128-draft1-verify4', batch=1, capacity=4096,
        target_prefill_chunk=128, auxiliary_prefill_chunk=128, auxiliary_draft_chunk=1, verification_chunk=4,
        max_input_tokens=3072, max_output_tokens=1024, prefix_cache=False, multimodal=False,
        native_continuous_batching=False, auxiliary_checkpoint=str(args.auxiliary.resolve()),
        memory_inventory_path=str(out/'memory-inventory.json'))
    write(frozen/'profile.json', profile)
    plans = []
    for plan in benchmark['plans']:
        folder = frozen/plan['id']
        folder.mkdir()
        cases, workload = args.experiment_root/plan['cases'], args.experiment_root/plan['workload']
        data = json.loads(workload.read_text())
        if digest(cases) != plan['cases_sha256'] or data['cases_sha256'] != plan['cases_sha256']:
            raise ValueError('Canonical request bytes changed')
        shutil.copy2(cases, folder/'cases.json')
        shutil.copy2(workload, folder/'workload.json')
        plans.append(dict(id=plan['id'], cases=str(folder/'cases.json'), workload=str(folder/'workload.json'),
                          expected_ids=[reference_ids[tuple(row['token_ids'])] for row in data['requests']]))
    write(frozen/'canonical.json', plans)
    write(out/'artifacts.json', dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        files={str(p.relative_to(frozen)):digest(p) for p in frozen.rglob('*') if p.is_file()},
        qualified_source_manifest_sha256=qualified['source_manifest_sha256'],
        qualified_library_sha256=library_sha, asset_manifest_sha256=digest(args.asset_manifest),
        auxiliary_sha256=digest(args.auxiliary), reference_worker_log_sha256=digest(reference_log),
        scope=__doc__))
    report = dict(status='starting', issues=[])
    save = lambda: write(out/'collection.json', report)
    save()
    url = 'http://127.0.0.1:'+str(args.port)
    command = [sys.executable, str(frozen/'local_coder_batch_http.py'), '--checkpoint', str(checkpoint),
        '--log-root', str(out/'server'), '--port', str(args.port), '--first-token-timeout', '3600', '--',
        str(args.runtime.resolve()), '-B', str(frozen/'local_coder_mtp_worker.py'), str(frozen), str(repo),
        str(checkpoint), str(frozen/'profile.json')]
    write(out/'command.json', command)
    environment = dict(os.environ, PYTHONPATH=str(frozen)+os.pathsep+str(source)+os.pathsep+str(repo/'test2026/cuda/qwen38_5090'))
    environment['GARNET_METAL_PROFILE_OPS'] = '0'
    with (out/'server.log').open('xb') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment, start_new_session=True)
        try:
            deadline = time.monotonic()+600
            while True:
                if process.poll() is not None:
                    raise RuntimeError('Native MTP HTTP process exited before ready')
                try:
                    with urllib.request.urlopen(url+'/health', timeout=2) as response:
                        report['health'] = json.load(response)
                    break
                except (urllib.error.URLError, TimeoutError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Native MTP HTTP startup deadline')
                    time.sleep(.5)
            report['status'] = 'collecting_coding'
            save()
            with (out/'quality-client.log').open('x') as client:
                subprocess.run([sys.executable, str(frozen/'run_local_coder_quality.py'), '--manifest', str(frozen/'quality-manifest.json'),
                    '--url', url, '--engine-label', 'native27b-mtp-single-active', '--output', str(out/'responses'),
                    '--parallel-tasks', '8', '--request-timeout', '3600'], stdout=client, stderr=subprocess.STDOUT, check=True)
            report['canonical'] = []
            for plan in plans:
                report.update(status='collecting_canonical', current_cell=plan['id'])
                save()
                with (out/(plan['id']+'.client.log')).open('x') as client:
                    subprocess.run([sys.executable, str(frozen/'local_coder_stream_benchmark.py'), '--url', url,
                        '--cases', plan['cases'], '--workload', plan['workload'], '--metadata', str(frozen/'profile.json'),
                        '--timeout', '3600', '--output', str(out/plan['id'])], stdout=client, stderr=subprocess.STDOUT, check=True)
                report['canonical'].append(plan['id'])
                save()
            # Reject a valid HTTP body that exceeds the native context envelope.
            invalid = json.loads(Path(plans[-1]['cases']).read_text())[0]
            invalid['messages'] = [dict(role='user', content='x '*5000)]
            request = urllib.request.Request(url+'/v1/chat/completions', data=json.dumps(invalid).encode(), headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request, timeout=30) as response:
                rejection = response.read().decode()
            (out/'rejected-profile.sse').write_text(rejection)
            if 'exceeds input/output/context profile bounds' not in rejection:
                raise ValueError('Missing native profile rejection')
            plan = plans[0]
            with (out/'after-rejection.client.log').open('x') as client:
                subprocess.run([sys.executable, str(frozen/'local_coder_stream_benchmark.py'), '--url', url,
                    '--cases', plan['cases'], '--workload', plan['workload'], '--metadata', str(frozen/'profile.json'),
                    '--timeout', '3600', '--output', str(out/'after-rejection')], stdout=client, stderr=subprocess.STDOUT, check=True)
            report['status'] = 'collected_pending_evaluation_and_audit'
        except Exception as error:
            report.update(status='failed', error=repr(error))
            raise
        finally:
            report['shutdown'] = stop_owned(process, graceful_native=True)
            save()
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
