#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Collect fixed comparator campaigns sequentially, with source/package guards."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(16*1024*1024), b''):
            sha.update(chunk)
    return sha.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def preflight(spec):
    repo = Path(spec['source_repository'])
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    overlay = subprocess.check_output(['git', 'diff', '--binary', 'HEAD'], cwd=repo)
    code = 'import importlib.metadata as m,json;print(json.dumps(sorted([(d.metadata.get("Name"),d.version) for d in m.distributions()])))'
    packages = subprocess.check_output([spec['python_executable'], '-B', '-c', code])
    observed = dict(source_commit=commit, source_overlay_sha256=hashlib.sha256(overlay).hexdigest(),
                    packages_sha256=hashlib.sha256(packages).hexdigest())
    for name, value in observed.items():
        if value != spec[name]:
            raise ValueError('Comparator changed before/after collection: '+name)
    for artifact in spec.get('native_artifacts', []):
        path = Path(artifact['path'])
        if path.stat().st_size != artifact['bytes'] or digest(path) != artifact['sha256']:
            raise ValueError('Comparator native binary changed: '+str(path))
    admission = json.loads(Path(spec['admission_audit']).read_text())
    proof = json.loads(Path(spec['prompt_id_audit']).read_text())
    quality_path = Path(spec['quality_report'])
    if admission['issues'] or not proof['passed']:
        raise ValueError('Comparator admission/prompt-ID audit failed')
    for name in ('admission_audit', 'prompt_id_audit', 'quality_report'):
        if digest(Path(spec[name])) != spec[name+'_sha256']:
            raise ValueError('Comparator evidence changed: '+name)
    for artifact in spec['checkpoint_files']:
        path = Path(artifact['path'])
        if path.stat().st_size != artifact['bytes'] or digest(path) != artifact['sha256']:
            raise ValueError('Comparator checkpoint changed: '+str(path))
    observed['quality'] = json.loads(quality_path.read_text())['passed']
    return observed


def main():
    def terminate(signum, frame):
        raise KeyboardInterrupt('Reference sequence received SIGTERM')
    signal.signal(signal.SIGTERM, terminate)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--engine-specs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    cohorts = manifest['required_fresh_process_cohorts']
    if type(cohorts) is not int or cohorts < 1:
        raise ValueError('Positive fresh-process cohort count required')
    specs = [json.loads(path.read_text()) for path in args.engine_specs]
    if len({spec['engine'] for spec in specs}) != len(specs):
        raise ValueError('Comparator names must be unique')
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = args.output/'sources'
    frozen.mkdir()
    for name in ('run_local_coder_baseline.py', 'local_coder_stream_benchmark.py',
                 'sample_local_coder_memory.py', 'summarize_local_coder_baseline.py', Path(__file__).name):
        shutil.copy2(Path(__file__).with_name(name), frozen/name)
    shutil.copy2(args.manifest, frozen/'manifest.json')
    for index, path in enumerate(args.engine_specs):
        shutil.copy2(path, frozen/('engine-'+str(index)+'.json'))
    status = dict(status='preflight', engine_order=[spec['engine'] for spec in specs],
                  manifest_sha256=digest(args.manifest), engines=[],
                  scope='Sequential repeated performance observations; coding admission is separate and its failures remain visible')
    write(args.output/'status.json', status)
    child = None
    try:
        observations = [preflight(spec) for spec in specs]
        write(args.output/'preflight.json', observations)
        if args.preflight_only:
            status['status'] = 'preflight_passed_no_inference'
            return
        for index, spec in enumerate(specs):
            preflight(spec)
            destination = (args.output/spec['engine']).resolve()
            status.update(status='collecting', current_engine=spec['engine'])
            write(args.output/'status.json', status)
            argv = [sys.executable, '-B', str((frozen/'run_local_coder_baseline.py').resolve()),
                    '--experiment-root', str(args.experiment_root.resolve()),
                    '--manifest', str((frozen/'manifest.json').resolve()),
                    '--engine-spec', str((frozen/('engine-'+str(index)+'.json')).resolve()),
                    '--cohorts', str(cohorts), '--output', str(destination)]
            with (args.output/(spec['engine']+'.collector.log')).open('x') as log:
                child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
                status['collector_pid'] = child.pid
                write(args.output/'status.json', status)
                result = child.wait()
                child = None
                status['collector_pid'] = None
            if result:
                raise RuntimeError('Comparator collector failed: '+spec['engine'])
            audit_path = args.output/(spec['engine']+'.audit.json')
            subprocess.run([sys.executable, '-B', str(frozen/'summarize_local_coder_baseline.py'),
                            '--campaign', str(destination), '--output', str(audit_path)], check=True)
            audit = json.loads(audit_path.read_text())
            if audit['issues']:
                raise ValueError('Comparator accounting audit failed: '+spec['engine'])
            observed = preflight(spec)
            status['engines'].append(dict(engine=spec['engine'], completed_cohorts=cohorts,
                                          audit_sha256=digest(audit_path), postflight=observed))
            write(args.output/'status.json', status)
        status['status'] = 'completed_collection_not_automatic_qualification'
    except BaseException as error:
        status.update(status='failed', error=repr(error))
        raise
    finally:
        if child is not None and child.poll() is None:
            # Let the collector's finally block close its owned engine group.
            child.send_signal(signal.SIGINT)
            child.wait(timeout=180)
        write(args.output/'status.json', status)


if __name__ == '__main__':
    main()
