#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Two serial native operator cohorts; fixed model/inputs, reversed order.

Not full-model speed, quality, serving or observed ANE execution evidence.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'metal'))
from run_local_coder_baseline import engine_processes


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--export', type=Path, required=True)
    p.add_argument('--comparison-export', type=Path,
                   help='Optional second representation with identical weights/input; reuse Metal control')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not engine_processes(), 'An inference process is already live'
    a.output.mkdir(exist_ok=False)
    manifest = json.loads((a.export/'manifest.json').read_text())
    assert sha(a.checkpoint) == manifest['checkpoint_sha256']
    exports = [a.export]+([a.comparison_export] if a.comparison_export else [])
    manifests = [json.loads((root/'manifest.json').read_text()) for root in exports]
    for root, model in zip(exports, manifests):
        assert model['input_sha256'] == manifest['input_sha256'] and model['rows'] == manifest['rows']
        assert [w['native_rounded_sha256'] for w in model['weights']] == [w['native_rounded_sha256'] for w in manifest['weights']]
        for name, expected in model['files'].items():
            assert sha(root/name) == expected, 'Changed export: '+name
    status = dict(completed=False, scope=__doc__, probe=str(a.probe), probe_sha256=sha(a.probe),
                  model_manifest_sha256=sha(a.export/'manifest.json'),
                  exports=[dict(root=str(root),manifest_sha256=sha(root/'manifest.json'),
                                representation=model['representation']) for root,model in zip(exports,manifests)], runs=[])
    def save():
        (a.output/'status.json').write_text(json.dumps(status, indent=2)+'\n')
    save()
    for cohort in (0, 1):
        cases = [('metal', a.export, 'metal'), ('coreml_cpu', a.export, 'coreml_cpu'),
                 ('coreml_cpu_ane', a.export, 'coreml_cpu_ane')]
        if a.comparison_export:
            cases += [('coreml_cpu', a.comparison_export, 'comparison-coreml_cpu'),
                      ('coreml_cpu_ane', a.comparison_export, 'comparison-coreml_cpu_ane')]
        for engine, root, label in cases if cohort == 0 else cases[::-1]:
            assert not engine_processes(), 'Previous/competing inference process remains live'
            folder = a.output/f'cohort{cohort}-{label}'
            cmd = [str(a.probe), engine, str(a.checkpoint), str(root), str(folder), '0']
            record = dict(cohort=cohort, engine=engine, export_root=str(root), label=label,
                          argv=cmd, started_ns=time.time_ns())
            env = {k: v for k, v in os.environ.items() if not k.startswith('DYLD_')}
            env['GARNET_METAL_PROFILE_OPS'] = '0'
            with (a.output/f'cohort{cohort}-{label}.log').open('w') as log:
                process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
                record['pid'] = process.pid
                code = process.wait()
            record.update(returncode=code, finished_ns=time.time_ns(), output=str(folder))
            status['runs'].append(record)
            save()
            print('terminal', cohort, label, code, flush=True)
            if code:
                raise RuntimeError('Failed native probe; raw log/output retained: '+str(folder))
    assert not engine_processes(), 'An inference process remains live after collection'
    for root, model in zip(exports, manifests):
        for name, expected in model['files'].items():
            assert sha(root/name) == expected, 'Export changed during execution: '+name
    status['completed'] = True
    save()


if __name__ == '__main__':
    main()
