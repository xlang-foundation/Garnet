#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Collect two serial native attention operator cohorts; never launch reference engines."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from run_local_coder_baseline import engine_processes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    binary, root = args.binary.resolve(), args.output.resolve()
    assert not engine_processes(), 'Another inference engine is live'
    root.mkdir(exist_ok=False)
    status = dict(completed=False, binary=str(binary), binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                  runs=[], scope=__doc__)
    def save():
        (root/'status.json').write_text(json.dumps(status, indent=2)+'\n')
    save()
    for cohort in (0, 1):
        assert not engine_processes(), 'Previous/competing inference process is live'
        command = [str(binary), '--benchmark', str(root/f'cohort{cohort}')]
        record = dict(cohort=cohort, argv=command, started_ns=time.time_ns())
        env = {k: v for k, v in os.environ.items() if not k.startswith('DYLD_')}
        env['GARNET_METAL_PROFILE_OPS'] = '0'
        with (root/f'cohort{cohort}.log').open('w') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            record['pid'] = child.pid
            code = child.wait()
        record.update(returncode=code, finished_ns=time.time_ns())
        status['runs'].append(record)
        save()
        print('terminal', cohort, code, flush=True)
        if code:
            raise RuntimeError('Native attention probe failed; partial evidence retained')
    assert not engine_processes(), 'An inference process remains live after collection'
    assert hashlib.sha256(binary.read_bytes()).hexdigest() == status['binary_sha256'], 'Native binary changed'
    status['completed'] = True
    save()


if __name__ == '__main__':
    main()
