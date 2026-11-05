#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit full layout words and serial paired operator timings; not model speed."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import numpy as np


def require(value, message):
    if not value:
        raise ValueError(message)


def fixtures():
    cells = []
    for scenario in range(5):
        cells += [(4,128*4096,512,4096,scenario,False),
                  (512,4096,4,0,scenario,False), (4,128*1024,512,1024,scenario,False)]
    return cells+[(512,4100,512,4100,3,False), (512,3,512,3,3,False),
                  (4,128*4096,512,4096,3,True), (4,5*12,20,12,3,True)]


def oracle(shape):
    rows, columns, _, width, scenario, _ = shape
    patterns = np.array([0,0x80000000,1,0x80000001,0x7f800000,0xff800000,
                         0x7fc12345,0xffc54321,0x3f800000,0xbf800000], dtype='<u4')
    words = patterns[(np.arange(rows*columns, dtype=np.int64)*7+3)%10].reshape(rows, columns)
    if width:
        for row in range(rows):
            for token in range(columns//width):
                live = (True if scenario == 0 else row == 0 if scenario == 1 else
                        row+1 == rows if scenario == 2 else (row+token)%3 != 1 if scenario == 3 else False)
                if not live:
                    words[row,token*width:(token+1)*width] = 0
    return hashlib.sha256(words.tobytes()).hexdigest()


def audit(root, draft):
    root, draft = root.resolve(), draft.resolve()
    files = {}
    def sha(path):
        with path.open('rb') as stream:
            files[str(path)] = hashlib.file_digest(stream, 'sha256').hexdigest()
        return files[str(path)]
    def read(path):
        sha(path)
        return json.loads(path.read_text())
    status = read(root/'collector-status.json')
    require(status['completed'] is True and len(status['runs']) == 4, 'incomplete operator collection')
    shapes = fixtures()
    hashes = [oracle(shape) for shape in shapes]
    previous, pids, total, results = 0, set(), 0, {}
    for run, identity in zip(status['runs'], [(0,'control'),(0,'candidate'),(1,'candidate'),(1,'control')]):
        cohort, variant = identity
        label = f'cohort{cohort}-{variant}'
        require((run['cohort'],run['variant']) == identity and run['returncode'] == 0
                and type(run['pid']) is int and run['pid'] > 0 and run['pid'] not in pids
                and previous <= run['started_ns'] < run['finished_ns'], 'failed/overlapping native process')
        previous = run['finished_ns'];pids.add(run['pid'])
        require(sha(Path(run['argv'][0])) == run['binary_sha256']
                and run['argv'][1:] == ['--benchmark', str(root/label)], 'changed binary or benchmark command')
        path = root/(label+'.jsonl');sha(path)
        entries = [json.loads(line) for line in path.read_text().splitlines()]
        require(len(entries) == len(shapes)+2 and entries[0] == dict(kind='metadata',device='Apple M5',vector_compiled=variant=='candidate')
                and entries[-1] == dict(kind='closed',cases=len(shapes),owners=0), 'missing native metadata/cleanup/cases')
        for index, (shape, cell) in enumerate(zip(shapes, entries[1:-1])):
            rows, columns, out_rows, width, scenario, gpu_flags = shape
            out_columns, flag_columns = rows*columns//out_rows, columns//width if width else 1
            require(cell['kind'] == 'cell' and cell['index'] == index
                    and (cell['rows'],cell['columns'],cell['output_rows'],cell['token_width'],cell['scenario'],cell['gpu_flags']) == shape
                    and cell['output_columns'] == out_columns and cell['warmups'] == 3
                    and cell['cpu_reference_words'] == rows*columns, 'wrong layout shape/oracle coverage')
            expected_dispatches = (int(columns%4 == out_columns%4 == width%4 == 0)+int(gpu_flags and flag_columns%4 == 0)) if variant == 'candidate' else 0
            require(cell['vector_dispatches'] == expected_dispatches, 'wrong vector/scalar route')
            for field in ('gpu_ms','wall_ms'):
                require(len(cell[field]) == 9 and all(math.isfinite(t) and t > 0 for t in cell[field]), 'invalid native timing')
            path = root/label/f'case-{index}.f32'
            require(path.stat().st_size == rows*columns*4 and sha(path) == hashes[index], 'full layout bits differ from independent oracle')
            total += rows*columns
        results[identity] = entries[1:-1]
    comparisons = []
    for cohort in (0,1):
        for index, shape in enumerate(shapes):
            left = statistics.median(results[cohort,'control'][index]['gpu_ms'])
            right = statistics.median(results[cohort,'candidate'][index]['gpu_ms'])
            comparisons.append(dict(cohort=cohort,index=index,shape=shape,control_gpu_ms=left,
                                    candidate_gpu_ms=right,operator_speedup=left/right))
    source = root.parent/'mistral-layout-vector-source1'
    for name, digest in read(draft/'source1-manifest.json').items():
        require(sha(source/name) == digest, 'compiled source changed')
    sha(Path(__file__).resolve())
    return dict(passed=True,cases=4*len(shapes),full_cpu_oracle_words=total,
                candidate_control_compared_words=total//2,comparisons=comparisons,files=files,
                scope=__doc__,model_qualified=False,speed_targets_achieved=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('collection','draft','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.collection,args.draft)
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps({key:report[key] for key in ('passed','cases','full_cpu_oracle_words')}))
