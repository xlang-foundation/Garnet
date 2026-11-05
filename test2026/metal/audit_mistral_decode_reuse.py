#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit serial exact-FP32 projection sweeps; operator ratios are not token rates."""
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


def audit(root, draft):
    root, draft = root.resolve(), draft.resolve()
    files = {}
    def bind(path):
        with path.open('rb') as stream:files[str(path)] = hashlib.file_digest(stream,'sha256').hexdigest()
        return files[str(path)]
    collection = json.loads((root/'collector-status.json').read_text());bind(root/'collector-status.json')
    require(collection['completed'] is True and len(collection['runs']) == 6, 'incomplete serial sweep')
    previous, pids, data, words, samples = 0, set(), {}, 0, 0
    shapes = [(1,64),(7,192),(17,64),(33,128),(4096,4096),(1024,4096),(14336,4096),(4096,14336),(32768,4096)]
    expected = []
    for n,k in shapes:
        expected += [(n,k,r,False,[]) for r in (1,2,3,4)]
        expected += [(n,k,4,True,[r for r in range(4) if mask & (1<<r)]) for mask in range(16)]
    for run, identity in zip(collection['runs'],[(1,1),(1,2),(1,4),(2,4),(2,2),(2,1)]):
        cohort, columns = identity;label=f'cohort{cohort}-columns{columns}'
        require((run['cohort'],run['columns']) == identity and run['returncode'] == 0
                and type(run['pid']) is int and run['pid'] > 0 and run['pid'] not in pids
                and run['started_ns'] >= previous and run['finished_ns'] > run['started_ns'], 'failed/overlapping projection process')
        previous=run['finished_ns'];pids.add(run['pid'])
        binary=Path(run['argv'][0]);outputs=root/(label+'-outputs');raw=root/(label+'.jsonl')
        require(Path(run['argv'][1]).resolve() == outputs and run['argv'][2] == '9'
                and bind(binary) == run['binary_sha256'] and bind(raw) == run['raw_sha256'], 'changed projection executable/output/stream')
        rows=[json.loads(line) for line in raw.read_text().splitlines()]
        require(len(rows) == len(expected)+2 and rows[-1] == dict(kind='closed',cells=180,owners=0), 'incomplete projection output')
        metadata=rows[0]
        require(metadata['device'] == 'Apple M5' and metadata['warmups'] == 3 and metadata['repetitions'] == 9
                and metadata['maximum_rows'] == 4 and metadata['profile_is_upper_bound'] is True
                and metadata['coefficient_dtype'] == 'FP16', 'projection contract changed')
        cells=rows[1:-1]
        for i,(cell,(n,k,r,indexed,selected)) in enumerate(zip(cells,expected)):
            require(cell['kind'] == 'cell' and cell['index'] == i and cell['outputs'] == n and cell['inner'] == k
                    and cell['rows'] == r and cell['indexed'] == indexed and cell['selected'] == selected
                    and cell['words'] == r*n, 'wrong projection shape/mask')
            active=len(selected) if indexed else r
            require(cell['fp64_samples'] == 3*active and math.isfinite(cell['maximum_error_fraction'])
                    and 0 <= cell['maximum_error_fraction'] <= 1, 'missing/failed original FP64 bound')
            samples+=cell['fp64_samples']
            for key in ('gpu_ms','wall_ms'):
                require(len(cell[key]) == 9 and all(math.isfinite(x) and (x > 0 if active else x >= 0)
                                                  for x in cell[key]), 'invalid projection duration')
            path=outputs/f'case-{i}.bin';require(path.stat().st_size == 4*r*n, 'wrong projection output size')
            digest=bind(path);array=np.fromfile(path,dtype='<f4').reshape(r,n)
            require(np.isfinite(array).all(), 'nonfinite projection output')
            if indexed:
                for owner in set(range(r))-set(selected):require((array[owner].view('<u4') == 0).all(), 'stale inactive owner output')
            if identity != (1,1):
                require(digest == files[str(root/'cohort1-columns1-outputs'/f'case-{i}.bin')], 'complete FP32 output changed')
                words+=r*n
        data[identity]=cells
    comparisons=[]
    for cohort in (1,2):
        for columns in (2,4):
            for i,(n,k,r,indexed,selected) in enumerate(expected):
                if (len(selected) if indexed else r) == 0:continue
                control=statistics.median(data[cohort,1][i]['gpu_ms']);candidate=statistics.median(data[cohort,columns][i]['gpu_ms'])
                comparisons.append(dict(cohort=cohort,columns=columns,index=i,outputs=n,inner=k,rows=r,indexed=indexed,
                    selected=selected,control_gpu_ms=control,candidate_gpu_ms=candidate,operator_speedup=control/candidate))
    # Bind the source and linking commands that produced these three binaries.
    source=root.parent/'mistral-decode-reuse-source1'
    manifest=draft/'source1-manifest.json';bind(manifest)
    for name,digest in json.loads(manifest.read_text()).items():require(bind(source/name) == digest, 'changed frozen runtime source')
    manifest=draft/'benchmark-source2-manifest.json';bind(manifest)
    for name,digest in json.loads(manifest.read_text()).items():require(bind(Path(name)) == digest, 'changed benchmark source')
    for columns in (1,2,4):
        path=draft/f'link{columns}-v3-command.json';bind(path);command=json.loads(path.read_text())
        require(Path(command[-1]) == draft/f'benchmark{columns}-v3', 'link/collector executable differs')
        for argument in command:
            if argument.endswith('.a') or argument.endswith('.cpp'):bind(Path(argument))
    bind(Path(__file__).resolve())
    return dict(completed=True,cells_per_process=180,processes=6,compared_fp32_words=words,
        fp64_samples=samples,complete_output_bit_parity=True,comparisons=comparisons,files=files,
        scope=__doc__,whole_model_qualified=False,speed_targets_achieved=False,
        caveat='Several early small-occupancy cases in the first cohort are much slower than their second-cohort counterparts; the cause is not isolated. Retain every sample; do not select a favorable cohort as a whole-model speed claim.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection',type=Path,required=True)
    parser.add_argument('--draft',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=audit(args.collection,args.draft)
    with args.output.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps({key:report[key] for key in ('completed','compared_fp32_words','fp64_samples')}))
