#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit paired synthetic Mistral projection runs, never model throughput.

Campaign status.json records two serial cohorts, each with dense/indexed
strict-control/strict-reuse4096 runs. Each executable emits32 cells and dumps
all output words. FP64 coverage is sampled at real shapes; the separate native
reference test covers all words at reduced dimensions and routing boundaries.
"""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root):
    root = root.resolve()
    status = json.loads((root/'status.json').read_text())
    require(status['completed'] is True and len(status['runs']) == 8, 'incomplete campaign')
    shapes = {(m, n, k) for m in (4, 16, 32, 63, 64, 65, 127, 128)
              for n, k in ((1024, 4096), (4096, 4096), (14336, 4096), (4096, 14336))}
    datasets, word_hashes, files = {}, {}, {}
    previous_finish = 0
    fp64_samples = inactive_words = output_words = 0
    for run in status['runs']:
        key = run['cohort'], run['mode'], run['policy']
        require(key[0] in (0, 1) and key[1] in ('dense', 'indexed') and
                key[2] in ('strict-control', 'strict-reuse4096') and key not in datasets,
                'invalid/duplicate run identity')
        require(run['returncode'] == 0 and previous_finish <= run['started_ns'] < run['finished_ns'],
                'failed or overlapping run')
        previous_finish = run['finished_ns']
        binary = Path(run['command'][0])
        require(sha(binary) == run['binary_sha256'], 'benchmark binary changed')
        files[str(binary)] = run['binary_sha256']
        output = Path(run['output']).resolve()
        require(output.parent == root, 'run outside campaign')
        log = output/'run.jsonl'
        rows = [json.loads(line) for line in log.read_text().splitlines()]
        require(len(rows) == 34 and rows[-1] == {'kind': 'complete', 'owners': 0},
                'incomplete cells/native cleanup')
        meta = rows[0]
        require(meta['kind'] == 'metadata' and meta['operand_dtype'] == meta['coefficient_dtype'] == 'float16'
                and meta['accumulator_dtype'] == 'float32' and meta['reuse64_relaxed'] is False
                and meta['reuse64_min_outputs'] == (0 if key[2] == 'strict-control' else 4096)
                and meta['maximum_rows'] == 128 and meta['mode'] == key[1], 'wrong operator policy')
        cells = {}
        for cell in rows[1:-1]:
            shape = cell['logical_rows'], cell['outputs'], cell['inner']
            require(shape in shapes and shape not in cells and cell['kind'] == 'cell', 'invalid/duplicate shape')
            m, n, k = shape
            physical = 128 if key[1] == 'indexed' else m
            require(cell['mode'] == key[1] and cell['physical_rows'] == physical
                    and cell['profile_misses'] == cell['runtime_pipeline_compiles'] == 0,
                    'unprepared layout or runtime compilation')
            require(cell['reuse64_dispatches'] == int(key[2] == 'strict-reuse4096' and m >= 64 and n >= 4096),
                    'wrong cooperative route')
            require(cell['warmups'] == 3 and cell['repetitions'] == 9, 'wrong repetitions')
            for metric in ('gpu_ms', 'wall_ms'):
                require(len(cell[metric]) == 9 and all(type(x) in (int, float) and math.isfinite(x) and x > 0
                                                     for x in cell[metric]), 'invalid latency samples')
            require(type(cell['fp64_samples']) is int and cell['fp64_samples'] >= 30
                    and math.isfinite(cell['fp64_max_abs_error']) and cell['fp64_max_abs_error'] >= 0
                    and math.isfinite(cell['fp64_max_error_fraction']) and 0 <= cell['fp64_max_error_fraction'] <= 1,
                    'FP64 oracle failed')
            require(cell['inactive_zero_words'] == (physical-m)*n and cell['output_words'] == physical*n,
                    'wrong inactive/output word coverage')
            words = Path(cell['output_file']).resolve()
            require(words == output/'words'/f'{key[1]}-m{m}-n{n}-k{k}.f32'
                    and words.stat().st_size == physical*n*4, 'missing/wrong output words')
            digest = sha(words)
            word_hashes[str(words.relative_to(root))] = digest
            cell['output_sha256'] = digest
            cells[shape] = cell
            fp64_samples += cell['fp64_samples']
            inactive_words += cell['inactive_zero_words']
            output_words += cell['output_words']
        require(cells.keys() == shapes, 'missing projection shape')
        datasets[key] = cells
        files[str(log)] = sha(log)
    comparisons = []
    for cohort in (0, 1):
        for mode in ('dense', 'indexed'):
            control, candidate = datasets[cohort, mode, 'strict-control'], datasets[cohort, mode, 'strict-reuse4096']
            for shape in sorted(shapes):
                a, b = control[shape], candidate[shape]
                require(a['output_sha256'] == b['output_sha256'], 'paired output words differ')
                before, after = statistics.median(a['gpu_ms']), statistics.median(b['gpu_ms'])
                comparisons.append(dict(cohort=cohort, mode=mode, logical_rows=shape[0], outputs=shape[1],
                                        inner=shape[2], control_gpu_ms=before, candidate_gpu_ms=after,
                                        speedup=before/after, output_sha256=a['output_sha256']))
    files[str(root/'status.json')] = sha(root/'status.json')
    files[str(Path(__file__).resolve())] = sha(Path(__file__))
    return dict(passed=True, runs=8, cells=256, paired_output_files_equal=128,
                fp64_samples=fp64_samples, inactive_zero_words=inactive_words, output_words=output_words,
                comparisons=comparisons, word_sha256=word_hashes, files=files,
                scope='Synthetic FP16 operator diagnostics; FP64 samples plus native reduced-shape full oracle. '
                      'No trained-model correctness/coding-quality/TTFT/serving/comparator speed claim.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.campaign)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({key: result[key] for key in ('passed', 'runs', 'cells', 'paired_output_files_equal', 'fp64_samples')}))
