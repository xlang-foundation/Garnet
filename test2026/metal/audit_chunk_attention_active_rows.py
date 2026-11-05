#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit two serial synthetic request-schedule cohorts, without a model-speed claim."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def audit(root):
    root = root.resolve()
    status = json.loads((root/'status.json').read_text())
    require(status['completed'] is True and len(status['runs']) == 2, 'incomplete serial cohorts')
    binary = Path(status['binary'])
    require(sha(binary) == status['binary_sha256'], 'changed native executable')
    files = {str(binary): sha(binary), str(root/'status.json'): sha(root/'status.json')}
    expected = {(mask, prefix) for mask in (1, 8, 5, 3, 15) for prefix in (0, 3968)}
    previous, pids, words_by_cell, comparisons, ragged_policy = 0, set(), {}, [], None
    for cohort, run in enumerate(status['runs']):
        out = root/f'cohort{cohort}'
        require(run['cohort'] == cohort and run['returncode'] == 0
                and run['argv'] == [str(binary), '--benchmark', str(out)], 'wrong/failed command')
        require(run['started_ns'] >= previous and run['finished_ns'] > run['started_ns']
                and run['pid'] not in pids, 'overlap or reused process identity')
        previous = run['finished_ns']
        pids.add(run['pid'])
        report = json.loads((out/'report.json').read_text())
        require([report[k] for k in ('physical_batch', 'chunk', 'heads', 'kv_heads', 'dimension', 'capacity')]
                == [4, 128, 32, 8, 128, 4096] and report['native_owners_after_close'] == 0,
                'wrong physical profile or retained owners')
        require(len(report['cells']) == len(expected), 'missing/extra cell')
        seen = set()
        for cell in report['cells']:
            key = cell['mask'], cell['prefix']
            require(key in expected and key not in seen and cell['unknown_controls'] is False
                    and cell['gated'] is False, 'wrong/duplicate control policy')
            seen.add(key)
            # Source1 is the retained gapped/overwrite study; source2 explicitly
            # records its contiguous production-prefix policy.
            ragged = cell.get('ragged_controls', True)
            require(type(ragged) is bool, 'invalid ragged policy')
            if ragged_policy is None:
                ragged_policy = ragged
            require(ragged == ragged_policy, 'changed controls between cells/cohorts')
            owners = cell['mask'].bit_count()
            count = 4*128*32*128
            require(cell['warmups'] == 3 and cell['repetitions'] == 9 and cell['output_words'] == count
                    and cell['bitwise_equal'] is True and cell['fp64_samples'] == owners*12
                    and np.isfinite(cell['fp64_maximum_error_fraction'])
                    and 0 <= cell['fp64_maximum_error_fraction'] <= 1, 'missing independent oracle/counts')
            require(cell['full_matrices'] == 128 and cell['selected_matrices'] == owners*32
                    and cell['skipped_matrices'] == 128-owners*32, 'logical schedule counters differ')
            inactive = count-owners*(128-int(ragged))*4096
            require(cell['inactive_zero_words'] == inactive, 'inactive zero coverage differs')
            label = f'mask{key[0]}-p{key[1]}-u0'
            arrays = []
            for variant in ('full', 'selected'):
                path = out/(label+f'-{variant}.f32')
                values = np.fromfile(path, dtype='<f4')
                require(values.size == count and np.isfinite(values).all(), 'wrong/nonfinite raw output')
                values = values.reshape(4, 128, 4096)
                flags = np.array([[bool(key[0] & (1 << b)) and (not ragged or t != 3)
                                   for t in range(128)] for b in range(4)])
                require(np.count_nonzero(values[~flags]) == 0, 'inactive raw output is nonzero')
                bits = values.view(np.uint32)
                if cohort:
                    require(np.array_equal(bits, words_by_cell[key, variant]), 'raw output changed between fresh cohorts')
                else:
                    words_by_cell[key, variant] = bits
                arrays.append(bits)
                files[str(path)] = sha(path)
            require(np.array_equal(*arrays), 'selected schedule changed raw output bits')
            for metric in ('full_gpu_ms', 'selected_gpu_ms', 'full_host_ms', 'selected_host_ms'):
                values = cell[metric]
                require(len(values) == 9 and all(type(v) in (int, float) and np.isfinite(v) and v > 0
                                               for v in values), 'invalid measured samples')
            full = statistics.median(cell['full_gpu_ms'])
            selected = statistics.median(cell['selected_gpu_ms'])
            full_host = statistics.median(cell['full_host_ms'])
            selected_host = statistics.median(cell['selected_host_ms'])
            comparisons.append(dict(cohort=cohort, mask=key[0], prefix=key[1], active_owners=owners,
                full_gpu_median_ms=full, selected_gpu_median_ms=selected, gpu_speedup=full/selected,
                full_host_median_ms=full_host, selected_host_median_ms=selected_host,
                host_speedup=full_host/selected_host))
        files[str(out/'report.json')] = sha(out/'report.json')
    files[str(Path(__file__).resolve())] = sha(Path(__file__))
    return dict(collection_integrity_passed=True, ragged_controls=ragged_policy, comparisons=comparisons,
        raw_output_words_matched=2*10*4*128*4096, files=files,
        scope=__doc__, full_model_coding_quality_established=False, full_model_speed_established=False,
        schedule_counter_scope='Logical MPS matrix ranges encoded; not observed hardware work or skipped FLOPs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.campaign)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: report[k] for k in ('collection_integrity_passed', 'ragged_controls', 'raw_output_words_matched')}))
