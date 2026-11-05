#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Report teacher-forced numerical differences without declaring a speed/quality pass."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = [args.reference/'reference.json', args.reference/'logits.npy', args.native]
    reference = json.loads(paths[0].read_text())
    expected = np.asarray(np.load(paths[1], allow_pickle=False), dtype=np.float64)
    native = json.loads(args.native.read_text())
    assert native['completed'] and 'affine_coefficient_dtype=fp16' in native['backend']
    assert [row['chunk'] for row in native['records']] == [0, 16]
    assert expected.shape == (reference['steps'], 32768) and np.isfinite(expected).all()
    assert expected.argmax(axis=1).tolist() == reference['forced_ids'], 'teacher input differs from original greedy reference'
    for key in ('garnet_live_matrix_owners', 'garnet_live_affine_weight_owners',
                'garnet_live_bf16_weight_owners', 'garnet_live_pipeline_sets'):
        assert native['cleanup'][key] == 0
    modes, arrays = [], []
    for record in native['records']:
        actual = np.asarray(record['logits'], dtype=np.float64)
        assert actual.shape == expected.shape and np.isfinite(actual).all()
        assert record['slots'] == [len(reference['case']['prompt_ids'])+reference['steps']-1]
        arrays.append(actual)
        steps = []
        for step, (row, target) in enumerate(zip(actual, expected)):
            order = np.argsort(target)
            winner, reference_winner = int(row.argmax()), int(target.argmax())
            steps.append(dict(step=step, native_token=winner, reference_token=reference_winner,
                              token_match=winner == reference_winner,
                              reference_top2_margin=float(target[order[-1]]-target[order[-2]]),
                              native_reference_vs_native_margin=float(row[reference_winner]-row[winner]),
                              reference_native_vs_reference_margin=float(target[winner]-target[reference_winner]),
                              max_abs=float(np.abs(row-target).max()),
                              mean_abs=float(np.abs(row-target).mean())))
        modes.append(dict(chunk=record['chunk'], steps=steps))
    delta = np.abs(arrays[0]-arrays[1])
    report = dict(completed=True, modes=modes,
                  native_ordinary_vs_chunk=dict(max_abs=float(delta.max()), mean_abs=float(delta.mean()),
                                               argmax_equal=(arrays[0].argmax(axis=1) == arrays[1].argmax(axis=1)).tolist()),
                  files={str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
                  analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='Numerical diagnosis only. Small errors or close argmax margins do not by themselves prove model correctness or quality equivalence.')
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report['native_ordinary_vs_chunk']))
    for mode in modes:
        print(json.dumps(dict(chunk=mode['chunk'], differing_steps=[row for row in mode['steps'] if not row['token_match']])))


if __name__ == '__main__':
    main()
