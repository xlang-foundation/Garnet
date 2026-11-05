#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent FP64 arithmetic checks for native BF16-weight Medusa heads.

Checks normalization and every residual feature, then samples final projections
using captured native features. Not draft/target compatibility or speed proof.
"""
import argparse
import hashlib
import json
import mmap
from pathlib import Path
import struct
import numpy as np


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class Weights:
    def __init__(self, path):
        self.file = path.open('rb')
        self.data = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        length, = struct.unpack_from('<Q', self.data)
        require(length < 4*1024*1024, 'unexpected header size')
        self.header = json.loads(self.data[8:8+length])
        self.start = 8+length

    def tensor(self, name, shape, dtype='BF16'):
        entry = self.header[name]
        require(entry['dtype'] == dtype and entry['shape'] == list(shape), 'wrong coefficient shape/type')
        begin, end = entry['data_offsets']
        require(0 <= begin <= end and end-begin == int(np.prod(shape))*2 and self.start+end <= len(self.data),
                'invalid coefficient extent')
        return np.ndarray(shape, dtype='<u2', buffer=self.data, offset=self.start+begin)


def bf16(words):
    return (words.astype('<u4') << 16).view('<f4').astype(np.float64)


def audit(root, model, auxiliary):
    root = root.resolve()
    status = json.loads((root/'status.json').read_text())
    require(status['completed'] is True and len(status['records']) == 3, 'incomplete native probe')
    require(set(status['invalid_profiles_rejected']) == {'negative-head', 'head-four', 'wrong-width', 'wrong-rank',
            'oversized-batch', 'wrong-feature-mode', 'target-as-draft'}, 'missing invalid-profile controls')
    for key in ('garnet_live_matrix_owners', 'garnet_live_affine_weight_owners',
                'garnet_live_bf16_weight_owners', 'garnet_live_pipeline_sets'):
        require(status['cleanup'][key] == 0, 'retained native owner')
    prepared = status['prepared']
    require(prepared['metal_allocated_bytes'] <= prepared['metal_recommended_working_set_bytes']-256*1024*1024,
            'native sampled working set exceeded guard')
    checkpoint = model/'model.safetensors'
    draft = auxiliary/'model.safetensors'
    require(sha(checkpoint) == 'd7182d886f93a9efa73d6d556ee33ffe839a4e09e15965eeef46f295ab5b503d', 'changed target weights')
    require(sha(draft) == 'af35f952136cdb12626cbc272d6ff3334f6e807d49cb53bf121a8fcac7cbacc1', 'changed Medusa artifact')
    target, weights = Weights(checkpoint), Weights(draft)
    scale = target.tensor('model.norm.weight', (4096,), 'F16').view('<f2').astype(np.float64)
    epsilon = json.loads((model/'config.json').read_text())['rms_norm_eps']
    files, inputs, report = {}, {}, []
    def read(path):
        files[str(path)] = sha(path)
        return json.loads(path.read_text())
    labels = ['synthetic-small', 'synthetic-normal', 'trained-eight-token-hidden']
    require([r['label'] for r in status['records']] == labels, 'wrong native inputs')
    for record in status['records']:
        label = record['label']
        raw = np.asarray(read(root/(label+'-input.json')), dtype='<f4')
        normalized = np.asarray(read(root/(label+'-normalized.json')), dtype='<f4')
        require(raw.shape == normalized.shape == (4096,) and np.isfinite(raw).all()
                and np.isfinite(normalized).all(), 'wrong/nonfinite hidden input')
        wanted = raw.astype(np.float64)/np.sqrt(np.mean(raw.astype(np.float64)**2)+epsilon)*scale
        tolerance = 2e-6+4e-6*np.abs(wanted)
        fraction = np.max(np.abs(normalized.astype(np.float64)-wanted)/tolerance)
        require(fraction <= 1, 'target normalization differs from FP64 oracle')
        inputs[label] = normalized.astype(np.float64), record['predictions']
        report.append(dict(label=label, normalization_words=4096, normalization_maximum_error_fraction=float(fraction)))
    projections = []
    for head in range(3):
        linear = bf16(weights.tensor(str(head)+'.0.linear.weight', (4096,4096)))
        bias = bf16(weights.tensor(str(head)+'.0.linear.bias', (4096,)))
        vocabulary = weights.tensor(str(head)+'.1.weight', (32768,4096))
        for label, (normalized, predictions) in inputs.items():
            saved = read(root/(label+'-head'+str(head)+'.json'))
            features = np.asarray(saved['features'], dtype='<f4')
            logits = np.asarray(saved['logits'], dtype='<f4')
            require(features.shape == (4096,) and logits.shape == (32768,)
                    and np.isfinite(features).all() and np.isfinite(logits).all(), 'invalid residual/logits')
            dot = linear@normalized
            absolute = np.abs(linear)@np.abs(normalized)
            biased = dot+bias
            wanted = normalized+biased/(1+np.exp(-biased))
            # Original dot limit, plus bias rounding, SiLU's global derivative
            # bound (<1.2), and the original activation/residual tolerance.
            first_error = 2e-6+4e-6*absolute+1e-7*(np.abs(dot)+np.abs(bias))
            tolerance = 1.2*first_error+2e-6+4e-6*np.abs(wanted)
            fraction = np.max(np.abs(features.astype(np.float64)-wanted)/tolerance)
            require(fraction <= 1, 'residual features exceed independent FP64 bound')
            require(int(np.argmax(logits)) == predictions[head], 'native argmax differs from saved logits')
            maximum, count = 0., 0
            for index in sorted({0, 1, 8192, 16384, 32767, predictions[head]}):
                row = bf16(vocabulary[index])
                products = features.astype(np.float64)*row
                error = abs(float(logits[index])-float(np.sum(products)))
                limit = 2e-6+4e-6*float(np.sum(np.abs(products)))
                require(error <= limit, 'vocabulary projection exceeds original FP64 dot bound')
                maximum = max(maximum, error/limit)
                count += 1
            projections.append(dict(label=label, head=head, residual_words=4096,
                residual_maximum_error_fraction=float(fraction), final_projection_samples=count,
                final_projection_maximum_error_fraction=maximum))
    for path in (root/'status.json', checkpoint, draft, model/'config.json', Path(__file__).resolve()):
        files[str(path)] = sha(path)
    return dict(operator_reference_passed=True, inputs=report, projections=projections, files=files,
        scope=__doc__, draft_target_compatibility_established=False, coding_quality_established=False,
        performance_established=False, arithmetic='Native FP32 operands/intermediates with exact BF16 head weights; not the original BF16 activation path')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--auxiliary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.native, args.model, args.auxiliary)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print('operator_reference_passed', report['operator_reference_passed'])
