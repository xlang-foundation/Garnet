#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline Core ML fixture compiler only; native C++ runs all predictions.

Requires coremltools9.0. The tiny stateful identity-convolution accumulator
tests session/state transport, not model performance or actual ANE execution.
"""
import argparse
import hashlib
import json
from pathlib import Path
import coremltools as ct
import numpy as np
from coremltools.converters.mil.mil import Builder as mb, types

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('output', type=Path)
args = parser.parse_args()
assert ct.__version__ == '9.0', 'Pinned converter required'
args.output.mkdir(exist_ok=False)
shape = (1, 32, 1, 16)


@mb.program(input_specs=[mb.TensorSpec(shape, dtype=types.fp16),
                         mb.StateTensorSpec(shape, dtype=types.fp16)],
            opset_version=ct.target.iOS18)
def program(x, accumulator):
    old = mb.read_state(input=accumulator)
    total = mb.add(x=x, y=old)
    mb.coreml_update_state(state=accumulator, value=total)
    projected = mb.conv(x=total, weight=np.eye(32, dtype=np.float16).reshape(32, 32, 1, 1))
    return mb.mul(x=projected, y=np.float16(.5), name='y')


model = ct.convert(program, minimum_deployment_target=ct.target.iOS18,
                   compute_units=ct.ComputeUnit.CPU_ONLY, skip_model_load=True)
package = args.output/'state_fixture.mlpackage'
model.save(str(package))
compiled = ct.models.utils.compile_model(str(package), destination_path=str(args.output/'state_fixture.mlmodelc'))
manifest = dict(coremltools=ct.__version__, numpy=np.__version__, compiled=str(compiled),
                input_shape=shape, input='x', output='y', state='accumulator',
                scope=__doc__, files={})
for path in args.output.rglob('*'):
    if path.is_file():
        manifest['files'][str(path.relative_to(args.output))] = hashlib.sha256(path.read_bytes()).hexdigest()
(args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(compiled)
