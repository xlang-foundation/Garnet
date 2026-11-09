#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reject incomplete or inconsistent claims for channel/range export profiles.

Args: EXPORTED_MANIFEST NEW_OUTPUT. CPU-only; no model inference.
"""
import copy, hashlib, json, sys
from pathlib import Path
from audit_mistral_mlp_probe import validate_channel_profile

manifest, output = map(Path, sys.argv[1:])
model = json.loads(manifest.read_text())
assert model['intermediate_tile'] == 2048 and model['activation_scale'] == 64
validate_channel_profile(model)
mutations = {
 'unknown_tile': lambda m:m.update(intermediate_tile=1536),
 'boolean_tile': lambda m:m.update(intermediate_tile=True),
 'wrong_conv_count': lambda m:m.update(projection_convolutions=3),
 'wrong_reduction': lambda m:m.update(reduction='single_conv'),
 'lut_channel_tile': lambda m:m.update(representation='lut'),
 'mixed_inner_tile': lambda m:m.update(inner_tile=256),
 'wrong_arithmetic': lambda m:m.update(arithmetic='fp32_prefill_control'),
 'unknown_scale': lambda m:m.update(activation_scale=3),
 'boolean_scale': lambda m:m.update(activation_scale=True),
 'missing_scale_inverse': lambda m:m.pop('scale_cancellation'),
 'wrong_scale_inverse': lambda m:m.update(scale_cancellation='output scale omitted'),
 'missing_scale_words': lambda m:m.pop('scaled_up_roundtrip_words_matched'),
 'wrong_scale_words': lambda m:m.update(scaled_up_roundtrip_words_matched=1),
 'wrong_weight_coverage': lambda m:m.update(remapped_tile_weight_words_matched=1),
}
rejected=[]
for label,mutate in mutations.items():
 changed=copy.deepcopy(model);mutate(changed)
 try:validate_channel_profile(changed)
 except (ValueError,KeyError):rejected.append(label)
 else:raise ValueError('accepted profile corruption: '+label)
# Legacy default exports have no new fields; explicit default fields also work.
validate_channel_profile({})
validate_channel_profile(dict(intermediate_tile=0, activation_scale=1))
with output.open('x') as stream:
 json.dump(dict(passed=True,rejected_corruptions=rejected,manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),scope=__doc__),stream,indent=2)
print('channel profile negative checks PASS',len(rejected))
