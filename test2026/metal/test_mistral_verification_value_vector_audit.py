# SPDX-License-Identifier: Apache-2.0
"""Reject route corruption in a retained complete scalar/vector trace.

Args: RESULT OUTPUT. CPU-only; requires the tokenizer audit dependencies.
"""
import copy, hashlib, json, sys
from pathlib import Path
from audit_mistral_verification_value_vector import validate

data = json.loads(Path(sys.argv[1]).read_text())
assert len(validate(data)) == 16
mutations = {
    'missing_vector_capability': lambda d: d.update(backend=d['backend'].replace('mistral_verification_value_vector4', '')),
    'missing_vector_control': lambda d: d.update(backend=d['backend'].replace('verification_value_route_control', '')),
    'scalar_uses_vector': lambda d: d['records'][1]['rounds'][0]['metrics'].update(target_attention_vector4_dispatches=32),
    'vector_falls_back': lambda d: d['records'][2]['rounds'][0]['metrics'].update(target_attention_vector4_dispatches=0),
    'partial_vector': lambda d: d['records'][2]['rounds'][0]['metrics'].update(target_attention_vector4_dispatches=31),
    'missing_vector_counter': lambda d: d['records'][2]['rounds'][0]['metrics'].pop('target_attention_vector4_dispatches'),
    'scalar_loses_layout': lambda d: d['records'][1]['rounds'][0]['metrics'].update(target_attention_layout_dispatches=0),
    'vector_loses_layout': lambda d: d['records'][2]['rounds'][0]['metrics'].update(target_attention_layout_dispatches=0),
}
rejected = []
for label, mutate in mutations.items():
    changed = copy.deepcopy(data)
    mutate(changed)
    try:
        validate(changed)
    except (ValueError, KeyError, IndexError, TypeError):
        rejected.append(label)
    else:
        raise ValueError('accepted corrupted evidence: ' + label)
with open(sys.argv[2], 'x') as stream:
    json.dump(dict(passed=True, rejected_corruptions=rejected,
                   result_sha256=hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest(),
                   scope=__doc__), stream, indent=2)
print('value-vector audit corruption checks PASS', len(rejected))
