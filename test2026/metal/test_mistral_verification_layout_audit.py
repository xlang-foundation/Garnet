# SPDX-License-Identifier: Apache-2.0
"""Check layout-specific corruptions against the retained complete paired trace.
Args: RESULT OUTPUT. CPU-only; requires the same tokenizer dependencies as audit.
"""
import copy,hashlib,json,sys
from pathlib import Path
from audit_mistral_verification_layout import validate

data=json.loads(Path(sys.argv[1]).read_text());assert len(validate(data))==16
mutations={
 'missing_layout_capability':lambda d:d.update(backend=d['backend'].replace('mistral_verification_layout_b1t4_c4096_h32_kv8_d128','')),
 'missing_route_control':lambda d:d.update(backend=d['backend'].replace('verification_layout_route_control','')),
 'control_uses_layout':lambda d:d['records'][1]['rounds'][0]['metrics'].update(target_attention_layout_dispatches=32),
 'layout_falls_back':lambda d:d['records'][2]['rounds'][0]['metrics'].update(target_attention_layout_dispatches=0),
 'partial_layout':lambda d:d['records'][2]['rounds'][0]['metrics'].update(target_attention_layout_dispatches=31),
 'missing_layout_counter':lambda d:d['records'][2]['rounds'][0]['metrics'].pop('target_attention_layout_dispatches'),
}
rejected=[]
for label,mutate in mutations.items():
 changed=copy.deepcopy(data);mutate(changed)
 try:validate(changed)
 except (ValueError,KeyError,IndexError,TypeError):rejected.append(label)
 else:raise ValueError('accepted corruption: '+label)
result=dict(passed=True,rejected_corruptions=rejected,result_sha256=hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest(),scope=__doc__)
with open(sys.argv[2],'x') as f:json.dump(result,f,indent=2)
print('layout-specific audit corruption checks PASS',len(rejected))
