#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise real trace reference resolution and reject corrupted attribution.

Args: CAMPAIGN AUDIT OUTPUT. CPU-only; audit must come from the same campaign.
"""
import copy, hashlib, json, sys, tempfile
from pathlib import Path
import xml.etree.ElementTree as ET
from audit_mistral_ane_trace import validate, intervals

root, evidence, output = map(Path,sys.argv[1:])
report = json.loads(evidence.read_text())
records = report['records']
assert validate(records)
mutations = {
 'missing_control':lambda r:r.pop(),
 'reused_pid':lambda r:r[1].update(pid=r[0]['pid']),
 'failed_target':lambda r:r[0].update(exit_status=1),
 'overlapping_targets':lambda r:r[1].update(started_ns=r[0]['started_ns']),
 'missing_prediction':lambda r:r[0]['intervals'].pop(),
 'wrong_device_channel':lambda r:r[0]['intervals'][2].update(channel='CPU'),
 'inactive_prediction':lambda r:r[0]['intervals'][2].update(state='Idle'),
 'unattributed_model':lambda r:r[0]['intervals'][2].update(label='unrelated_model Prediction'),
 'negative_duration':lambda r:r[0]['intervals'][2].update(duration_ns=-1),
 'outside_capture':lambda r:r[0]['intervals'][2].update(start_ns=10**18),
 'overlapping_predictions':lambda r:r[0]['intervals'][3].update(start_ns=r[0]['intervals'][2]['start_ns']),
 'changed_output':lambda r:r[0].update(output_sha256='0'*64),
 'wrong_predict_count':lambda r:r[0]['report']['native_report'].update(predictions=11),
 'wrong_policy':lambda r:r[1]['report'].update(policy='cpu_ane'),
 'wrong_profile':lambda r:r[0]['report'].update(rows=512),
 'nan_latency':lambda r:r[0]['report']['host_to_host_ms'].__setitem__(0,float('nan')),
 'boolean_latency':lambda r:r[0]['report']['host_to_host_ms'].__setitem__(0,True),
}
rejected=[]
for label,mutate in mutations.items():
 changed=copy.deepcopy(records);mutate(changed)
 try:validate(changed)
 except ValueError:rejected.append(label)
 else:raise ValueError('accepted corruption: '+label)
# Explicitly test an invented CPU-only ANE event within its capture bounds.
changed=copy.deepcopy(records)
event=copy.deepcopy(records[0]['intervals'][2]);event.update(start_ns=0,duration_ns=1)
changed[1]['intervals']=[event]
try:validate(changed)
except ValueError:rejected.append('cpu_only_ane_event')
else:raise ValueError('accepted CPU-only ANE event')
source=root/'ane-hw-intervals1.xml'
assert len(intervals(source)) == len(records[0]['intervals'])
with tempfile.TemporaryDirectory(prefix='ane-xml-') as folder:
 for label in ('unknown_reference','wrong_schema','invalid_duration'):
  tree=ET.parse(source)
  if label=='unknown_reference':next(e for e in tree.iter() if 'ref' in e.attrib).set('ref','999999999')
  elif label=='wrong_schema':tree.find('.//schema').set('name','not-ane')
  else:tree.find('.//row/duration').text='nan'
  path=Path(folder)/'bad.xml';tree.write(path)
  try:intervals(path)
  except ValueError:rejected.append(label)
  else:raise ValueError('accepted XML corruption: '+label)
with output.open('x') as stream:
 json.dump(dict(passed=True,rejected_corruptions=rejected,audit_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),scope=__doc__),stream,indent=2)
print('ANE trace corruption checks PASS',len(rejected))
