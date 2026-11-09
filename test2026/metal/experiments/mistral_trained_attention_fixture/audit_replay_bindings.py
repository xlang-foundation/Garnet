"""Bind actual tensor capture, row-padded fixtures and native operator replay."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
r=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
load=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
assert load(r/'trained-capture-audit2.json')['passed'] and load(r/'replay1/audit1.json')['passed']
assert load(r/'replay-status1.json')['status']=='terminal' and load(r/'replay-status1.json')['returncode']==0
assert load(r/'replay-compile-status1.json')['returncode']==0
b=load(r/'replay-fixture1/fixture-bindings1.json');assert b['layers']==[0,15,31]
assert sha(r/'capture2/capture.json')==b['source_capture_metadata_sha256'] and sha(r/'bindings2.json')==b['model_bindings_sha256']
old=r.parent/'mistral-strict-mpp-attention1';assert sha(old/'combined6.metal')==sha(r/'replay-combined1.metal')==b['qualified_shader_sha256']
assert sha(old/'strict_attention3.mm')==b['base_host_sha256']
meta=load(r/'capture2/capture.json');padded=0;gold=0;equal_old=0;inputs={};actual_hashes=load(r/'capture2-payload-hashes1.json')
for filename,h in actual_hashes.items():
 assert sha(r/'capture2'/filename)==h
 equal_old+=int(sha(r/'capture1'/filename)==h)
for case,layer in enumerate(meta['layers']):
 for i,p in enumerate(layer['planes']):
  filename=f'case{case}-'+('native-output' if i==8 else f'input{i}')+'.f32'
  record=b['outputs' if i==8 else 'inputs'][filename];assert record['layer']==layer['layer'] and record['source_file']==p['file']
  assert sha(r/'capture2'/p['file'])==record['source_sha256'];assert sha(r/'replay-fixture1'/filename)==record['sha256']
  source=np.memmap(r/'capture2'/p['file'],dtype='<u4',mode='r')[16:-16].reshape(p['rows'],p['columns'])
  destination=np.memmap(r/'replay-fixture1'/filename,dtype='<u4',mode='r');stride=(destination.size-32)//p['rows'];assert stride>=p['columns']
  parent=destination[16:-16].reshape(p['rows'],stride);assert np.array_equal(source,parent[:,:p['columns']]);assert np.all(parent[:,p['columns']:]==0x7fc12345)
  if i<8:
   assert sha(r/'replay1'/filename)==record['sha256'];inputs[filename]=record['sha256'];padded+=p['logical_elements']
  else:gold+=p['logical_elements']
a=load(r/'replay1/audit1.json');assert a['original_bound']=='1e-5 + 2e-4*abs(FP64 expected)' and a['fp64_values_per_variant']==18432
assert all(len(c['fp64_bound_fraction_max'])==4 and max(c['fp64_bound_fraction_max'])<=1 and max(c['complete_control_bound_fraction_max'])<=1 for c in a['cases'])
report=load(r/'replay1/correctness1.json');assert report['register_layout_status']==[0]*32 and report['actual_trained_operands'] and not report['timing_admitted']
files=[r/'trained_attention_replay1.mm',r/'trained_attention_replay1',r/'replay-combined1.metal',r/'audit_trained_replay1.py',r/'replay-compile-command1.json',r/'replay-command1.json']
result=dict(passed=True,scope=__doc__,layers=[0,15,31],bit_preserved_input_words=padded,native_gold_words=gold,complete_native_gold_bound_comparisons=gold*3,FP64_points_per_variant=18432,FP64_variant_comparisons=18432*4,variant_order=['original4mps','committed2mps','strict_mpp_register','captured_native_gold'],max_native_gold_bound_fraction_by_layer=[c['complete_control_bound_fraction_max'] for c in a['cases']],max_FP64_bound_fraction_by_layer=[c['fp64_bound_fraction_max'] for c in a['cases']],payloads_exact_to_rejected_first_capture=equal_old,source_files_sha256={str(p.relative_to(r)):sha(p) for p in files},operator_input_files_sha256=inputs,no_precision_conversion=True,native_input_and_gold_capture_bound_to_model_state=True,performance_measured=False,full_model_new_numeric_policy_qualified=False,quality_qualified=False,goal_complete=False)
(r/'replay-bindings-audit1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
