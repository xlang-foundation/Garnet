# SPDX-License-Identifier: Apache-2.0
"""Synthetic corruption tests for the diagnostic auditor, not native math tests."""
import copy,json,sys
from audit_mistral_verifier_profile import audit
labels=['EncodeAffineInt4MatMul/4/4096/14336']*225+['EncodeStateKernel/22']*32+['EncodeStateKernel/23']*32+['EncodeStateKernel/26']*64
enc=[dict(label=label,start_ticks=100+i*1000,end_ticks=1100+i*1000,ms=.001,valid=True) for i,label in enumerate(labels)]
sample=dict(sequence=0,context=1,gpu_ms=.4,timestamp_hz=24000000,dropped_encoders=0,invalid_encoders=0,
 complete_compute_sampling=True,command_success=True,encoders=enc,calibration=dict(cpu_start_ns=100,gpu_start=100,cpu_end_ns=1000100,gpu_end=1000100,ns_per_counter_tick=1.,valid=True))
round0=dict(slot=1,consumed=1,submitted_ids=[1,9,0,0],emitted_ids=[2],metrics=dict(target_affine_dispatches=225,target_affine_misses=0,target_gpu_ms=.4))
row=dict(case_id='fixture',mode='resident',generated_ids=[1,2],state_sha256=[['a','b']],hidden_sha256='h',rounds=[round0])
cleanup=dict(garnet_live_matrix_owners=0,garnet_live_affine_weight_owners=0,garnet_live_bf16_weight_owners=0,garnet_live_pipeline_sets=0,metal_allocated_bytes=0)
original=dict(completed=True,inputs=dict(cases=[dict(id='fixture')]),records=[copy.deepcopy(row)])
replay=dict(completed=True,cases=['fixture'],records=[row],cleanup=cleanup)
base=([sample],replay,original)
assert audit(*base)['passed']
mutations={
 'dropped':lambda s,r,o:s[0].update(dropped_encoders=1),
 'invalid_count':lambda s,r,o:s[0].update(invalid_encoders=1),
 'incomplete_sampling':lambda s,r,o:s[0].update(complete_compute_sampling=False),
 'failed_command':lambda s,r,o:s[0].update(command_success=False),
 'invalid_sample':lambda s,r,o:s[0]['encoders'][0].update(valid=False),
 'wrong_units':lambda s,r,o:s[0]['calibration'].update(ns_per_counter_tick=41.6667),
 'reversed_time':lambda s,r,o:s[0]['encoders'][0].update(end_ticks=0),
 'wrong_duration':lambda s,r,o:s[0]['encoders'][0].update(ms=1.),
 'missing_score':lambda s,r,o:s[0]['encoders'].pop(225),
 'missing_backup':lambda s,r,o:s[0]['encoders'].pop(),
 'missing_affine':lambda s,r,o:s[0]['encoders'].pop(0),
 'duplicate_command':lambda s,r,o:s.append(copy.deepcopy(s[0])),
 'mismatched_command':lambda s,r,o:s[0].update(gpu_ms=.41),
 'invalid_calibration':lambda s,r,o:s[0]['calibration'].update(cpu_end_ns=0),
 'missing_samples':lambda s,r,o:s.clear(),
 'changed_ids':lambda s,r,o:r['records'][0].update(generated_ids=[1,3]),
 'changed_kv':lambda s,r,o:r['records'][0].update(state_sha256=[['bad']]),
 'changed_hidden':lambda s,r,o:r['records'][0].update(hidden_sha256='bad'),
 'changed_emitted':lambda s,r,o:r['records'][0]['rounds'][0].update(emitted_ids=[3]),
 'missing_round':lambda s,r,o:r['records'][0]['rounds'].clear(),
 'live_owner':lambda s,r,o:r['cleanup'].update(garnet_live_matrix_owners=1),
 'cleanup_bytes':lambda s,r,o:r['cleanup'].update(metal_allocated_bytes=5000000),
 'missing_case':lambda s,r,o:r['cases'].clear(),
 'wrong_case':lambda s,r,o:r['records'][0].update(case_id='wrong'),
 'incomplete_run':lambda s,r,o:r.update(completed=False),
 'unprepared_projection':lambda s,r,o:r['records'][0]['rounds'][0]['metrics'].update(target_affine_misses=1),
 'counter_outside_calibration':lambda s,r,o:s[0]['encoders'][0].update(start_ticks=0),
 'unexplained_sampling_gap':lambda s,r,o:(s[0].update(gpu_ms=2.),r['records'][0]['rounds'][0]['metrics'].update(target_gpu_ms=2.)),
}
passed=[]
for name,mutate in mutations.items():
 args=copy.deepcopy(base);mutate(*args)
 try:audit(*args)
 except (ValueError,KeyError,StopIteration):passed.append(name)
 else:raise AssertionError('corruption accepted: '+name)
result=dict(passed=True,positive=1,rejected_corruptions=passed,scope=__doc__)
if len(sys.argv)>1:
 with open(sys.argv[1],'w') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
