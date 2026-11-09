"""CPU FP64 semantic probe of original vs half-rounded trained attention operands.
This diagnoses a new numerical policy; it admits no kernel, timing or quality result.
"""
import sys
from pathlib import Path
import hashlib
import json
import math
import numpy as np

p=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
f=p.parent/'mistral-trained-attention-fixture1/replay-fixture1'
def sha(q):
    h=hashlib.sha256()
    with q.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
binding=json.loads((f/'fixture-bindings1.json').read_text())
assert binding['layers']==[0,15,31]
for group in ['inputs','outputs']:
    for name,record in binding[group].items():assert sha(f/name)==record['sha256']
assert not (p/'half-operand-probe1.json').exists()
layers=[];payloads={}
for case,layer in enumerate([0,15,31]):
    read=lambda name:np.memmap(f/f'case{case}-{name}.f32',dtype='<f4',mode='r')[16:-16]
    q=read('input0').reshape(4,128,4100)[:,:,:4096].reshape(4,128,32,128)
    k=read('input1').reshape(4,128,1028)[:,:,:1024].reshape(4,128,8,128)
    v=read('input2').reshape(4,128,1028)[:,:,:1024].reshape(4,128,8,128)
    kc=read('input3').reshape(4,8,4096,128);vc=read('input4').reshape(4,8,4096,128)
    slots=read('input5').reshape(4,131)[:,:128]
    active=read('input6').reshape(4,133)[:,:128]
    native=read('native-output').reshape(4,128,4100)[:,:,:4096].reshape(4,128,32,128)
    assert np.array_equal(slots,np.broadcast_to(np.arange(1920,2048),(4,128))) and np.all(active==1)
    rows=[];originals=[];halves=[];gold=[];pv_budgets=[]
    for b in range(4):
        for t in [0,64,127]:
            visible=int(slots[b,t])+1
            for h in [0,7,8,31]:
                kh=h//4
                keys=kc[b,kh,:visible].astype(np.float64).copy()
                vals=vc[b,kh,:visible].astype(np.float64).copy()
                keys[1920:visible]=k[b,:t+1,kh];vals[1920:visible]=v[b,:t+1,kh]
                query=q[b,t,h].astype(np.float64)
                qh=query.astype(np.float16).astype(np.float64)
                keys_h=keys.astype(np.float16).astype(np.float64)
                vals_h=vals.astype(np.float16).astype(np.float64)
                assert np.isfinite(qh).all() and np.isfinite(keys_h).all() and np.isfinite(vals_h).all()
                score=np.sum(keys*query,axis=1)/math.sqrt(128)
                score_h=np.sum(keys_h*qh,axis=1)/math.sqrt(128)
                prob=np.exp(score-score.max());prob/=prob.sum()
                prob_h=np.exp(score_h-score_h.max());prob_h/=prob_h.sum()
                exact=np.sum(prob[:,None]*vals,axis=0)
                exact_h=np.sum(prob_h[:,None]*vals_h,axis=0)
                # Diagnostic budget for ONE rounding of normalized P to half.
                # Online tiled P rounding and GPU float summation are not yet
                # covered by this term and cannot be admitted using it.
                pv_budget=np.sum(prob_h[:,None]*np.abs(vals_h),axis=0)*2**-11
                native_row=native[b,t,h].astype(np.float64)
                old_tolerance=1e-5+2e-4*np.abs(exact)
                assert np.all(np.abs(native_row-exact)<=old_tolerance)
                delta=np.abs(exact_h-exact)
                rows.append(dict(batch=b,query=t,head=h,visible=visible,
                    half_operand_old_bound_failures=int(np.sum(delta>old_tolerance)),
                    half_operand_old_bound_fraction_max=float(np.max(delta/old_tolerance)),
                    max_logit_change=float(np.max(np.abs(score_h-score))),
                    probability_L1_change=float(np.sum(np.abs(prob_h-prob))),
                    q_quantization_max_abs=float(np.max(np.abs(qh-query))),
                    k_quantization_max_abs=float(np.max(np.abs(keys_h-keys))),
                    v_quantization_max_abs=float(np.max(np.abs(vals_h-vals)))))
                originals.append(exact);halves.append(exact_h);gold.append(native_row);pv_budgets.append(pv_budget)
    original=np.stack(originals);half=np.stack(halves);native_gold=np.stack(gold)
    tolerance=1e-5+2e-4*np.abs(original);error=half-original
    for name,a in [('original_fp64',original),('half_operands_fp64',half),('native_gold',native_gold),('single_normalized_P_rounding_term',np.stack(pv_budgets))]:
        payloads[f'layer{layer}_{name}']=a
    layers.append(dict(layer=layer,values=original.size,rows=rows,
        max_abs_output_change=float(np.max(np.abs(error))),rms_output_change=float(np.sqrt(np.mean(error*error))),
        half_operand_old_bound_failures=int(np.sum(np.abs(error)>tolerance)),
        half_operand_old_bound_fraction_max=float(np.max(np.abs(error)/tolerance)),
        original_native_gold_bound_fraction_max=float(np.max(np.abs(native_gold-original)/tolerance))))
np.savez(p/'half-operand-probe1.npz',**payloads)
result=dict(passed=True,scope=__doc__,layers=layers,FP64_values=18432,
    fixture_binding_sha256=sha(f/'fixture-bindings1.json'),payload_sha256=sha(p/'half-operand-probe1.npz'),
    original_bound='1e-5 + 2e-4*abs(original FP64 reference)',
    original_native_gold_gate_passed=True,half_policy_kernel_qualified=False,
    original_fp32_contract_qualified=False,probability_rounding_included=False,
    native_integration=False,serving_speed_claim=False,quality_qualified=False,goal_complete=False,
    interpretation='This independently measures Q/K/V half rounding only. Original-policy failures remain failures. No GPU or timing work. Full FP16 policy, probability rounding, model logits, cache/batching and coding quality require separate evidence.')
(p/'half-operand-probe1.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'passed':True,'layers':[{k:l[k] for k in ['layer','max_abs_output_change','half_operand_old_bound_failures','half_operand_old_bound_fraction_max']} for l in layers]},indent=2))
