"""Independently audit saved native outputs against frozen trained inputs."""
import sys
from pathlib import Path
import copy,hashlib,json,math,sys
import numpy as np
root=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
fixture=root.parent/'mistral-trained-attention-fixture1/replay-fixture1'
out=root/'trained-native4'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
load=lambda p:json.loads(p.read_text())
binding=load(fixture/'fixture-bindings1.json')
for section in ['inputs','outputs']:
    for name,entry in binding[section].items():assert sha(fixture/name)==entry['sha256'],name
budget=root.parent/'mistral-fp16-attention-policy1/numeric-policy-budget1.json'
assert sha(budget)=='6ec2fb97d4438446a551c8968d5d4e8840968d99ff19ff987a95ba8ae7816217'
policy=load(budget)
relative=policy['p16_relative_probability_rounding'];additive=policy['p16_additive_probability_rounding']
assert relative==2**-11 and additive==2**-40
assert load(root/'trained-native4-status.json')['returncode']==0
assert load(root/'trained-reference3-status.json')['returncode']==0
for mode,count in [('on',52),('off',50)]:
    assert load(root/('suite-'+mode+'3-status.json'))['returncode']==0
    assert f'100% tests passed out of {count}' in (root/('suite-'+mode+'3.stdout')).read_text()

def values(path):return np.memmap(path,dtype='<f4',mode='r')
def input_values(ci,i):
    x=values(fixture/f'case{ci}-input{i}.f32');rows=512 if i<3 or i==7 else 4
    width=4096 if i in [0,7] else 1024 if i in [1,2] else 8*4096*128 if i in [3,4] else 128
    assert (x.size-32)%rows==0
    stride=(x.size-32)//rows;assert stride>=width
    assert np.all(x.view('<u4')[:16]==0x7fc12345) and np.all(x.view('<u4')[-16:]==0x7fc12345)
    return x[16:-16].reshape(rows,stride)[:,:width]
def check_fixed(error,bound):
    assert np.all(np.isfinite(error)) and np.all(np.isfinite(bound)) and np.all(bound>0)
    assert np.all(error<=bound),'Fixed half attention numerical budget exceeded'
    return float(np.max(error/bound))

cases=[];payloads={}
for ci in range(3):
    native=values(out/f'case{ci}-native-output.f32').reshape(4,128,32,128)
    rounded=values(out/f'case{ci}-rounded-control.f32').reshape(native.shape)
    original=values(out/f'case{ci}-original-control.f32').reshape(native.shape)
    parent=values(out/f'case{ci}-output-parent.f32').view('<u4')
    assert np.all(parent[:16]==0x7fc12345) and np.all(parent[-16:]==0x7fc12345)
    assert np.array_equal(parent[16:-16],native.reshape(-1).view('<u4'))
    golden=values(fixture/f'case{ci}-native-output.f32')[16:-16].reshape(512,4100)[:,:4096].reshape(native.shape)
    original_fraction=check_fixed(np.abs(original.astype(np.float64)-golden),1e-5+2e-4*np.abs(golden.astype(np.float64)))
    q=input_values(ci,0).reshape(4,128,32,128).astype(np.float16).astype(np.float64)
    k=input_values(ci,1).reshape(4,128,8,128).astype(np.float16).astype(np.float64)
    v=input_values(ci,2).reshape(4,128,8,128).astype(np.float16).astype(np.float64)
    slots=input_values(ci,5);active=input_values(ci,6)
    assert np.array_equal(slots,np.tile(np.arange(1920,2048,dtype=np.float32),(4,1)))
    assert np.all(active==1)
    keys=input_values(ci,3).reshape(4,8,4096,128)[:,:,:2048,:].astype(np.float16).astype(np.float64)
    vals=input_values(ci,4).reshape(4,8,4096,128)[:,:,:2048,:].astype(np.float16).astype(np.float64)
    keys[:,:,1920:2048,:]=k.transpose(0,2,1,3)
    vals[:,:,1920:2048,:]=v.transpose(0,2,1,3)
    assert np.all(np.isfinite(q)) and np.all(np.isfinite(keys)) and np.all(np.isfinite(vals))
    full_fraction=0.;sample_fraction=0.;sample_count=0
    for b in range(4):
        for h in range(32):
            kh=h//4
            max_prefix=np.maximum.accumulate(np.max(np.abs(vals[b,kh]),axis=1))
            visible=np.arange(1921,2049)
            reference=rounded[b,:,h].astype(np.float64)
            bound=1e-5+2e-4*np.abs(reference)+(relative+additive*visible[:,None])*max_prefix[visible-1,None]
            full_fraction=max(full_fraction,check_fixed(np.abs(native[b,:,h].astype(np.float64)-reference),bound))
        for t in [0,64,127]:
            visible=1921+t
            for h in [0,7,8,31]:
                kh=h//4;kv=keys[b,kh,:visible];vv=vals[b,kh,:visible]
                scores=np.sum(kv*q[b,t,h],axis=1)/math.sqrt(128.)
                prob=np.exp(scores-scores.max());prob/=prob.sum()
                expected=np.sum(prob[:,None]*vv,axis=0)
                bound=1e-5+2e-4*np.abs(expected)+relative*np.sum(prob[:,None]*np.abs(vv),axis=0)+additive*visible*np.max(np.abs(vv),axis=0)
                sample_fraction=max(sample_fraction,check_fixed(np.abs(native[b,t,h].astype(np.float64)-expected),bound))
                sample_count+=128
    old_error=np.abs(native.astype(np.float64)-original)
    old_failures=int(np.sum(old_error>1e-5+2e-4*np.abs(original.astype(np.float64))))
    cases.append(dict(layer=[0,15,31][ci],full_output_words=native.size,FP64_values=sample_count,full_half_policy_max_fraction=full_fraction,FP64_half_policy_max_fraction=sample_fraction,
        original_control_vs_frozen_native_max_fraction=original_fraction,old_FP32_contract_failure_words=old_failures,guard_parent_words=parent.size))
    for path in sorted(out.glob(f'case{ci}-*.f32')):payloads[path.name]=sha(path)

# Exercise the numerical audit itself with a nonfinite result, an out-of-bound
# finite result and a mutated guard; none may be admitted.
rejected=[]
for name,error,bound in [('nonfinite',np.array([np.nan]),np.array([1.])),('over_budget',np.array([1.0001]),np.array([1.]))]:
    try:check_fixed(error,bound)
    except AssertionError:rejected.append(name)
    else:raise AssertionError('Corrupt numerical fixture admitted')
guard=np.full(32,0x7fc12345,dtype='<u4');guard[0]=0
try:assert np.all(guard==0x7fc12345)
except AssertionError:rejected.append('guard_mutation')
else:raise AssertionError('Corrupt guard admitted')
result=dict(passed=True,scope=__doc__,cases=cases,full_output_words=sum(c['full_output_words'] for c in cases),FP64_values=sum(c['FP64_values'] for c in cases),
    fixed_budget_sha256=sha(budget),trained_fixture_binding_sha256=sha(fixture/'fixture-bindings1.json'),candidate_binding_sha256=sha(root/'candidate-bindings7.json'),payloads_sha256=payloads,
    native_status_sha256=sha(root/'trained-native4-status.json'),negative_controls_rejected=rejected,
    old_FP32_contract_qualified=False,full_model_quality_qualified=False,prefix_reuse_qualified=False,serving_speed_claim=False,goal_complete=False)
dest=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else root/'trained-audit7.json';assert not dest.exists();dest.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='payloads_sha256'},indent=2))
