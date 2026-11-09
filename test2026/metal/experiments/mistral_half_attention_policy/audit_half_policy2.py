"""Independent saved-data audit of a new half-QKV policy; old FP32 failures remain visible."""
from pathlib import Path
import hashlib,importlib.util,json,math,sys
import numpy as np
p=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
old=p.parent/'mistral-compensated-mpp-attention1'
def sha(q):
    h=hashlib.sha256()
    with q.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
load=lambda q:json.loads(q.read_text())
frozen=load(old/'provenance1.json')['files_sha256']
module=old/'audit_compensated_attention1.py';assert sha(module)==frozen[module.name]
spec=importlib.util.spec_from_file_location('original_guard_and_FP64_audit',module)
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
bindings=load(p/'source-bindings2.json')
for name,h in bindings['sha256'].items():assert sha(p/name)==h,name
policy=load(p/'numeric-policy-budget1.json')
assert policy['p16_relative_probability_rounding']==2**-11 and policy['p22_relative_probability_rounding']==2**-22
assert policy['p16_additive_probability_rounding']==2**-40 and policy['p22_additive_probability_rounding']==2**-39
results=[];old_controls=0;metadata_hashes={};payload_hashes={}
for kind in ['synthetic','trained']:
    raw=p/f'raw-{kind}2';rounded=p/f'rounded-{kind}2';fixture=p/f'half-fixture-{kind}1'
    a=load(raw/'correctness1.json');b=load(rounded/'correctness1.json')
    assert len(a['cases'])==len(b['cases'])==(10 if kind=='synthetic' else 3)
    conversion=load(fixture/'conversion-bindings1.json');assert conversion['passed']
    for name,r in conversion['files'].items():
        assert sha(raw/name)==r['raw_sha256']
        assert sha(fixture/name)==sha(rounded/name)==r['rounded_sha256']
    for directory,report in [(raw,a),(rounded,b)]:
        assert load(p/(directory.name+'-status.json'))['returncode']==0
        assert report['collected'] and report['actual_trained_operands']==(kind=='trained')
        assert report['variant_names']==['original4mps','committed2mps','fp16_qkv_p16_v1','fp16_qkv_p22_v1']
        assert report['relaxed_precision'] is True and report['math_mode']=='Safe' and report['qkv_scale']==1 and report['probability_scale']==32768
        assert report['packed_half_storage'] and report['Q_pack_inside_gpu_span'] and report['half_KV_fixture_prepared_before_timing']
        assert not report['timing_admitted'] and not report['native_integration'] and not report['serving_speed_claim']
        assert report['half_policy_layout_capacities']==[8,16,16,8,16,16]*32
        assert report['register_layout_status']==[0]*32
        assert report['half_policy_coordinate_parent_words']==[0x7fc12345]*16+[1]*2560+[0]+[0x7fc12345]*16
        metadata_hashes[directory.name]=sha(directory/'correctness1.json')
    for ci,(c,d) in enumerate(zip(a['cases'],b['cases'])):
        assert c==d
        # Reuse the independent original guard/control/causal FP64 audit without
        # changing that frozen source or writing into its study.
        original_raw=original.audit_cell(raw,ci,c)
        original_rounded=original.audit_cell(rounded,ci,d)
        assert all(original_raw['qualified_variants'][v] for v in [0,1])
        assert all(original_rounded['qualified_variants'][v] for v in [0,1])
        for variant in [0,1]:
            prior=old/('synthetic1' if kind=='synthetic' else 'trained1')/f'case{ci}-output{variant}.f32'
            current=raw/prior.name;assert sha(current)==sha(prior)
            old_controls+=current.stat().st_size//4
        for directory,shape in [(raw,c),(rounded,d)]:
            Bc,Tc,Cc,Ec=[shape[k] for k in ['batch','chunk','capacity','extent']]
            for half_name,float_name in [('half-k','merged-k'),('half-v','merged-v'),('packed-q','input0')]:
                actual=np.fromfile(directory/f'case{ci}-{half_name}.f16',dtype='<u2')
                source=np.fromfile(directory/f'case{ci}-{float_name}.f32',dtype='<f4')
                expected=np.full(actual.size,0x7e09,dtype='<u2')
                assert actual.size==source.size
                if half_name=='packed-q':
                    dest=expected[16:-16].reshape(Bc*Tc,4100)[:,:4096]
                    src=source[16:-16].reshape(Bc*Tc,4100)[:,:4096]
                else:
                    dest=expected[16:-16].reshape(Bc,8,Cc,128)[:,:,:Ec,:]
                    src=source[16:-16].reshape(Bc,8,Cc,128)[:,:,:Ec,:]
                dest[:]=src.astype(np.float16).view(np.uint16)
                assert np.array_equal(actual,expected),half_name+' physical payload/guard mismatch'
        for variant in [2,3]:
            assert sha(raw/f'case{ci}-output{variant}.f32')==sha(rounded/f'case{ci}-output{variant}.f32'),'Half conversion must be idempotent'
        B,T,C,E=[c[k] for k in ['batch','chunk','capacity','extent']];r=c['params']
        data=lambda folder,name:np.memmap(folder/f'case{ci}-{name}.f32',dtype='<f4',mode='r')
        q=data(rounded,'input0')[16:-16].reshape(B,T,4100)[:,:,:4096].reshape(B,T,32,128)
        k=data(rounded,'input1')[16:-16].reshape(B,T,1028)[:,:,:1024].reshape(B,T,8,128)
        v=data(rounded,'input2')[16:-16].reshape(B,T,1028)[:,:,:1024].reshape(B,T,8,128)
        kc=data(rounded,'input3')[16:-16].reshape(B,8,C,128)
        vc=data(rounded,'input4')[16:-16].reshape(B,8,C,128)
        slots=data(rounded,'input5')[16:-16].reshape(B,r[11])[:,:T]
        active=data(rounded,'input6')[16:-16].reshape(B,r[12])[:,:T]
        gate=data(rounded,'input7')[16:-16].reshape(B,T,4100)[:,:,:4096].reshape(B,T,32,128)
        out=lambda folder,variant:original.logical_output(data(folder,'output'+str(variant)),B,T,4100,active)
        reference=out(rounded,0);candidate=[out(raw,v) for v in [2,3]]
        merged_v=data(rounded,'merged-v')[16:-16].reshape(B,8,C,128)
        full_max_fraction=[0.,0.];sample_max_fraction=[0.,0.];sample_old_half_fraction=[0.,0.]
        old_raw_failure_counts=[0,0];raw_control=out(raw,0)
        for vi,result in enumerate(candidate):
            old_tolerance=1e-5+2e-4*np.abs(raw_control.astype(np.float64))
            old_raw_failure_counts[vi]=int(np.sum(np.abs(result.astype(np.float64)-raw_control)>old_tolerance))
        relative=[2**-11,2**-22];additive=[2**-40,2**-39];points=0
        for owner in range(B):
            count=c['counts'][owner]
            if not count:continue
            for head in range(32):
                kh=head//4
                max_prefix=np.maximum.accumulate(np.max(np.abs(merged_v[owner,kh,:E].astype(np.float64)),axis=1))
                for t in range(count):
                    visible=int(slots[owner,t])+1;maxv=max_prefix[visible-1]
                    expected=reference[owner,t,head].astype(np.float64)
                    base=1e-5+2e-4*np.abs(expected)
                    for vi in range(2):
                        bound=base+(relative[vi]+additive[vi]*visible)*maxv
                        fraction=np.abs(candidate[vi][owner,t,head].astype(np.float64)-expected)/bound
                        full_max_fraction[vi]=max(full_max_fraction[vi],float(np.max(fraction)))
            for t in sorted(set([0,count//2,count-1])):
                visible=int(slots[owner,t])+1
                for head in [0,7,8,31]:
                    kh=head//4;keys=kc[owner,kh,:visible].astype(np.float64).copy();values=vc[owner,kh,:visible].astype(np.float64).copy()
                    for j in range(t+1):
                        position=int(slots[owner,j]);keys[position]=k[owner,j,kh];values[position]=v[owner,j,kh]
                    scores=np.sum(keys*q[owner,t,head].astype(np.float64),axis=1)/math.sqrt(128)
                    prob=np.exp(scores-scores.max());prob/=prob.sum()
                    expected=np.sum(prob[:,None]*values,axis=0)
                    weighted=np.sum(prob[:,None]*np.abs(values),axis=0);maxv=np.max(np.abs(values))
                    if c['gated']:expected/=1+np.exp(-gate[owner,t,head].astype(np.float64))
                    base=1e-5+2e-4*np.abs(expected)
                    for vi in range(2):
                        error=np.abs(candidate[vi][owner,t,head].astype(np.float64)-expected)
                        bound=base+relative[vi]*weighted+additive[vi]*visible*maxv
                        sample_max_fraction[vi]=max(sample_max_fraction[vi],float(np.max(error/bound)))
                        sample_old_half_fraction[vi]=max(sample_old_half_fraction[vi],float(np.max(error/base)))
                    points+=128
        results.append(dict(kind=kind,case=ci,FP64_values_per_variant=points,
            full_new_policy_bound_fraction_max=full_max_fraction,FP64_new_policy_bound_fraction_max=sample_max_fraction,
            FP64_original_float_bound_on_half_inputs_fraction_max=sample_old_half_fraction,
            original_raw_FP32_control_failures=old_raw_failure_counts,
            qualified_variants=[full_max_fraction[i]<=1 and sample_max_fraction[i]<=1 for i in range(2)],
            original_raw_diagnostics=original_raw,original_rounded_diagnostics=original_rounded))
    for directory in [raw,rounded]:
        payload_hashes[directory.name]={q.name:sha(q) for q in sorted([*directory.glob('*.f32'),*directory.glob('*.f16')])}
qualified=[all(c['qualified_variants'][i] for c in results) for i in range(2)]
result=dict(passed=all(qualified),qualified_variants=qualified,variant_names=['fp16_qkv_p16_v1','fp16_qkv_p22_v1'],
    scope=__doc__,cases=results,exact_preserved_original_MPS_control_word_pairs=old_controls,
    FP64_values_per_candidate=sum(c['FP64_values_per_variant'] for c in results),
    metadata_sha256=metadata_hashes,numeric_budget_sha256=sha(p/'numeric-policy-budget1.json'),
    source_bindings_sha256=sha(p/'source-bindings2.json'),original_FP32_contract_qualified=False,packed_half_storage=True,relaxed_precision=True,Q_pack_inside_gpu_span=True,
    native_integration=False,quality_qualified=False,serving_speed_claim=False,goal_complete=False)
(p/'payload-hashes2.json').write_text(json.dumps(payload_hashes,indent=2)+'\n')
(p/'numeric-audit2.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'passed':result['passed'],'qualified_variants':qualified,'cases':[{'kind':c['kind'],'case':c['case'],'full_bound':c['full_new_policy_bound_fraction_max'],'FP64_bound':c['FP64_new_policy_bound_fraction_max']} for c in results]},indent=2))
