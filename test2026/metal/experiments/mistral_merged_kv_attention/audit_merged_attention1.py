"""Independent retained-parent/FP64 audit for merged-KV FP32 attention.
Usage: python3 audit_merged_attention1.py ABSOLUTE_CORRECTNESS_DIRECTORY
No inference or benchmark collection.
"""
from pathlib import Path
import hashlib
import json
import math
import sys
import numpy as np

GUARD=16;POISON=np.uint32(0x7fc12345)

def logical_output(a,batch,chunk,stride,active):
    assert a.size==2*GUARD+batch*chunk*stride
    words=a.view(np.uint32)
    assert np.all(words[:GUARD]==POISON) and np.all(words[-GUARD:]==POISON),'Output guard'
    full=a[GUARD:-GUARD].reshape(batch,chunk,stride)
    assert np.all(full[:,:,4096:].view(np.uint32)==POISON),'Output row padding'
    result=full[:,:,:4096].reshape(batch,chunk,32,128)
    assert np.isfinite(result).all(),'Nonfinite logical output'
    assert np.all(result[~active.astype(bool)]==0),'Inactive output'
    return result


def audit_cell(directory,index,cell):
    batch,chunk,capacity,extent=(cell[n] for n in ['batch','chunk','capacity','extent'])
    p=cell['params'];assert len(p)==16 and p[:6]==[batch,chunk,32,8,128,capacity]
    qs,ks,vs,kcs,vcs,ss,aas,gs,os=p[6:15]
    assert kcs==vcs==8*capacity*128 and qs==gs==os==4100 and ks==vs==1028 and p[15]==int(cell['gated'])
    assert cell['pitches']==[128,capacity,((chunk+3)//4)*4]
    load=lambda name:np.fromfile(directory/f'case{index}-{name}.f32',dtype='<f4')
    inputs=[load(f'input{i}') for i in range(8)]
    lengths=[batch*chunk*qs,batch*chunk*ks,batch*chunk*vs,batch*kcs,batch*vcs,batch*ss,batch*aas,batch*chunk*gs]
    for a,n in zip(inputs,lengths):
        assert a.size==n+2*GUARD
        assert np.all(a.view(np.uint32)[:GUARD]==POISON) and np.all(a.view(np.uint32)[-GUARD:]==POISON)
    assert sum(a.size for a in inputs)==cell['readonly_parent_words']
    q=inputs[0][GUARD:-GUARD].reshape(batch,chunk,qs)[:,:,:4096].reshape(batch,chunk,32,128)
    k=inputs[1][GUARD:-GUARD].reshape(batch,chunk,ks)[:,:,:1024].reshape(batch,chunk,8,128)
    v=inputs[2][GUARD:-GUARD].reshape(batch,chunk,vs)[:,:,:1024].reshape(batch,chunk,8,128)
    kc=inputs[3][GUARD:-GUARD].reshape(batch,8,capacity,128);vc=inputs[4][GUARD:-GUARD].reshape(batch,8,capacity,128)
    slots=inputs[5][GUARD:-GUARD].reshape(batch,ss)[:,:chunk]
    active=inputs[6][GUARD:-GUARD].reshape(batch,aas)[:,:chunk]
    gate=inputs[7][GUARD:-GUARD].reshape(batch,chunk,gs)[:,:,:4096].reshape(batch,chunk,32,128)
    for b in range(batch):
        count,first=cell['counts'][b],cell['first'][b]
        assert first+count<=extent<=capacity
        assert np.array_equal(active[b],np.arange(chunk)<count)
        assert np.array_equal(slots[b],first+np.arange(chunk))
    for i,(stride,width,nrows) in enumerate([(qs,4096,batch*chunk),(ks,1024,batch*chunk),(vs,1024,batch*chunk),(kcs,kcs,batch),(vcs,vcs,batch),(ss,chunk,batch),(aas,chunk,batch),(gs,4096,batch*chunk)]):
        assert np.all(inputs[i][GUARD:-GUARD].reshape(nrows,stride)[:,width:].view(np.uint32)==POISON)
    merge_words=0
    for name,cache,live in [('k',kc,k),('v',vc,v)]:
        actual=load('merged-'+name);expected=np.full(actual.size,POISON,dtype=np.uint32).view(np.float32)
        assert actual.size==batch*8*capacity*128+2*GUARD
        e=expected[GUARD:-GUARD].reshape(batch,8,capacity,128)
        for b in range(batch):
            first,count=cell['first'][b],cell['counts'][b]
            e[b,:,:extent,:]=0
            if count:
                e[b,:,:first,:]=cache[b,:,:first,:]
                e[b,:,first:first+count,:]=live[b,:count,:,:].transpose(1,0,2)
        assert np.array_equal(actual.view(np.uint32),expected.view(np.uint32)),name+' materialization/guard mismatch'
        merge_words+=actual.size
    parents=[load('output'+str(i)) for i in range(3)]
    assert np.array_equal(parents[0].view(np.uint32),parents[1].view(np.uint32)),'Exact softmax control changed'
    outs=[logical_output(a,batch,chunk,os,active) for a in parents]
    diff=np.abs(outs[2].astype(np.float64)-outs[0])
    bound=1e-5+2e-4*np.abs(outs[0].astype(np.float64))
    assert np.all(diff<=bound),'Complete output exceeds fixed control bound'
    oracle_count=0;max_fraction=[0.,0.,0.]
    for b in range(batch):
        count=cell['counts'][b]
        for t in sorted(set([0,count//2,count-1])) if count else []:
            visible=int(slots[b,t])+1
            for h in [0,7,8,31]:
                kh=h//4
                # Derive visible K/V from raw slots and causal latest-writer semantics.
                keys=kc[b,kh,:visible,:].astype(np.float64).copy()
                values=vc[b,kh,:visible,:].astype(np.float64).copy()
                for j in range(t+1):
                    if active[b,j]:
                        position=int(slots[b,j]);keys[position]=k[b,j,kh];values[position]=v[b,j,kh]
                scores=np.sum(keys*q[b,t,h].astype(np.float64),axis=1)/math.sqrt(128)
                probability=np.exp(scores-scores.max());probability/=probability.sum()
                expected=np.sum(probability[:,None]*values,axis=0)
                if cell['gated']:expected/=1+np.exp(-gate[b,t,h].astype(np.float64))
                tolerance=1e-5+2e-4*np.abs(expected)
                for variant,output in enumerate(outs):
                    fraction=np.abs(output[b,t,h].astype(np.float64)-expected)/tolerance
                    assert np.all(fraction<=1),f'FP64 bound failure case{index} variant{variant}'
                    max_fraction[variant]=max(max_fraction[variant],float(fraction.max()))
                oracle_count+=128
    rejected=0
    for kind in ['guard','nonfinite','inactive','truncated']:
        changed=parents[2].copy()
        if kind=='guard':changed.view(np.uint32)[0]=0
        elif kind=='nonfinite':changed[GUARD]=np.nan
        elif kind=='inactive':
            inactive=np.argwhere(active==0)
            if not len(inactive):continue
            b,t=map(int,inactive[0]);changed[GUARD+(b*chunk+t)*os]=1
        else:changed=changed[:-1]
        try:logical_output(changed,batch,chunk,os,active)
        except AssertionError:rejected+=1
        else:raise AssertionError('Corruption accepted: '+kind)
    return dict(case=index,batch=batch,chunk=chunk,capacity=capacity,extent=extent,
        original_and_contiguous_control_exact=True,complete_control_bound_fraction_max=float(np.max(diff/bound)),
        fp64_values_per_variant=oracle_count,fp64_bound_fraction_max=max_fraction,
        complete_merged_parent_words_verified=merge_words,output_parent_words_verified=sum(a.size for a in parents),
        corruption_controls_rejected=rejected,temporary_merged_kv_bytes=cell['temporary_merged_kv_bytes'])


if __name__=='__main__':
    r=Path(sys.argv[1]).resolve();report=json.loads((r/'correctness1.json').read_text())
    assert report['collected'] and len(report['cases'])==7
    cases=[audit_cell(r,i,c) for i,c in enumerate(report['cases'])]
    hashes={}
    for path in sorted(r.glob('*.f32')):
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
        hashes[path.name]=h.hexdigest()
    (r/'payload-hashes1.json').write_text(json.dumps(hashes,indent=2)+'\n')
    result=dict(passed=True,scope=__doc__,cases=cases,full_merged_parent_words_verified=sum(c['complete_merged_parent_words_verified'] for c in cases),
        fp64_values_per_variant=sum(c['fp64_values_per_variant'] for c in cases),
        original_bound='1e-5 + 2e-4*abs(FP64 expected)',
        native_integration=False,serving_speed_claim=False,quality_gate_passed=False,goal_complete=False)
    (r/'audit1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
