#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit every saved native streamed projection against independent full FP64 products."""
import argparse, hashlib, json, math, statistics
from pathlib import Path
import numpy as np

def need(ok,message):
    if not ok: raise ValueError(message)

def array(path,dtype,count):
    need(path.stat().st_size==np.dtype(dtype).itemsize*count,'wrong payload length: '+str(path))
    return np.fromfile(path,dtype=dtype)

def verify(actual,expected,absolute,active):
    need(actual.shape==expected.shape and np.isfinite(actual).all(),'missing/nonfinite output')
    mask=np.zeros(len(expected),dtype=bool);mask[active]=True
    need(np.count_nonzero(actual[~mask])==0,'inactive row is nonzero')
    maximum=0.
    for first in range(0,len(expected),32):
        chosen=mask[first:first+32]
        if not chosen.any():continue
        delta=np.abs(actual[first:first+32][chosen].astype(np.float64)-expected[first:first+32][chosen])
        bound=2e-6+4e-6*absolute[first:first+32][chosen]
        maximum=max(maximum,float((delta/bound).max()))
    need(maximum<=1,'original FP64 bound exceeded')
    return maximum

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    need(not a.output.exists(),'fresh audit output required');r=a.directory;report=json.loads((r/'results.json').read_text());need(report['passed'] is True and len(report['cases'])==29,'incomplete native collection')
    geometries=[(512,14336,4096),(512,4096,14336),(512,4096,4096),(512,1024,4096),(256,4096,4096)]
    results=[];checked=0;hashes={};rejected=[]
    for index,(m,n,k) in enumerate(geometries):
        root=r/('case-'+str(index));cells=[x for x in report['cases'] if (x['rows'],x['outputs'],x['inner'])==(m,n,k)]
        need(len(cells)==(7 if m==512 else 1),'wrong native shape coverage')
        input=array(root/'input.f32','<f4',m*k).reshape(m,k).astype(np.float16).astype(np.float64)
        packed=array(root/'weight.u32','<u4',n*k//8).reshape(n,k//8)
        scales=array(root/'scales.f16','<f2',n*k//64).reshape(n,k//64).astype(np.float64)
        biases=array(root/'biases.f16','<f2',n*k//64).reshape(n,k//64).astype(np.float64)
        need(np.isfinite(input).all() and np.isfinite(scales).all() and np.isfinite(biases).all(),'nonfinite reference inputs')
        expected=np.empty((m,n),dtype=np.float64);absolute=np.empty_like(expected)
        for first in range(0,n,256):
            count=min(256,n-first)
            codes=((packed[first:first+count,:,None]>>np.arange(0,32,4,dtype=np.uint32))&15).reshape(count,k)
            # Half coefficients times four-bit integers fit exactly in FP64.
            # Apply explicit FP32 dequant rounding then operand half rounding.
            w=(codes*scales[first:first+count].repeat(64,axis=1)+biases[first:first+count].repeat(64,axis=1)).astype(np.float32).astype(np.float16).astype(np.float64)
            expected[:,first:first+count]=input@w.T;absolute[:,first:first+count]=np.abs(input)@np.abs(w).T
        labels=set()
        for c in cells:
            label=c['label'];need(label not in labels and '/' not in label,'duplicate/unsafe cell label');labels.add(label)
            active=c['active'];need(all(type(x) is int and 0<=x<m for x in active) and len(set(active))==len(active),'invalid active rows')
            if c['dense']:need(active==list(range(m)),'dense map changed')
            else:
                wanted=[(i*127+5)%m for i in range(len(active))]
                if '-reverse' in label:wanted.reverse()
                need(active==wanted,'changed indexed permutation')
            eligible=m==512 and index<3;need(c['eligible']==eligible and c['streamed']==(eligible and len(active)==512),'wrong route witness')
            need(c['plans']==int(eligible),'wrong bounded plan count')
            timed=m==512 and len(active)==512 and '-reverse' not in label
            need(c['warmups']==(3 if timed else 0) and c['repetitions']==(9 if timed else 1),'changed sampling')
            for variant in ['control','candidate']:
                times=c[variant+'_gpu_ms'];need(len(times)==c['repetitions'] and all(type(x) in (float,int) and math.isfinite(x) and x>0 for x in times),'bad timings')
                path=root/(label+'-'+variant+'.f32');actual=array(path,'<f4',m*n).reshape(m,n);fraction=verify(actual,expected,absolute,active);checked+=m*n
                hashes[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
                results.append(dict(index=index,label=label,variant=variant,rows=m,outputs=n,inner=k,active_rows=len(active),full_fp64_fraction=fraction,gpu_median_ms=statistics.median(times),streamed=(variant=='candidate' and c['streamed'])))
        bad=expected[:1].astype(np.float32);bad[0,0]=np.nan
        try:verify(bad,expected[:1],absolute[:1],[0])
        except ValueError:rejected.append(str(index)+'-missing-write')
        else:raise ValueError('missing write accepted')
        bad=expected[:1].astype(np.float32);bad[0,0]+=np.float32(10)
        try:verify(bad,expected[:1],absolute[:1],[0])
        except ValueError:rejected.append(str(index)+'-wrong-result')
        else:raise ValueError('wrong result accepted')
        for name in ['input.f32','weight.u32','scales.f16','biases.f16']:
            path=root/name;hashes[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
        print('full_native_fp64',index,m,n,k,'PASS',flush=True)
    need(checked==170917888,'wrong retained native word coverage')
    a.output.write_text(json.dumps(dict(completed=True,scope=__doc__,checked_retained_words=checked,cells=results,raw_files=hashes,rejected_corruptions=rejected,no_model_or_serving_qualification=True),indent=2)+'\n')
    print('native audit PASS',checked,flush=True)
if __name__=='__main__':main()
