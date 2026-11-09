#!/usr/bin/env python3
"""Independent full-output FP64 audit of the retained synthetic projection data."""
import argparse,hashlib,json,math,statistics
from pathlib import Path
import numpy as np
def need(b,m):
 if not b:raise ValueError(m)
def inputs(m,k):
 # Exact closed-form blocks of the C++ uint32 LCG; no floating recurrence.
 a,c,mask=1664525,1013904223,0xffffffff;power=1;offset=0;p=[];q=[]
 for _ in range(1024):power=(a*power)&mask;offset=(a*offset+c)&mask;p.append(power);q.append(offset)
 p=np.array(p,dtype=np.uint64);q=np.array(q,dtype=np.uint64);result=np.empty(m*k,dtype=np.float32);seed=37
 for first in range(0,m*k,1024):
  count=min(1024,m*k-first);values=(p[:count]*np.uint64(seed)+q[:count])&np.uint64(mask);seed=int(values[-1]);result[first:first+count]=(((values>>8)%65521).astype(np.int32)-32760).astype(np.float32)*np.float32(2**-18)
 return result.reshape(m,k).astype(np.float16).astype(np.float64)
def weights(first,count,k):
 idx=np.arange(first*k//8,(first+count)*k//8,dtype=np.uint64);packed=(idx*np.uint64(2654435761)+np.uint64(131)).astype(np.uint32).reshape(count,k//8)
 codes=((packed[:,:,None]>>np.arange(0,32,4,dtype=np.uint32))&15).reshape(count,k)
 coeff=np.arange(first*k//64,(first+count)*k//64,dtype=np.uint64)
 scale=(np.uint16(0x2c00)+(coeff%8).astype(np.uint16)).view(np.float16).astype(np.float64).reshape(count,k//64)
 bias=(np.uint16(0xb000)+(coeff%7).astype(np.uint16)).view(np.float16).astype(np.float64).reshape(count,k//64)
 decoded=(codes*scale.repeat(64,axis=1)+bias.repeat(64,axis=1)).astype(np.float32).astype(np.float16).astype(np.float64)
 return decoded
def verify(actual,wanted,absolute):
 need(actual.shape==wanted.shape,'wrong output shape');maximum=0.
 for row in range(0,len(actual),64):
  values=actual[row:row+64];need(np.isfinite(values).all(),'missing/nonfinite output')
  ratio=np.abs(values.astype(np.float64)-wanted[row:row+64])/(2e-6+4e-6*absolute[row:row+64]);maximum=max(maximum,float(ratio.max()))
 need(maximum<=1.,'original FP64 bound exceeded');return maximum
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--collection',choices=['operator1','operator2'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();need(not a.output.exists(),'fresh output required');r=a.root
 rows=[128,512] if a.collection=='operator1' else [128,256,512]; case_count=3*len(rows)
 records=[json.loads(s) for s in (r/(a.collection+'.jsonl')).read_text().splitlines()];need(records[0]['kind']=='metadata' and records[-1]['kind']=='closed' and records[-1]['cases']==case_count,'incomplete operator collection')
 cells=[x for x in records if x['kind']=='cell'];need(len(cells)==10*case_count,'wrong cell count');need(records[0]['warmups']==3 and records[0]['reps']==9,'changed sampling');need(records[0].get('reuse64_control_threads')==128,'missing corrected reuse64 dispatch stamp; legacy reuse64 data is not qualified')
 variants=['mpp64','reuse64','reuse128','mps_cached','mps_streamed'];results=[];hashes={};checks=0;rejected=[]
 for n,k in [(14336,4096),(4096,14336),(4096,4096)]:
  x=inputs(512,k);expected=np.empty((512,n),dtype=np.float64);absolute=np.empty_like(expected)
  for first in range(0,n,256):
   count=min(256,n-first);w=weights(first,count,k);expected[:,first:first+count]=x@w.T;absolute[:,first:first+count]=np.abs(x)@np.abs(w).T
  for m in rows:
   group=[z for z in cells if (z['rows'],z['outputs'],z['inner'])==(m,n,k)];need(len(group)==10,'missing shape/condition')
   for swept in [False,True]:
    selected=[z for z in group if z['swept'] is swept];need(sorted(z['variant'] for z in selected)==sorted(variants),'wrong variants')
    for z in selected:
     need(len(z['gpu_ms'])==len(z['wall_ms'])==9 and all(type(t) in (int,float) and math.isfinite(t) and t>0 for t in z['gpu_ms']+z['wall_ms']),'bad timings')
     path=r/a.collection/z['output'];need(path.stat().st_size==m*n*4,'bad payload length');actual=np.memmap(path,dtype='<f4',mode='r',shape=(m,n));fraction=verify(actual,expected[:m],absolute[:m]);checks+=m*n;hashes[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
     results.append(dict(index=z['index'],rows=m,outputs=n,inner=k,swept=swept,variant=z['variant'],full_fp64_fraction=fraction,gpu_median_ms=statistics.median(z['gpu_ms']),wall_median_ms=statistics.median(z['wall_ms']),expanded_weight_bytes=z['expanded_weight_bytes']))
   bad=expected[:1].astype(np.float32);bad[0,0]=np.nan
   try:verify(bad,expected[:1],absolute[:1])
   except ValueError:rejected.append(f'{m}-{n}-{k}-missing-write')
   else:raise ValueError('missing write accepted')
   bad=expected[:1].astype(np.float32);bad[0,0]+=np.float32(10.)
   try:verify(bad,expected[:1],absolute[:1])
   except ValueError:rejected.append(f'{m}-{n}-{k}-wrong-result')
   else:raise ValueError('wrong result accepted')
   print('full_fp64_shape',m,n,k,'PASS',flush=True)
 need(checks==10*sum(rows)*(14336+4096+4096),'wrong physical coverage')
 need(records[-1]['sampled_fp64_checks']==case_count*2*5*12*9,'wrong in-process sampled coverage')
 out=dict(completed=True,collection=a.collection,scope=__doc__,cells=results,checked_retained_words=checks,rejected_corruptions=rejected,raw_files=hashes,no_runtime_or_model_speed_qualification=True);a.output.write_text(json.dumps(out,indent=2)+'\n')
 print('audit PASS',checks,'retained words',flush=True)
if __name__=='__main__':main()
