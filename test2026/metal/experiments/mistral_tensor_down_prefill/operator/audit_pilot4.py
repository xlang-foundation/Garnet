"""Independent saved-data dtype/FP64 audit; no GPU execution or timing admission."""
from pathlib import Path
import hashlib,json,math
import numpy as np
R=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
def need(ok,why):
 if not ok:raise ValueError(why)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 plan=read(R/'plan4.json');root=R/'pilot4-output';d=read(root/'result1.json');need(d['completed'] and d['pilot_only'] and not d['performance_admitted'] and not d['writer_map_qualified'],'pilot scope')
 pre=read(R/'pilot4-preflight.json');need(read(R/'pilot4-postflight.json')['passed'] and read(R/'pilot4-status.json')['returncode']==0,'terminal collection')
 for n,h in pre['files_sha256'].items():need(sha(Path(n))==h,'input binding '+n)
 need(len(d['cases'])==20 and len(d['prepared'])==2 and len(d['pipeline_resources'])==20,'case coverage')
 words=0;half_words=0;samples=[];output_paths=[]
 for down in [False,True]:
  shape='down' if down else 'gate';k=14336 if down else 4096;n=4096 if down else 14336;assets=Path(plan['trained_assets'])/'layer-15';captures=Path(plan['trained_captures'])/'layer-15'
  inp=np.memmap(captures/('silu.f32' if down else 'norm-input.f32'),mode='r',dtype='<f4',shape=(512,k));gold=np.memmap(captures/(shape+'.f32'),mode='r',dtype='<u4',shape=(512,n))
  packed=np.memmap(assets/(shape+'-weight.u32'),mode='r',dtype='<u4',shape=(n,k//8));scale=np.memmap(assets/(shape+'-scales.f16'),mode='r',dtype='<f2',shape=(n,k//64));bias=np.memmap(assets/(shape+'-biases.f16'),mode='r',dtype='<f2',shape=(n,k//64))
  need(np.isfinite(inp).all() and np.isfinite(scale).all() and np.isfinite(bias).all(),'nonfinite fixture')
  prep=next(x for x in d['prepared'] if x['shape']==shape);path=root/prep['payload'];parent=np.memmap(path,mode='r',dtype='<u2');need(parent.size==n*k+64 and np.all(parent[:32]==0x35a5) and np.all(parent[-32:]==0x35a5) and sha(path)==prep['parent_sha256'],'prepared extent/guards')
  actual=parent[32:-32].reshape(n,k);shifts=np.arange(8,dtype=np.uint32)*4
  for first in range(0,n,64):
   count=min(64,n-first);codes=((packed[first:first+count,:,None]>>shifts)&15).reshape(count,k)
   # F16 coefficient times a 4-bit integer is exact in F32, so this product
   # followed by one F32 sum equals the shader's explicit fused dequantization.
   expected=(codes.astype(np.float32)*np.repeat(scale[first:first+count].astype(np.float32),64,axis=1)+np.repeat(bias[first:first+count].astype(np.float32),64,axis=1)).astype('<f2').view('<u2')
   need(np.array_equal(expected,actual[first:first+count]),'prepared F16 bit mismatch');half_words+=expected.size
  positions={(0,0),(0,n-1),(511,0),(511,n-1),(256,n//2)}
  for i in range(59):positions.add(((i*193+17)%512,(i*1543+31)%n))
  oracle=[]
  for row,col in sorted(positions):
   codes=((packed[col,:,None]>>shifts)&15).reshape(k);weights=(codes.astype(np.float32)*np.repeat(scale[col].astype(np.float32),64)+np.repeat(bias[col].astype(np.float32),64)).astype(np.float16).astype(np.float64)
   x=inp[row].astype(np.float16).astype(np.float64);products=x*weights;total=float(np.sum(products,dtype=np.float64));sumabs=float(np.sum(np.abs(products),dtype=np.float64));oracle.append((row,col,total,sumabs))
  cases=[x for x in d['cases'] if x['shape']==shape];need({x['mode'] for x in cases}=={'inline','manual','tensor32','tensor64m','tensor64n','shared-tensor','inline-rowfast','shared-rowfast','inline-pair','shared-pair'},'mode coverage')
  for case in cases:
   need(case['rows']==512 and case['inner']==k and case['columns']==n and case['guards_exact'] and case['all_input_and_prepared_weight_bits_unchanged'],'case binding')
   path=root/case['output'];p=np.memmap(path,mode='r',dtype='<u4');need(p.size==512*n+32 and np.all(p[:16]==0x1937abef) and np.all(p[-16:]==0x1937abef) and sha(path)==case['parent_sha256'],'output parent guards/hash')
   matrix=p[16:-16].reshape(512,n);need(np.array_equal(matrix,gold) and case['exact_words']==case['total_words']==512*n and case['maximum_absolute_difference']==0,'complete output equality');words+=p.size;output_paths.append(path)
   for row,col,total,sumabs in oracle:
    value=float(matrix.view('<f4')[row,col]);fraction=abs(value-total)/(2e-6+4e-6*sumabs);need(math.isfinite(value) and fraction<=1,'independent unchanged FP64 bound');samples.append(dict(shape=shape,mode=case['mode'],row=row,column=col,actual=value,oracle=total,sum_abs=sumabs,fraction=fraction))
   resource=next(x for x in d['pipeline_resources'] if x['shape']==shape and x['mode']==case['mode']);need(resource['simd_width']==32 and resource['maximum_threads']>=resource['threads'] and resource['shared_bytes']==(0 if case['mode'] in ('tensor32','tensor64m','tensor64n') else 17408),'resource profile')
 report=dict(passed=True,scope=__doc__,full_parent_words_compared=words,full_prepared_half_words_compared=half_words,independent_FP64_values=len(samples),maximum_unchanged_FP64_fraction=max(x['fraction'] for x in samples),all_candidate_bits_equal_native_trained_capture=True,guards_and_readonly_parents_passed=True,rows_qualified_only=512,writer_map_qualified=False,profile_native_integration=False,performance_admitted=False,no_comparator_execution=True,goal_complete=False,samples=samples,files_sha256={str(p):sha(p) for p in [R/'audit_pilot4.py',R/'plan4.json',R/'pilot4-preflight.json',R/'pilot4-postflight.json',root/'result1.json']+output_paths+[root/p['payload'] for p in d['prepared']]})
 p=R/'pilot4-audit.json';need(not p.exists(),'fresh report destination');p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('samples','files_sha256')},indent=2))
if __name__=='__main__':main()
