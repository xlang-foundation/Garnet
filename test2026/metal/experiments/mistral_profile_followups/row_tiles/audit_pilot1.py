"""Independent saved-parent, RNE dtype, FP64 and bounded geometry audit; no GPU execution."""
from pathlib import Path
import hashlib,json,math,re
import numpy as np
R=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
def need(v,why):
 if not v:raise ValueError(why)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 plan=read(R/'plan1.json');root=R/'pilot1-output';d=read(root/'result1.json');need(d['completed'] and d['pilot_only'] and not d['performance_admitted'] and not d['writer_map_qualified'],'pilot scope');need(read(R/'pilot1-postflight.json')['passed'] and read(R/'pilot1-status.json')['returncode']==0,'terminal process')
 for n,h in read(R/'pilot1-preflight.json')['files_sha256'].items():need(sha(Path(n))==h,'input binding '+n)
 need(len(d['cases'])==12 and len(d['pipeline_resources'])==12 and not d['prepared'],'six methods/two shapes/no expansion')
 variants=(R/'row_tiles1.metal').read_text();base=Path(plan['base_operator_root']);need((R/'combined1.metal').read_text()==(base/'combined3.metal').read_text()+'\n'+variants,'shader composition changed')
 methods=plan['methods'];need([m['name'] for m in methods]==['inline','tensor256','row512n64','row512n32','row256n64t128','row256n32t128'],'method set')
 geometry=[]
 for m in methods:
  threads=m['threads'];simds=threads//32;rm=m['rows_per_group'];cn=m['columns_per_group'];dm=rm//simds
  need(rm in [256,512] and cn in [32,64] and threads in [128,256] and dm in [32,64],'bounded row/thread geometry')
  rows=np.zeros(512,dtype=np.uint8)
  for group in range(512//rm):
   for simd in range(simds):rows[group*rm+simd*dm:group*rm+(simd+1)*dm]+=1
  need(np.all(rows==1),'output SIMD row coverage')
  writes=np.zeros((cn,128),dtype=np.uint8)
  for tid in range(threads):
   for n in range(tid//4,cn,threads//4):writes[n,tid%4*32:(tid%4+1)*32]+=1
  need(np.all(writes==1),'shared tile writer coverage')
  if m['name'] not in ['inline','tensor256']:
   function=variants.split('kernel void '+m['function']+'(',1)[1].split('\n}',1)[0]
   need(f'block.y*{rm}+simd*{dm},column=block.x*{cn}' in function and f'matmul2d_descriptor({dm},{cn},128' in function,'source geometry binding')
   need(f'localN<{cn};localN+={threads//4}' in function and f'decoded[{cn}*STRIDE]' in function,'source weight writer binding')
  geometry.append(dict(method=m['name'],logical_rows=512,rows_per_simd=dm,columns_per_group=cn,simdgroups=simds,host_row_and_shared_writer_coverage=True))
 samples=[];words=0;inputs=0;paths=[]
 shifts=np.arange(8,dtype=np.uint32)*4
 for down in [False,True]:
  shape='down' if down else 'gate';k=14336 if down else 4096;n=4096 if down else 14336;assets=Path(plan['trained_assets'])/'layer-15';captures=Path(plan['trained_captures'])/'layer-15'
  inp=np.memmap(captures/('silu.f32' if down else 'norm-input.f32'),mode='r',dtype='<f4',shape=(512,k));gold=np.memmap(captures/(shape+'.f32'),mode='r',dtype='<u4',shape=(512,n))
  packed=np.memmap(assets/(shape+'-weight.u32'),mode='r',dtype='<u4',shape=(n,k//8));scale=np.memmap(assets/(shape+'-scales.f16'),mode='r',dtype='<f2',shape=(n,k//64));bias=np.memmap(assets/(shape+'-biases.f16'),mode='r',dtype='<f2',shape=(n,k//64));need(np.isfinite(inp).all() and np.isfinite(scale).all() and np.isfinite(bias).all(),'nonfinite fixtures')
  cp=root/(shape+'-compact.u16');compact=np.memmap(cp,mode='r',dtype='<u2');need(compact.size==512*k+64 and np.all(compact[:32]==0x35a5) and np.all(compact[-32:]==0x35a5),'compact input guards/extent');need(np.array_equal(compact[32:-32].reshape(512,k),inp.astype('<f2').view('<u2')),'full input F16 RNE mismatch');inputs+=compact.size;paths.append(cp)
  positions={(0,0),(0,n-1),(511,0),(511,n-1),(256,n//2)}
  for i in range(59):positions.add(((i*193+17)%512,(i*1543+31)%n))
  oracle=[]
  for row,col in sorted(positions):
   codes=((packed[col,:,None]>>shifts)&15).reshape(k);weights=(codes.astype(np.float32)*np.repeat(scale[col].astype(np.float32),64)+np.repeat(bias[col].astype(np.float32),64)).astype(np.float16).astype(np.float64);x=inp[row].astype(np.float16).astype(np.float64);products=x*weights;oracle.append((row,col,float(np.sum(products,dtype=np.float64)),float(np.sum(np.abs(products),dtype=np.float64))))
  cases=[c for c in d['cases'] if c['shape']==shape];need([c['mode'] for c in cases]==[m['name'] for m in methods],'case order/set')
  for c,m in zip(cases,methods):
   need(c['rows']==512 and c['inner']==k and c['columns']==n and c['guards_exact'] and c['all_input_and_quantized_weight_bits_unchanged'],'case input/readonly binding')
   p=root/c['output'];data=np.memmap(p,mode='r',dtype='<u4');need(data.size==512*n+32 and np.all(data[:16]==0x1937abef) and np.all(data[-16:]==0x1937abef) and sha(p)==c['parent_sha256'],'output guards/hash');matrix=data[16:-16].reshape(512,n);need(np.array_equal(matrix,gold) and c['exact_words']==c['total_words']==512*n and c['maximum_absolute_difference']==0,'full output bits mismatch');words+=data.size;paths.append(p)
   resource=next(v for v in d['pipeline_resources'] if v['shape']==shape and v['mode']==m['name']);need(resource['threads']==m['threads'] and resource['shared_bytes']==m['shared_bytes'] and resource['simd_width']==32 and resource['maximum_threads']>=m['threads'],'prepared resource bounds')
   for row,col,total,sumabs in oracle:
    value=float(matrix.view('<f4')[row,col]);fraction=abs(value-total)/(2e-6+4e-6*sumabs);need(math.isfinite(value) and fraction<=1,'unchanged FP64 tolerance');samples.append(dict(shape=shape,mode=c['mode'],row=row,column=col,actual=value,oracle=total,sum_abs=sumabs,fraction=fraction))
 report=dict(passed=True,full_parent_words_compared=words,full_input_half_parent_words=inputs,independent_FP64_values=len(samples),maximum_unchanged_FP64_fraction=max(s['fraction'] for s in samples),all_bits_equal_native_trained_capture=True,host_geometry=geometry,writer_map_gpu_qualified=False,full_weight_expansion=False,model_or_http_admitted=False,performance_admitted=False,no_comparator_execution=True,goal_complete=False,samples=samples,files_sha256={str(p):sha(p) for p in [R/'audit_pilot1.py',R/'plan1.json',R/'pilot1-preflight.json',R/'pilot1-postflight.json',root/'result1.json']+paths})
 p=R/'pilot-audit1.json';need(not p.exists(),'report exists');p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['samples','files_sha256']},indent=2))
if __name__=='__main__':main()
