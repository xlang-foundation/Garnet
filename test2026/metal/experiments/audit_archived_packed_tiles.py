"""Audit original archive metadata and complete retained existing-tile outputs.
Static resources and component timings never establish occupancy or serving speed.
"""
from pathlib import Path
import json,hashlib,math,statistics,copy,sys
import numpy as np
r=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent;captures=r.parent/'mistral-gpu-shader-profile1'
metadata=json.loads((r/'metadata2.json').read_text());screen=json.loads((r/'tile-screen1/result.json').read_text());arrays={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def arr(p,dtype):
 key=(str(p),dtype)
 if key not in arrays:arrays[key]=np.memmap(p,dtype=dtype,mode='r')
 return arrays[key]
def check(meta,d,fault=None,fast=False):
 assert meta['passed'] is True and meta['runtime_source_equal_to_native'] is True
 assert meta['register_count_observed'] is False and meta['occupancy_observed'] is False and meta['stall_causes_observed'] is False and meta['instruction_disassembly_available'] is False
 assert len(meta['records'])==6
 shapes=[(4096,14336),(14336,4096),(4096,4096)]
 for i,v in enumerate(meta['records']):
  record=v['record'];k,n=shapes[i//2];tile=128 if i%2==0 else 256
  assert (record['K'],record['N'],record['tile'])==(k,n,tile) and record['name']=='half_m'+str(tile)+'k128'
  assert [record['function_constant_'+str(x)] for x in [10,11,12,13]]==[k,n,k,n]
  assert record['archive_reload_without_fallback'] is True and record['empty_archive_rejected'] is True and record['empty_archive_error_domain']=='AGXMetalG17G' and record['empty_archive_error_code']==3
  assert record['simd_width']==32 and record['maximum_threadgroup_threads']==1024 and record['static_threadgroup_bytes']==17408
  assert v['architectures']==['air64_v28','applegpu_g17g'];p=r/'runtime2'/record['archive'];assert p.stat().st_size==v['archive_bytes'] and sha(p)==v['archive_sha256']
  assert v['reflection']['static_local_allocations']==[dict(size=17408,alignment=2)] and v['reflection']['nodes'][0]['node']['name']==record['name']
  assert v['descriptor']['pipeline']['compute_function']==record['name'] and v['descriptor']['pipeline']['compute_descriptor']['max_total_threads_per_threadgroup']==0
 assert d['completed'] is True and d['math_mode']=='Safe' and d['cooperative_abi_guard_passed'] is True and d['arithmetic_dispatches']==72 and d['logical_output_observations']==276824064
 assert d['all_parent_outputs_and_guards_exact'] is True and d['read_only_inputs_exact'] is True and d['warmups_per_tile']==3 and d['measurements_per_tile']==9 and len(d['cells'])==3
 words=guards=points=comparisons=0;maximum=0.;rows=[]
 for index,(cell,folder,(k,n)) in enumerate(zip(d['cells'],['capture-up2','capture-down1','capture-qo1'],shapes)):
  assert (cell['M'],cell['K'],cell['N'])==(512,k,n) and cell['source']==str(captures/folder) and cell['original_selected_tile']==(128 if index==2 else 256)
  assert len(cell['samples'])==18
  for i,s in enumerate(cell['samples']):assert (s['round'],s['order'],s['tile'])==(i//2,i%2,256 if (i//2+i%2)%2 else 128) and math.isfinite(s['gpu_ms_inclusive_gather']) and s['gpu_ms_inclusive_gather']>0
  ref=arr(captures/folder/'reference-parent.f32','<f4');inp=arr(captures/folder/'input-parent.f32','<f4');assert ref.size==514*n and inp.size==514*k
  assert np.all(inp[:k]==42.5) and np.all(inp[-k:]==-37.25);guards+=2*k
  values={};out={}
  for tile in [128,256]:
   path=r/'tile-screen1'/('M'+str(tile)+'-K'+str(k)+'-N'+str(n)+'-parent.f32');a=arr(path,'<f4');assert a.size==ref.size
   if index==0 and tile==128 and fault in ['changed-output','missing-output','guard-overwrite']:
    a=np.array(a)
    if fault=='changed-output':a.view('<u4')[n+1]^=1
    elif fault=='missing-output':a[n+1]=np.nan
    else:a[0]=0
   assert np.array_equal(a.view('<u4'),ref.view('<u4')) and np.isfinite(a[n:-n]).all()
   assert np.all(a[:n]==42.5) and np.all(a[-n:]==-37.25);guards+=2*n;words+=512*n;out[tile]=a[n:-n].reshape(512,n);values[tile]=[s['gpu_ms_inclusive_gather'] for s in cell['samples'] if s['tile']==tile]
  packed=arr(captures/folder/'weight.u32','<u4');scales=arr(captures/folder/'scales.f16','<f2');biases=arr(captures/folder/'biases.f16','<f2');assert packed.size==n*k//8 and scales.size==biases.size==n*k//64
  x=inp[k:-k].reshape(512,k);j=np.arange(k)
  for row in [0,256,511]:
   for col in [0,1,15,16,31,32,n//2,n-1]:
    code=((packed[col*(k//8)+j//8]>>((j%8)*4))&15).astype(np.float64);weight=(code*scales[col*(k//64)+j//64].astype(np.float64)+biases[col*(k//64)+j//64].astype(np.float64)).astype('<f4').astype('<f2').astype(np.float64);products=x[row].astype('<f2').astype(np.float64)*weight;total=float(products.sum(dtype=np.float64));absolute=float(np.abs(products).sum(dtype=np.float64));points+=1
    for tile in [128,256]:
     fraction=abs(float(out[tile][row,col])-total)/(2e-6+4e-6*absolute);assert fraction<=1;maximum=max(maximum,fraction);comparisons+=1
  paired=[]
  for round in range(9):
   v={s['tile']:s['gpu_ms_inclusive_gather'] for s in cell['samples'] if s['round']==round};paired.append(v[128]/v[256])
  rows.append(dict(M=512,N=n,K=k,original_selected_tile=cell['original_selected_tile'],M128_gpu_ms=statistics.median(values[128]),M256_gpu_ms=statistics.median(values[256]),median_paired_M128_over_M256=statistics.median(paired),ratio_of_medians_M128_over_M256=statistics.median(values[128])/statistics.median(values[256]),paired_range=[min(paired),max(paired)]))
  if fast:break
 return dict(retained_logical_comparisons=words,external_parent_guard_words=guards,independent_FP64_points=points,independent_value_comparisons=comparisons,maximum_original_bound_fraction=maximum,rows=rows)
summary=check(metadata,screen);rejected=[]
for name in ['false_registers','false_archive_reload','false_empty_archive','wrong_constants','wrong_static_memory','wrong_geometry','wrong_sample_order','invalid_time','changed-output','missing-output','guard-overwrite','false_native_observations']:
 meta=copy.deepcopy(metadata);d=copy.deepcopy(screen)
 if name=='false_registers':meta['register_count_observed']=True
 elif name=='false_archive_reload':meta['records'][0]['record']['archive_reload_without_fallback']=False
 elif name=='false_empty_archive':meta['records'][0]['record']['empty_archive_rejected']=False
 elif name=='wrong_constants':meta['records'][0]['record']['function_constant_10']=14336
 elif name=='wrong_static_memory':meta['records'][0]['record']['static_threadgroup_bytes']=9216
 elif name=='wrong_geometry':d['cells'][0]['K']=1024
 elif name=='wrong_sample_order':d['cells'][0]['samples'][0]['order']=1
 elif name=='invalid_time':d['cells'][0]['samples'][0]['gpu_ms_inclusive_gather']=0
 elif name=='false_native_observations':d['logical_output_observations']-=1
 try:check(meta,d,name,True)
 except (AssertionError,KeyError,ValueError):rejected.append(name)
 else:raise AssertionError('corrupt evidence admitted '+name)
pre=json.loads((r/'tile-screen-precollection1.json').read_text());assert all(sha(Path(p))==v for p,v in pre['files_sha256'].items())
assert (r/'tile-screen1.status').read_text().strip()=='0' and (r/'archive2.status').read_text().strip()=='0'
assert "couldn't initialize disassembler" in (r/'native-disassembly2.stderr').read_text() and (r/'native-disassembly2.status').read_text().strip()=='0'
(r/'binary-tile-audit1.json').write_text(json.dumps(dict(passed=True,scope=__doc__,summary=summary,rejected_corruptions=rejected,selected_policy_unchanged=True,no_model_or_serving_speed_claim=True),indent=2)+'\n');print('binary/archive/component independent audit PASS',json.dumps(summary),'corruptions',len(rejected),flush=True)
