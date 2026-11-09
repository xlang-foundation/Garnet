"""Replay every output parent and independent FP64 projection samples."""
from pathlib import Path
import copy,hashlib,json,statistics
import numpy as np
R=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def need(ok,why):
 if not ok:raise ValueError(why)
plan=json.loads((R/'plan1.json').read_text());pre=json.loads((R/'preflight1.json').read_text())
for p,h in pre['files_sha256'].items():need(sha(Path(p))==h,'Bound input changed: '+p)
for name,h in json.loads((R/'execution-bindings1.json').read_text()).items():need(sha(R/name)==h,'Execution source changed: '+name)
for name in ['build1-status.json','run1-cohort1-status.json','run1-cohort2-status.json']:need(json.loads((R/name).read_text())['exit_code']==0,'Process failed: '+name)
reports=[json.loads((R/f'run1-cohort{c}/result1.json').read_text()) for c in [1,2]]
def audit(reports,overrides=None,fp64=True):
 overrides=overrides or {};pairs=0;samples=0;metrics=[]
 def read(p):return overrides.get(str(p),np.fromfile(p,dtype='<u4'))
 selections=[list(range(c)) for c in range(17)]+plan['extra_masks']
 for cohort,report in enumerate(reports,1):
  need(report['completed'] and report['cohort']==cohort,'Incomplete cohort')
  need(report['all_parent_bits_exact_ordinary_fp32'] and report['compile_language']=='3.1' and report['fast_math'] is False,'FP32 contract')
  need(report['no_runtime_integration'] and not report['serving_speed_claim'] and not report['goal_complete'],'Scope')
  need(report['close_allocated_bytes']<=report['baseline_allocated_bytes']+4194304,'Retained resources')
  definitions=plan['cases'] if cohort==1 else list(reversed(plan['cases']))
  need([c['name'] for c in report['cases']]==[c['name'] for c in definitions],'Case coverage')
  need(report['methods']==plan['methods'] and report['tile_rows_by_method']==[4,4,8] and report['tile_columns_by_method']==[4,4,2],'Tile/method contract')
  cohortpairs=0
  for definition,case in zip(definitions,report['cases']):
   name=definition['name'];k=definition['k'];n=definition['n'];need((case['k'],case['n'])==(k,n),'Geometry')
   need([c['layout'] for c in case['layouts']]==[0,1],'Layout coverage')
   for layout,cell in enumerate(case['layouts']):
    off=16 if layout else 0;pad=8 if layout else 0;astride=k+pad;ostride=n+pad;stem=name+f'-l{layout}';folder=R/f'run1-cohort{cohort}'
    need((cell['offset_words'],cell['a_stride'],cell['o_stride'],cell['parent_words'])==(off,astride,ostride,off+16*ostride+16),'Memory profile')
    need(cell['weights_readonly'] and cell['input_parent_readonly'],'Readonly')
    source=read(folder/(stem+'-input.u32'));after=read(folder/(stem+'-input-after.u32'));need(np.array_equal(source,after),'Input parent changed')
    reference=read(folder/(stem+'-single-row-reference.u32'));need(hashlib.sha256(reference.tobytes()).hexdigest()==cell['reference_sha256'],'Reference binding')
    need(len(source)==off+16*astride+16 and len(reference)==cell['parent_words'],'Parent bounds')
    source_view=source[off:off+16*astride].reshape(16,astride);refview=reference[off:off+16*ostride].reshape(16,ostride)
    guard=np.uint32(0x7f1ab37d)
    for words,stride,columns in [(source,astride,k),(reference,ostride,n)]:
     need(np.all(words[:off]==guard) and np.all(words[off+16*stride:]==guard),'Parent guard')
     need(np.all(words[off:off+16*stride].reshape(16,stride)[:,columns:]==guard),'Stride guard')
    captured=np.fromfile(definition['input'],dtype='<u4',count=16*k).reshape(16,k)
    need(np.array_equal(source_view[:,:k],captured),'Captured inputs changed')
    need(np.all(np.isfinite(refview[:,:n].copy().view('<f4'))),'Nonfinite reference')
    if fp64 and cohort==1:
     packed=np.memmap(definition['weight'],dtype='<u4',mode='r',shape=(n,k//8));scales=np.memmap(definition['scales'],dtype='<f2',mode='r',shape=(n,k//64));biases=np.memmap(definition['biases'],dtype='<f2',mode='r',shape=(n,k//64));x=source_view[:,:k].copy().view('<f4').astype('float64');y=refview[:,:n].copy().view('<f4')
     for row in [0,7,15]:
      for col in sorted(set([0,min(7,n-1),n//2,n-1])):
       q=((packed[col,:,None]>>np.arange(0,32,4,dtype='uint32'))&15).reshape(-1).astype('float64')
       weight=q*np.repeat(scales[col].astype('float64'),64)+np.repeat(biases[col].astype('float64'),64)
       products=x[row]*weight;expected=np.sum(products,dtype='float64');bound=2e-6+4e-6*np.sum(np.abs(products),dtype='float64')
       need(abs(float(y[row,col])-expected)<=bound,'Independent FP64 projection sample');samples+=1
    need(len(cell['cells'])==len(selections),'Mask coverage')
    for idx,(selection,record) in enumerate(zip(selections,cell['cells'])):
     need(record['id']==idx and record['selection']==selection,'Selected row identity')
     expected=np.full(reference.shape,guard,dtype='<u4');view=expected[off:off+16*ostride].reshape(16,ostride);view[:,:n]=0;view[selection,:n]=refview[selection,:n]
     need(sorted(r['method'] for r in record['outputs'])==[0,1,2],'Method coverage')
     for output in record['outputs']:
      need(output['exact'],'False exactness claim');file=stem+f'-c{idx}-m{output["method"]}.u32';need(output['file']==file,'Wrong output path')
      actual=read(folder/file);need(hashlib.sha256(actual.tobytes()).hexdigest()==output['sha256'],'Output binding');need(np.array_equal(actual,expected),'Ordinary FP32 whole-parent mismatch');pairs+=len(expected);cohortpairs+=len(expected)
    timings=cell['timing'];need(len(timings)==(153 if layout==0 else 0),'Timing coverage')
    if layout==0:
     by={}
     for t in timings:
      count=t['count'];method=t['method'];round=t['round'];order=t['order'];need(count in [4,8,16] and method in [0,1,2] and 0<=round<17 and order in [0,1,2],'Timing key')
      key=count,method,round;need(key not in by,'Duplicate timing');by[key]=t
      want=(order+(round+4)%3)%3;want=want if cohort==1 else 2-want;need(method==want,'Timing order')
      need(t['host_start_ns']<t['host_end_ns'] and 0<t['gpu_start_seconds']<t['gpu_end_seconds'],'Invalid clocks')
      need(abs(t['gpu_ms']-(t['gpu_end_seconds']-t['gpu_start_seconds'])*1000)<1e-9,'False GPU duration')
     for count in [4,8,16]:
      med={method:statistics.median(by[count,method,r]['gpu_ms'] for r in range(17)) for method in [0,1,2]}
      for method in [1,2]:metrics.append(dict(cohort=cohort,name=name,selected_rows=count,method=['control','current4x4','candidate8x2'][method],control_gpu_ms=med[0],candidate_gpu_ms=med[method],median_paired_speed_ratio=statistics.median(by[count,0,r]['gpu_ms']/by[count,method,r]['gpu_ms'] for r in range(17)),median_paired_ratio_vs_current4x4=statistics.median(by[count,1,r]['gpu_ms']/by[count,method,r]['gpu_ms'] for r in range(17))))
   need(cohortpairs<=report['parent_word_pairs'],'Word count bounds')
  need(cohortpairs==report['parent_word_pairs'],'Word count')
 return dict(passed=True,cohorts=2,cases=len(plan['cases']),layouts_per_case=2,masks_per_layout=len(selections),methods=3,parent_word_pairs=pairs,independent_fp64_samples=samples,all_parent_bits_exact_ordinary_fp32=True,metrics=metrics,no_runtime_integration=True,serving_speed_claim=False,goal_complete=False)
result=audit(reports)
negative=[]
def reject(name,mutate,override=None):
 bad=copy.deepcopy(reports);mutate(bad)
 try:audit(bad,override,False)
 except (ValueError,KeyError,TypeError,IndexError):negative.append(name);return
 raise AssertionError('Corruption accepted: '+name)
reject('wrong_tile_contract',lambda r:r[0].update(tile_rows_by_method=[4,4,4]))
reject('false_fp32_contract',lambda r:r[0].update(fast_math=True))
reject('missing_cohort_case',lambda r:r[1]['cases'].pop())
reject('wrong_selected_row',lambda r:r[0]['cases'][0]['layouts'][0]['cells'][1].update(selection=[15]))
reject('missing_tail_count',lambda r:r[0]['cases'][0]['layouts'][0]['cells'].pop(15))
reject('false_exact',lambda r:r[0]['cases'][0]['layouts'][0]['cells'][0]['outputs'][0].update(exact=False))
reject('wrong_reference_binding',lambda r:r[0]['cases'][0]['layouts'][0].update(reference_sha256='0'*64))
reject('false_readonly',lambda r:r[0]['cases'][0]['layouts'][0].update(weights_readonly=False))
reject('false_gpu_time',lambda r:r[0]['cases'][0]['layouts'][0]['timing'][0].update(gpu_ms=999))
reject('duplicate_timing',lambda r:r[0]['cases'][0]['layouts'][0]['timing'].__setitem__(1,copy.deepcopy(r[0]['cases'][0]['layouts'][0]['timing'][0])))
p=R/'run1-cohort1/l0-q-l0-c0-m0.u32';words=np.fromfile(p,dtype='<u4');words[-1]^=np.uint32(1)
reject('raw_parent_guard_bit',lambda r:None,{str(p):words})
result['negative_controls_rejected']=negative
result['files_sha256']={str(p):sha(p) for p in [R/'plan1.json',R/'preflight1.json',R/'screen1.mm',R/'screen1',R/'prepare1.py']+[R/f'run1-cohort{c}/result1.json' for c in [1,2]]}
(R/'audit1.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['files_sha256','metrics']}))
for count in [4,8,16]:
 for method in ['current4x4','candidate8x2']:
  values=[m['median_paired_speed_ratio'] for m in result['metrics'] if m['selected_rows']==count and m['method']==method]
  print(count,method,'median_across_geometry_cohort_ratios',statistics.median(values),'range',min(values),max(values))
