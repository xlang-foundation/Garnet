"""Audit balanced operator timings and exact pre/post parent witnesses."""
from pathlib import Path
import copy,hashlib,json,math,statistics
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
METHODS=[('inline',0,False),('inline-rowfast',6,False),('inline-pair',8,False),('shared-tensor',5,False),('shared-rowfast',7,False),('shared-pair',9,False)]
def check(d,reverse,expected_hashes,resource_hashes):
 need(d['completed'] and d['operator_timing_only'] and d['reverse']==reverse and not d['performance_admitted'] and not d['writer_map_qualified'] and d['no_native_integration'],'scope')
 need(len(d['cases'])==24 and len(d['samples'])==216 and len(d['prepared'])==2 and len(d['pipeline_resources'])==20,'sample/witness coverage')
 shapes=['down','gate'] if reverse else ['gate','down'];expected_order=[]
 for shape in shapes:
  for round in range(-6,12):
   for position in range(6):
    index=(position+round+6)%6
    if reverse:index=5-index
    method,mode,expand=METHODS[index];expected_order.append((shape,round,position,index,method,mode,expand,round<0))
 actual=[tuple(x[k] for k in ('shape','round','position','method_index','method','mode','expand','warmup')) for x in d['samples']];need(actual==expected_order,'balanced sample order')
 last=0
 for x in d['samples']:
  start=x['gpu_start'];end=x['gpu_end'];need(all(math.isfinite(x[k]) for k in ('gpu_start','gpu_end','gpu_ms','wall_ms')) and end>start>0 and start>=last,'GPU command ordering/bounds');last=end
  need(abs(x['gpu_ms']-(end-start)*1000)<=2e-9 and x['wall_ms']>0,'raw clock metric')
 for shape in shapes:
  n=14336 if shape=='gate' else 4096
  need([x['method'] for x in d['cases'] if x['shape']==shape and x['phase']=='pre']==[x[0] for x in METHODS] and [x['method'] for x in d['cases'] if x['shape']==shape and x['phase']=='post']==[x[0] for x in METHODS],'pre/post method order')
  for c in [x for x in d['cases'] if x['shape']==shape]:
   method=next(x for x in METHODS if x[0]==c['method']);need(c['mode']==method[1] and c['expand']==method[2] and c['exact_words']==c['total_words']==512*n and c['maximum_absolute_difference']==0 and c['parent_sha256']==expected_hashes[shape] and c['all_parents_unchanged'],'complete pre/post parent witnesses')
  for mode in ('inline','manual','tensor32','tensor64m','tensor64n','shared-tensor','inline-rowfast','shared-rowfast','inline-pair','shared-pair'):
   resource=next(x for x in d['pipeline_resources'] if x['shape']==shape and x['mode']==mode);need(resource==resource_hashes[(shape,mode)],'prepared pipeline resources')
  for method,_,_ in METHODS:
   rows=[x for x in d['samples'] if x['shape']==shape and x['method']==method and not x['warmup']];need(len(rows)==12 and all(sum(x['position']==p for x in rows)==2 for p in range(6)),'balanced position coverage')
 return {(shape,method):statistics.median(x['gpu_ms'] for x in d['samples'] if x['shape']==shape and x['method']==method and not x['warmup']) for shape in shapes for method,_,_ in METHODS}
def main():
 need(read(R/'pilot4-audit.json')['passed'] and read(R/'timing2-postflight.json')['passed'],'prior gates')
 for n,h in read(R/'timing2-preflight.json')['files_sha256'].items():need(sha(Path(n))==h,'provenance '+n)
 plan=read(R/'plan4.json');expected={}
 for shape,n in [('gate',14336),('down',4096)]:
  gold=np.memmap(Path(plan['trained_captures'])/'layer-15'/(shape+'.f32'),mode='r',dtype='<u4',shape=(512*n,));parent=np.full(512*n+32,0x1937abef,dtype='<u4');parent[16:-16]=gold;expected[shape]=hashlib.sha256(parent.tobytes()).hexdigest()
 pilot=read(R/'pilot4-output/result1.json');resources={(x['shape'],x['mode']):x for x in pilot['pipeline_resources']};results=[];raw=[];last=0;files=[]
 for reverse in [False,True]:
  label='timing-'+('reverse' if reverse else 'forward')+'2';path=R/label/'result1.json';d=read(path);status=read(R/(label+'-status.json'));need(status['returncode']==0 and status['status']=='terminal' and status['started_ns']>last,'serial process evidence');last=status['finished_ns'];medians=check(d,reverse,expected,resources);raw.append(d);files.extend([path,R/(label+'-status.json'),R/(label+'-command.json')])
  for shape in ['gate','down']:
   base=medians[(shape,'inline')];rows={method:dict(gpu_median_ms=medians[(shape,method)],native_over_method=base/medians[(shape,method)]) for method,_,_ in METHODS};results.append(dict(cohort='reverse' if reverse else 'forward',shape=shape,rows=rows))
 negative=[]
 def reject(name,fn):
  d=copy.deepcopy(raw[0]);fn(d)
  try:check(d,False,expected,resources)
  except (ValueError,KeyError,IndexError,TypeError,StopIteration):negative.append(name);return
  raise AssertionError('accepted corruption '+name)
 reject('missing sample',lambda d:d['samples'].pop())
 reject('wrong balanced order',lambda d:d['samples'][49].__setitem__('position',(d['samples'][49]['position']+1)%6))
 reject('cold sample admitted',lambda d:d['samples'][0].__setitem__('warmup',False))
 reject('false GPU duration',lambda d:d['samples'][50].__setitem__('gpu_ms',d['samples'][50]['gpu_ms']*2))
 reject('reversed GPU clock',lambda d:d['samples'][50].__setitem__('gpu_end',0))
 reject('nonfinite wall',lambda d:d['samples'][50].__setitem__('wall_ms',float('nan')))
 reject('fabricated extra expansion',lambda d:d['samples'][50].__setitem__('expand',True))
 reject('wrong result parent',lambda d:d['cases'][0].__setitem__('parent_sha256','0'*64))
 reject('prepared input mutated',lambda d:d['cases'][0].__setitem__('all_parents_unchanged',False))
 reject('wrong native baseline resource',lambda d:d['pipeline_resources'][0].__setitem__('shared_bytes',0))
 reject('fabricated profile admission',lambda d:d.__setitem__('no_native_integration',False))
 need(all(row['rows']['shared-tensor']['native_over_method']>1 for row in results),'nonrepeatable shared-operand component result')
 report=dict(passed=True,scope=__doc__,cohorts=results,measured_gpu_commands=288,warmup_gpu_commands=144,pre_post_witnesses=48,input_gather_included=True,full_weight_expansion_used=False,extra_output_scatter_used=False,negative_controls_rejected=negative,decision='Keep original dispatch order. Direct MPP reads from the existing shared Q4 tile show a repeatable modest512-row component improvement; proceed only to bounded native qualification, with no model/HTTP/2x admission.',native_runtime_changed=False,writer_map_qualified=False,model_or_http_admitted=False,quality_admitted=False,goal_complete=False)
 files+=[R/n for n in ['audit_timing3.py','pilot4-audit.json','audit_pilot4.py','plan4.json','timing2-preflight.json','timing2-postflight.json','timing-preparation2.json']];report['files_sha256']={str(p):sha(p) for p in files}
 p=R/'timing-audit3.json';need(not p.exists(),'fresh report');p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('files_sha256','cohorts')},indent=2))
if __name__=='__main__':main()
