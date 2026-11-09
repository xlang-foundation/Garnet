"""Audit balanced learned-input row-tile operator timings and exact parent witnesses."""
from pathlib import Path
import copy,hashlib,json,math,statistics as st
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
METHODS=['inline','tensor256','row512n64','row512n32','row256n64t128','row256n32t128']
def check(d,reverse,expected,resources):
 need(d['completed'] and d['operator_timing_only'] and d['reverse']==reverse and not d['performance_admitted'] and not d['writer_map_qualified'] and d['no_native_integration'],'scope')
 need(len(d['cases'])==24 and len(d['samples'])==216 and not d['prepared'] and len(d['pipeline_resources'])==12,'coverage/no expanded weights')
 shapes=['down','gate'] if reverse else ['gate','down'];order=[]
 for shape in shapes:
  for round in range(-6,12):
   for position in range(6):
    index=(position+round+6)%6
    if reverse:index=5-index
    order.append((shape,round,position,index,METHODS[index],round<0))
 need([tuple(x[k] for k in ['shape','round','position','method_index','mode','warmup']) for x in d['samples']]==order,'balanced order')
 last=0
 for x in d['samples']:
  need(set(x)=={'shape','round','position','method_index','mode','warmup','gpu_start','gpu_end','gpu_ms','wall_ms'},'sample schema/cost fields')
  a=x['gpu_start'];b=x['gpu_end'];need(all(math.isfinite(x[k]) for k in ['gpu_start','gpu_end','gpu_ms','wall_ms']) and b>a>0 and a>=last,'GPU clocks/order');last=b
  need(abs(x['gpu_ms']-(b-a)*1000)<=2e-9 and x['wall_ms']>0,'raw metric')
 for shape in shapes:
  n=14336 if shape=='gate' else 4096
  for phase in ['pre','post']:need([c['mode'] for c in d['cases'] if c['shape']==shape and c['phase']==phase]==METHODS,'parent witness order')
  for c in [v for v in d['cases'] if v['shape']==shape]:need(c['exact_words']==c['total_words']==512*n and c['maximum_absolute_difference']==0 and c['parent_sha256']==expected[shape] and c['all_parents_unchanged'],'full parent witness')
  for mode in METHODS:
   resource=next(v for v in d['pipeline_resources'] if v['shape']==shape and v['mode']==mode);need(resource==resources[(shape,mode)],'pipeline resource proof')
   samples=[x for x in d['samples'] if x['shape']==shape and x['mode']==mode and not x['warmup']];need(len(samples)==12 and all(sum(x['position']==p for x in samples)==2 for p in range(6)),'balanced positions')
 return {(shape,mode):st.median(x['gpu_ms'] for x in d['samples'] if x['shape']==shape and x['mode']==mode and not x['warmup']) for shape in shapes for mode in METHODS}
def main():
 need(read(R/'pilot-audit1.json')['passed'] and read(R/'timing1-postflight.json')['passed'],'prior gates')
 for n,h in read(R/'timing1-preflight.json')['files_sha256'].items():need(sha(Path(n))==h,'provenance '+n)
 plan=read(R/'plan1.json');expected={}
 for shape,n in [('gate',14336),('down',4096)]:
  gold=np.memmap(Path(plan['trained_captures'])/'layer-15'/(shape+'.f32'),mode='r',dtype='<u4',shape=(512*n,));parent=np.full(512*n+32,0x1937abef,dtype='<u4');parent[16:-16]=gold;expected[shape]=hashlib.sha256(parent.tobytes()).hexdigest()
 pilot=read(R/'pilot1-output/result1.json');resources={(r['shape'],r['mode']):r for r in pilot['pipeline_resources']};results=[];raw=[];last=0;files=[]
 for reverse in [False,True]:
  order='reverse' if reverse else 'forward';name='timing-'+order+'1';path=R/name/'result1.json';d=read(path);status=read(R/(name+'-status.json'));need(status['returncode']==0 and status['status']=='terminal' and status['started_ns']>last,'serial terminal processes');last=status['finished_ns'];medians=check(d,reverse,expected,resources);raw.append(d);files += [path,R/(name+'-status.json'),R/(name+'-command.json')]
  for shape in ['gate','down']:
   rows={mode:dict(gpu_median_ms=medians[(shape,mode)],native_over_method=medians[(shape,'inline')]/medians[(shape,mode)],tensor256_over_method=medians[(shape,'tensor256')]/medians[(shape,mode)]) for mode in METHODS};results.append(dict(cohort=order,shape=shape,rows=rows))
 negative=[]
 def reject(name,mutate):
  d=copy.deepcopy(raw[0]);mutate(d)
  try:check(d,False,expected,resources)
  except (ValueError,KeyError,TypeError,IndexError,StopIteration):negative.append(name);return
  raise AssertionError('accepted corruption '+name)
 reject('missing sample',lambda d:d['samples'].pop())
 reject('unbalanced order',lambda d:d['samples'][49].__setitem__('position',(d['samples'][49]['position']+1)%6))
 reject('warmup admitted',lambda d:d['samples'][0].__setitem__('warmup',False))
 reject('false GPU duration',lambda d:d['samples'][50].__setitem__('gpu_ms',2*d['samples'][50]['gpu_ms']))
 reject('reversed clock',lambda d:d['samples'][50].__setitem__('gpu_end',0))
 reject('nonfinite wall',lambda d:d['samples'][50].__setitem__('wall_ms',float('nan')))
 reject('fabricated expansion',lambda d:d['samples'][50].__setitem__('expand',True))
 reject('wrong output parent',lambda d:d['cases'][0].__setitem__('parent_sha256','0'*64))
 reject('changed input parent',lambda d:d['cases'][0].__setitem__('all_parents_unchanged',False))
 reject('wrong pipeline shared bytes',lambda d:d['pipeline_resources'][0].__setitem__('shared_bytes',0))
 reject('fabricated model admission',lambda d:d.__setitem__('no_native_integration',False))
 candidates={shape:[mode for mode in METHODS[2:] if all(c['rows'][mode]['tensor256_over_method']>1.03 for c in results if c['shape']==shape)] for shape in ['gate','down']}
 report=dict(passed=True,cohorts=results,candidates_above_three_percent_over_qualified_tensor256=candidates,measured_gpu_commands=288,warmup_gpu_commands=144,pre_post_parent_witnesses=48,negative_controls_rejected=negative,input_gather_included=True,full_weight_expansion=False,extra_scatter=False,native_model_or_http_admitted=False,writer_map_gpu_qualified=False,no_comparator_execution=True,goal_complete=False)
 files += [R/n for n in ['audit_timing1.py','pilot-audit1.json','plan1.json','timing1-preflight.json','timing1-postflight.json','timing-preparation1.json']];report['files_sha256']={str(p):sha(p) for p in files}
 p=R/'timing-audit1.json';need(not p.exists(),'fresh report');p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='files_sha256'},indent=2))
if __name__=='__main__':main()
