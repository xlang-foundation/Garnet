"""Audit balanced whole-MLP timings without admitting model or serving performance."""
from pathlib import Path
import copy,hashlib,importlib.util,json,math,statistics as st
R=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
def need(v,why):
 if not v:raise ValueError(why)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
spec=importlib.util.spec_from_file_location('whole_mlp_pilot_auditor',R/'audit_pilot1.py');pilot_audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot_audit)
def check(d,reverse,pilot):
 need(d['completed'] and d['candidate_exact'] and d['timing_measured'] and d['reverse']==reverse and not d['native_integration'] and not d['serving_speed_claim'] and not d['quality_qualified'],'scope/admission')
 layers=[31,15,0] if reverse else [0,15,31];need([x['layer'] for x in d['layers']]==layers,'layer order');canonical=copy.deepcopy(d);canonical['layers'].sort(key=lambda x:x['layer']);pilot_audit.check(canonical)
 refs={x['layer']:x for x in pilot['layers']}
 for x in d['layers']:need(x==refs[x['layer']],'pre-timing full parent/FP64/writer witnesses differ')
 need(len(d['samples'])==192 and len(d['timing_witnesses'])==24,'sample/witness coverage');order=[]
 for layer in layers:
  for round in range(-4,12):
   for position in range(4):
    method=(position+round+4)%4
    if reverse:method=3-method
    order.append((layer,method,round,position,round<0))
 need([tuple(x[k] for k in ['layer','method','round','position','warmup']) for x in d['samples']]==order,'balanced sample order')
 last=0
 for x in d['samples']:
  need(set(x)=={'layer','method','round','position','warmup','gpu_start','gpu_end','gpu_ms','wall_ms'},'sample cost schema')
  a=x['gpu_start'];b=x['gpu_end'];need(all(math.isfinite(x[k]) for k in ['gpu_start','gpu_end','gpu_ms','wall_ms']) and b>a>0 and a>=last,'GPU clocks/order');last=b
  need(abs(x['gpu_ms']-(b-a)*1000)<=2e-9 and x['wall_ms']>0,'raw metric')
 expected_witnesses=[(layer,method,phase) for layer in layers for phase in ['pre','post'] for method in range(4)]
 need([(w['layer'],w['method'],w['phase']) for w in d['timing_witnesses']]==expected_witnesses,'pre/post witness order')
 for w in d['timing_witnesses']:need(w['half_mismatches']==w['down_mismatches']==0 and w['all_input_and_weight_parents_unchanged'] and w['guards_exact'] and w['no_intermediates_materialized'],'post/pre exact parent/guard/readonly witness')
 for layer in layers:
  for method in range(4):
   samples=[x for x in d['samples'] if x['layer']==layer and x['method']==method and not x['warmup']];need(len(samples)==12 and all(sum(x['position']==p for x in samples)==3 for p in range(4)),'balanced positions')
 return {(layer,method):st.median(x['gpu_ms'] for x in d['samples'] if x['layer']==layer and x['method']==method and not x['warmup']) for layer in layers for method in range(4)}
def main():
 need(read(R/'pilot-audit1.json')['passed'] and read(R/'timing1-postflight.json')['passed'],'qualification/terminal gates')
 for n,h in read(R/'timing1-preflight.json')['files_sha256'].items():need(sha(Path(n))==h,'provenance '+n)
 pilot=read(R/'pilot1-output/result1.json');raw=[];results=[];files=[];last=0
 for reverse in [False,True]:
  order='reverse' if reverse else 'forward';name='timing-'+order+'1';path=R/name/'result1.json';status=read(R/(name+'-status.json'));need(status['status']=='terminal' and status['returncode']==0 and status['started_ns']>last,'serial process proof');last=status['finished_ns'];d=read(path);medians=check(d,reverse,pilot);raw.append(d);files += [path,R/(name+'-command.json'),R/(name+'-status.json')]
  for layer in [0,15,31]:
   results.append(dict(cohort=order,layer=layer,methods_gpu_ms={str(m):medians[(layer,m)] for m in range(4)},legacy_over_both_tensor=medians[(layer,0)]/medians[(layer,2)],new_gate_up_ratio_with_tensor_down=medians[(layer,1)]/medians[(layer,2)],new_gate_up_ratio_with_original_down=medians[(layer,0)]/medians[(layer,3)]))
 negative=[]
 def reject(name,mutate):
  d=copy.deepcopy(raw[0]);mutate(d)
  try:check(d,False,pilot)
  except (ValueError,KeyError,TypeError,IndexError):negative.append(name);return
  raise AssertionError('accepted corruption '+name)
 reject('missing sample',lambda d:d['samples'].pop())
 reject('unbalanced order',lambda d:d['samples'][51].__setitem__('position',(d['samples'][51]['position']+1)%4))
 reject('warmup admitted',lambda d:d['samples'][0].__setitem__('warmup',False))
 reject('false GPU duration',lambda d:d['samples'][50].__setitem__('gpu_ms',2*d['samples'][50]['gpu_ms']))
 reject('nonfinite wall',lambda d:d['samples'][50].__setitem__('wall_ms',float('nan')))
 reject('backward clock',lambda d:d['samples'][50].__setitem__('gpu_end',0))
 reject('invented cost field',lambda d:d['samples'][50].__setitem__('expanded',True))
 reject('missing post witness',lambda d:d['timing_witnesses'].pop())
 reject('post half mismatch',lambda d:d['timing_witnesses'][-1].__setitem__('half_mismatches',1))
 reject('post weight mutation',lambda d:d['timing_witnesses'][-1].__setitem__('all_input_and_weight_parents_unchanged',False))
 reject('post intermediate materialization',lambda d:d['timing_witnesses'][-1].__setitem__('no_intermediates_materialized',False))
 reject('wrong GPU writer map',lambda d:d['layers'][0]['variants'][8].__setitem__('writers_exact',False))
 reject('fabricated serving admission',lambda d:d.__setitem__('serving_speed_claim',True))
 report=dict(passed=True,cohorts=results,measured_gpu_commands=288,warmup_gpu_commands=96,pre_post_parent_witnesses=48,negative_controls_rejected=negative,input_gather_and_both_projection_stages_included=True,no_hash_reset_or_write_in_timed_commands=True,all_outputs_exact_on_trained_captures=True,new_gate_up_above_three_percent_every_cell=all(r['new_gate_up_ratio_with_tensor_down']>1.03 and r['new_gate_up_ratio_with_original_down']>1.03 for r in results),native_model_or_http_admitted=False,no_comparator_execution=True,goal_complete=False)
 files += [R/n for n in ['audit_timing1.py','audit_pilot1.py','pilot-audit1.json','plan1.json','timing-plan1.json','timing1-preflight.json','timing1-postflight.json']];report['files_sha256']={str(p):sha(p) for p in files};p=R/'timing-audit1.json';need(not p.exists(),'fresh report');p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='files_sha256'},indent=2))
if __name__=='__main__':main()
