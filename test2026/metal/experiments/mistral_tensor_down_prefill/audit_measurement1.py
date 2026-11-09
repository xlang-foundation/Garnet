"""CPU audit of serial same-library prefill diagnostics, with order-balanced ON/OFF contrasts."""
from pathlib import Path
import copy,hashlib,json,math,statistics as st
R=Path(__file__).resolve().parent;Q=R.parent/'mistral-causal-batch4-helper1'
read=lambda p:json.loads(p.read_text())
def need(v,why):
 if not v:raise ValueError(why)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check(d,order,inputs,expected):
 need(d['completed'] and d['order']==order and d['inputs']==inputs,'run input/completion binding')
 need(d['library']==str(R/'build-on1/libgarnet.dylib') and 'native_mistral_down_tensor512_optional_v1' in d['backend'],'native binary binding')
 c=d['cleanup'];need(c['metal_allocated_bytes']<=4194304 and all(c[k]==0 for k in ['garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets']),'cleanup leak')
 m=d['prepared']['memory'];need(m['metal_allocated_bytes']<=m['metal_recommended_working_set_bytes']-536870912,'profile memory bound')
 need(len(d['waves'])==12,'wave count');pattern=[False,True,True,False] if order=='forward' else [True,False,False,True];rows={False:[],True:[]};last=0
 for i,w in enumerate(d['waves']):
  need(w['wave']==i and w['enabled']==pattern[i%4] and w['warmup']==(i<4),'balanced order/warmup')
  need(all(type(w[k]) in [int,float] and math.isfinite(w[k]) for k in ['began','ended','elapsed_seconds']),'nonfinite clock')
  need(w['began']>=last and w['ended']>w['began'] and abs(w['ended']-w['began']-w['elapsed_seconds'])<1e-9,'elapsed clock');last=w['ended']
  need([j['label'] for j in w['jobs']]==list('ABCD'),'job coverage')
  need(len({j['row'] for j in w['jobs']})==4 and all(j['epoch']>0 for j in w['jobs']),'lease coverage')
  times=[]
  for j,f in zip(w['jobs'],inputs['jobs']):
   need(j['prompt_ids']==f['prompt_ids'] and j['prefilled']==len(f['prompt_ids']) and j['generated']==expected[j['label']],'tokens/prefill progression')
   t=j['first_prediction_seconds'];need(type(t) in [int,float] and math.isfinite(t) and 0<t<=w['elapsed_seconds'],'first prediction clock');times.append(t)
  selected=0;active_total=0
  for r in w['route_trace']:
   active_total+=r['active_tokens'];rs=r['rows'];need(rs and len(set(rs))==len(rs) and all(0<=v<4 for v in rs),'route owner rows')
   need(r['width'] in [2,4,8,16,32,64,128,256,512] and 0<r['active_tokens']<=len(rs)*r['width'],'actual bounded chunk')
   compact=len(rs) in [1,2] and rs==list(range(rs[0],rs[0]+len(rs)));physical=(len(rs) if compact else 4)*r['width'];count=32 if w['enabled'] and physical==512 and r['active_tokens']==512 else 0
   need(type(r['dispatches']) is int and r['dispatches']==count,'wrong actual kernel route');selected+=bool(count)
  need(active_total==sum(len(f['prompt_ids']) for f in inputs['jobs']),'total actual input work')
  need(selected==16 if w['enabled'] else selected==0,'missing full512 profile work')
  if not w['warmup']:rows[w['enabled']].append(dict(elapsed=w['elapsed_seconds'],first_prediction=st.median(times)))
 return dict(order=order,disabled=dict(native_first_prediction_seconds=st.median(x['first_prediction'] for x in rows[False]),burst_seconds=st.median(x['elapsed'] for x in rows[False])),enabled=dict(native_first_prediction_seconds=st.median(x['first_prediction'] for x in rows[True]),burst_seconds=st.median(x['elapsed'] for x in rows[True])),paired_blocks=[dict(block=b,disabled_over_enabled=st.mean(d['waves'][i]['elapsed_seconds'] for i in range(b,b+4) if not d['waves'][i]['enabled'])/st.mean(d['waves'][i]['elapsed_seconds'] for i in range(b,b+4) if d['waves'][i]['enabled'])) for b in [4,8]])
def main():
 need(read(R/'measurement1-postflight.json')['passed'] and read(R/'qualification-audit1.json')['passed'],'missing terminal/correctness gate');inputs=read(R/'prefill-measure-input1.json');original=read(Q/'productive-input1.json');derived=copy.deepcopy(original)
 for j in derived['jobs']:j['limit']=1
 derived['scope']=inputs['scope'];need(derived==inputs,'fixture changed beyond one output budget')
 expected={j['label']:j['generated'][:1] for j in read(Q/'ordinary-target-result1.json')['jobs']};runs={o:read(R/('measure-'+o+'-result1.json')) for o in ['forward','reverse']};reports=[check(d,o,inputs,expected) for o,d in runs.items()]
 statuses=[read(R/('measure-'+o+'-status1.json')) for o in ['forward','reverse']];need(all(s['status']=='terminal' and s['returncode']==0 for s in statuses) and statuses[0]['finished_ns']<=statuses[1]['started_ns'],'process overlap/failure')
 negative=[]
 def reject(name,mutate):
  d=copy.deepcopy(runs['forward']);mutate(d)
  try:check(d,'forward',inputs,expected)
  except (ValueError,TypeError,KeyError,IndexError):negative.append(name);return
  raise AssertionError('accepted corruption '+name)
 reject('wrong warm flag',lambda d:d['waves'][4].__setitem__('warmup',True))
 reject('wrong order',lambda d:d['waves'][4].__setitem__('enabled',True))
 reject('wrong elapsed duration',lambda d:d['waves'][4].__setitem__('elapsed_seconds',1.0))
 reject('nonfinite first prediction',lambda d:d['waves'][4]['jobs'][0].__setitem__('first_prediction_seconds',float('nan')))
 reject('wrong output token',lambda d:d['waves'][4]['jobs'][0].__setitem__('generated',[0]))
 reject('wrong prefill work',lambda d:d['waves'][4]['jobs'][0].__setitem__('prefilled',0))
 reject('false enabled route',lambda d:d['waves'][5]['route_trace'][0].__setitem__('dispatches',0))
 reject('false disabled route',lambda d:d['waves'][4]['route_trace'][0].__setitem__('dispatches',32))
 reject('wrong library',lambda d:d.__setitem__('library','/tmp/foreign.dylib'))
 reject('native owner leak',lambda d:d['cleanup'].__setitem__('garnet_live_matrix_owners',1))
 names=['audit_measurement1.py','prefill_measure_worker1.py','prefill-measure-input1.json','run_measurement1.py','measurement1-preflight.json','measurement1-postflight.json','qualification-audit1.json']+[f'measure-{o}-{n}1.json' for o in runs for n in ['result','command','status','pid']]
 report=dict(passed=True,scope=__doc__,cohorts=reports,warm_measured_bursts=16,warmup_bursts=8,exact_first_prediction_ids=96,negative_controls_rejected=negative,includes_all_native_prefill_and_first_prediction_work=True,source_or_fixture_hashes_during_timed_rounds=False,no_http_decode_or_comparator_claim=True,no_comparator_execution=True,two_x_target_met=False,goal_complete=False,files_sha256={str(R/n):sha(R/n) for n in names})
 p=R/'measurement-audit1.json';need(not p.exists(),'report exists');p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='files_sha256'},indent=2))
if __name__=='__main__':main()
