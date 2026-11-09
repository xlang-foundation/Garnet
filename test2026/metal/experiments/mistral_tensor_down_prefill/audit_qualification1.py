"""Audit full model tokens/state, native route selection, and row lifecycle against frozen controls."""
from pathlib import Path
import copy,hashlib,importlib.util,json,re
R=Path(__file__).resolve().parent;Q=R.parent/'mistral-causal-batch4-helper1';N=R.parent/'mistral-gqa16-value-native1'
read=lambda p:json.loads(p.read_text())
def need(v,why):
 if not v:raise ValueError(why)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('prior_target_auditor',Q/'audit_productive1.py');prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
def cleanup(d):
 need(d['completed'],'incomplete run');c=d['cleanup'];need(c['metal_allocated_bytes']<=4194304 and all(c[k]==0 for k in ['garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets']),'native cleanup')
 m=d['prepared']['memory'];need(m['metal_allocated_bytes']<=m['metal_recommended_working_set_bytes']-536870912,'profile memory bound')
def routes(d,enabled):
 prefills=[p for t in d['ticks'] for p in t['plan']['batches'] if p['stage']=='prefill'];need(len(prefills)==len(d['route_trace']),'route coverage');selected=0;fallback=0;compact=0
 for p,r in zip(prefills,d['route_trace']):
  rows=[q['row'] for q in p['requests']];width=p.get('prefill_chunk',128);active=sum(q['prefill_tokens'] for q in p['requests'])
  need(r['rows']==rows and r['width']==width and r['active_tokens']==active,'route plan binding')
  contiguous=len(rows) in (1,2) and rows==list(range(rows[0],rows[0]+len(rows)))
  physical=(len(rows) if contiguous else 4)*width
  expected=32 if enabled and active==512 and physical==512 else 0
  need(type(r['dispatches']) is int and r['dispatches']==expected,'wrong tensor down route');selected+=int(expected>0);fallback+=int(expected==0);compact+=int(expected>0 and len(rows)<4)
 need(not enabled or selected>0,'new route never executed')
 return dict(prefill_calls=len(prefills),selected=selected,fallback=fallback,compact_selected=compact)
def ordinary(d,reference,case):
 need(d['configuration']==case['configuration'] and d['mode']=='ordinary' and d['inputs']==reference['inputs'],'wrong model arm/inputs');cleanup(d)
 need(d['library']==str(R/('build-'+case['build']+'1/libgarnet.dylib')),'wrong native library')
 need(('native_mistral_down_tensor512_optional_v1' in d['backend'])==(case['build']=='on'),'compiled extension tag')
 normal=copy.deepcopy(d);normal['library']=str(Q/'build-on1/libgarnet.dylib')
 result=prior.target_run(normal,d['inputs'],'ordinary');ref=prior.target_run(reference,reference['inputs'],'ordinary')
 need(result==ref,'target tokens/KV/hidden progression mismatch');need(d['ticks']==reference['ticks'] and d['snapshots']==reference['snapshots'] and d['jobs']==reference['jobs'],'full ordinary trace mismatch')
 return dict(exact_output_ids=sum(len(j['generated']) for j in d['jobs']),full_four_row_digest_pairs=len(d['snapshots'])*260,routes=routes(d,case['configuration']=='enabled'))
def lifecycle(d,reference):
 cleanup(d);need(d['configuration']=='enabled' and d['mode']=='ordinary' and d['inputs']==reference['inputs'],'wrong lifecycle fixture')
 need(d['library']==str(R/'build-on1/libgarnet.dylib') and 'native_mistral_down_tensor512_optional_v1' in d['backend'],'lifecycle library')
 for k in ['jobs','ticks','witnesses','rejections']:need(d[k]==reference[k],'lifecycle mismatch '+k)
 jobs={j['label']:j for j in d['jobs']};need(set(jobs)==set('ABCDEFG'),'lifecycle owner coverage')
 need(jobs['A']['row']==jobs['E']['row'] and jobs['E']['epoch']>jobs['A']['epoch'] and jobs['C']['row']==jobs['F']['row'] and jobs['F']['epoch']>jobs['C']['epoch'],'row reuse/epoch')
 need(jobs['C']['status']=='cancelled' and all(j['status']=='finished' for k,j in jobs.items() if k!='C'),'finish/cancel status')
 need([w['label'] for w in d['witnesses']]==['initial_prefilled','finish_A','claim_E','mid_E','cancel_C','claim_F','finish_E','finish_B','before_late_G','all_done'],'lifecycle witness coverage')
 need(all(d['witnesses'][-1]['status'][k]==0 for k in ['active','queued','tracked','inflight']),'scheduler leaked owners')
 r=routes(d,True);need(r['compact_selected']>0 and r['fallback']>0,'compact and fallback route coverage')
 return dict(exact_output_ids=sum(len(j['generated']) for j in d['jobs']),full_four_row_digest_pairs=len(d['witnesses'])*260,routes=r)
def main():
 need(read(R/'qualification1-postflight.json')['passed'],'no terminal collection');plan=read(R/'qualification-plan1.json')['cases'];reference=read(Q/'ordinary-target-result1.json');life_reference=read(N/'c4-ordinary-result1.json');reports={};data={}
 for case in plan:
  d=read(R/(case['name']+'-result1.json'));data[case['name']]=d;need(read(R/(case['name']+'-status1.json'))['returncode']==0,'process failed')
  reports[case['name']]=ordinary(d,reference,case) if case['worker']=='ordinary' else lifecycle(d,life_reference)
 negative=[]
 def reject(name,fn,base,mutate):
  d=copy.deepcopy(base);mutate(d)
  try:fn(d)
  except (ValueError,KeyError,TypeError,IndexError,AssertionError):negative.append(name);return
  raise AssertionError('accepted corruption '+name)
 base=data['on-enabled'];fn=lambda d:ordinary(d,reference,plan[0])
 reject('published output',fn,base,lambda d:d['jobs'][0]['generated'].__setitem__(0,1))
 reject('KV state',fn,base,lambda d:d['snapshots'][0]['states'][0][0].__setitem__(0,'0'*64))
 reject('hidden state',fn,base,lambda d:d['snapshots'][0]['hidden'].__setitem__(0,'0'*64))
 reject('missing route',fn,base,lambda d:d['route_trace'].pop())
 reject('false selected route',fn,base,lambda d:d['route_trace'][0].__setitem__('dispatches',0))
 reject('wrong active count',fn,base,lambda d:d['route_trace'][0].__setitem__('active_tokens',511))
 reject('wrong library',fn,base,lambda d:d.__setitem__('library','/tmp/foreign.dylib'))
 reject('native memory leak',fn,base,lambda d:d['cleanup'].__setitem__('garnet_live_matrix_owners',1))
 fn2=lambda d:lifecycle(d,life_reference);life=data['lifecycle-on']
 reject('reused owner epoch',fn2,life,lambda d:next(j for j in d['jobs'] if j['label']=='E').__setitem__('epoch',1))
 reject('cancel state',fn2,life,lambda d:next(j for j in d['jobs'] if j['label']=='C').__setitem__('status','finished'))
 reject('late owner state',fn2,life,lambda d:d['witnesses'][-1]['states'][-1][0].__setitem__(0,'0'*64))
 names=['qualification-plan1.json','ordinary_worker1.py','lifecycle_worker1.py','qualification1-preflight.json','qualification1-postflight.json','source-bindings1.json','build-audit1.json','audit_qualification1.py']+[c['name']+'-result1.json' for c in plan]
 report=dict(passed=True,cases=reports,negative_controls_rejected=negative,reference_outputs_read_during_generation=False,performance_claim=False,http_qualified=False,quality_suite_qualified=False,xmodel_operator_changes=False,no_comparator_execution=True,goal_complete=False,files_sha256={str(R/n):sha(R/n) for n in names},frozen_references_sha256={str(Q/'ordinary-target-result1.json'):sha(Q/'ordinary-target-result1.json'),str(Q/'audit_productive1.py'):sha(Q/'audit_productive1.py'),str(N/'c4-ordinary-result1.json'):sha(N/'c4-ordinary-result1.json')})
 out=R/'qualification-audit1.json';need(not out.exists(),'report exists');out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if not k.endswith('sha256')},indent=2))
if __name__=='__main__':main()
