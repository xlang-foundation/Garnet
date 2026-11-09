"""Audit untouched model regression and actual trained attention tensors, no GPU work."""
from pathlib import Path
import copy,hashlib,json,math,re,sys
import numpy as np
r=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
load=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def events(p):return [json.loads(l) for l in p.read_text().splitlines() if l.startswith('{')]
b=load(r/'bindings2.json');fixture=load(r/'fixture1.json');status=load(r/'model-state-status2.json')
assert status['status']=='collected_pending_audit' and len(status['waves'])==1
assert status['waves'][0]['mode']=='on' and status['waves'][0]['returncode']==0 and status['waves'][0]['last_event']=='closed'
for name,key in [('worker-state2.py','worker_sha256'),('source-audit2.json','source_audit_sha256'),('fixture1.json','fixture_sha256')]:assert sha(r/name)==b[key]
for name,h in load(r/'source-audit2.json')['candidate_owned_files_sha256'].items():assert sha(Path(b['source'])/name)==h,name
for mode,tests in [('on',51),('off',50)]:
 assert sha(Path(b['libraries'][mode]['path'])/'libgarnet.dylib')==b['libraries'][mode]['sha256']
 assert load(r/f'ctest-{mode}2-status.json')['returncode']==0 and f'100% tests passed out of {tests}' in (r/f'ctest-{mode}2.stdout').read_text()
assert sha(r/'profile-on2.json')==b['libraries']['on']['profile_sha256']
prior=Path(b['prior_state_study']);assert sha(prior/'provenance1.json')==b['prior_provenance_sha256'] and sha(prior/'model-on1.stdout')==b['prior_state_stdout_sha256']
assert sha(Path(b['original_control']))==b['original_control_sha256']
original=load(Path(b['original_control']));old=events(prior/'model-on1.stdout');es=events(r/'model-on2.stdout');meta=load(r/'capture2/capture.json')
keys=['garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets']
def verify_model(es):
 assert es and not [e for e in es if e['event']=='error'] and len([e for e in es if e['event']=='ready'])==1
 complete=[e for e in es if e['event']=='completed'];assert len(complete)==4
 for f in fixture['requests']:
  key='trained-'+str(f['row']);e=next(e for e in complete if e['id']==key)
  assert e['prompt_ids']==f['token_ids']==original[key]['prompt_ids'] and e['generated_ids']==original[key]['generated_ids'] and len(e['generated_ids'])==8
  assert e['text']==original[key]['text'] and e['cached_tokens']==0 and e['finish_reason']=='length'
 route=[e for e in es if e['event']=='diagnostic_native_kv_hoist'];assert route==[e for e in old if e['event']=='diagnostic_native_kv_hoist']
 st=[e for e in es if e['event']=='diagnostic_full_state'];assert st==[e for e in old if e['event']=='diagnostic_full_state']
 assert [e['invocation'] for e in st]==[1,16,18,25] and sum(e['dispatches'] for e in route)==1088
 arm=[e for e in es if e['event']=='diagnostic_capture_armed'];done=[e for e in es if e['event']=='diagnostic_trained_attention_fixture'];assert len(arm)==len(done)==1
 assert arm[0]['invocation']==done[0]['invocation']==16 and arm[0]['directory']==done[0]['directory']==str(r/'capture2') and done[0]['metadata']==meta
 for k in keys:assert arm[0]['before_memory'][k]==arm[0]['armed_memory'][k]==done[0]['after_memory'][k]
 assert len([e for e in es if e['event']=='closed'])==1 and es[-1]['event']=='closed'
 for k in keys:assert es[-1]['memory'][k]==0
 assert es[-1]['memory']['metal_allocated_bytes']<=4194304
 return st
states=verify_model(es);negative=[]
mutations={
 'wrong_ids':lambda d:next(e for e in d if e['event']=='completed').update(generated_ids=[0]*8),
 'wrong_state':lambda d:next(e for e in d if e['event']=='diagnostic_full_state').update(hidden='0'*64),
 'wrong_epochs':lambda d:next(e for e in d if e['event']=='diagnostic_native_kv_hoist').update(epochs=[0]*4),
 'wrong_capture_frame':lambda d:next(e for e in d if e['event']=='diagnostic_trained_attention_fixture').update(invocation=15),
 'false_capture':lambda d:next(e for e in d if e['event']=='diagnostic_trained_attention_fixture')['metadata'].update(completed=False),
 'owner_leak':lambda d:d[-1]['memory'].update(garnet_live_matrix_owners=1),
}
for label,mutate in mutations.items():
 d=copy.deepcopy(es);mutate(d)
 try:verify_model(d)
 except (AssertionError,KeyError,ValueError,TypeError):negative.append(label)
 else:raise AssertionError('Corrupt record admitted: '+label)
assert meta['completed'] and not meta['timing_admitted'] and meta['attention_calls']==32
assert [meta[k] for k in ['batch','chunk','capacity','heads','kv_heads','dim','first_slot']]==[4,128,4096,32,8,128,1920]
assert meta['gated'] is False and meta['reserved_bytes']<512*1024*1024 and [x['layer'] for x in meta['layers']]==[0,15,31]
shapes=[(512,4096),(512,1024),(512,1024),(4,4194304),(4,4194304),(4,128),(4,128),(512,4096),(512,4096)]
names=['q','k','v','cache_k','cache_v','slots','active','gate','output'];state16=next(e for e in states if e['invocation']==16)
result=dict(passed=False,scope=__doc__,full_model_ids_match=True,complete_state_digest_pairs=260,source_cache_matches_state16_digests=0,layers=[],negative_model_controls_rejected=negative,performance_measured=False,quality_qualified=False,goal_complete=False)
expected_files={'capture.json'};payloads={};failures=[]
for record in meta['layers']:
 layer=record['layer'];assert len(record['planes'])==9;arrays={};statistics={}
 for p,(rows,cols),name in zip(record['planes'],shapes,names):
  filename=f'layer{layer}-{name}.f32';assert p['name']==name and p['file']==filename
  assert (p['rows'],p['columns'],p['logical_elements'],p['parent_elements'],p['guard_elements'])==(rows,cols,rows*cols,rows*cols+32,16)
  assert p['source_offset']>=0 and p['source_offset']%4==0 and p['source_row_bytes']>=cols*4 and p['source_row_bytes']%4==0
  path=r/'capture2'/filename;expected_files.add(filename);a=np.memmap(path,dtype='<f4',mode='r');assert a.size==rows*cols+32
  assert np.all(a[:16].view('<u4')==0x7fc12345) and np.all(a[-16:].view('<u4')==0x7fc12345)
  data=a[16:-16].reshape(rows,cols);assert np.isfinite(data).all();arrays[name]=data
  payloads[filename]=sha(path)
  if name in ['q','k','v','gate','output']:statistics[name]=dict(min=float(data.min()),max=float(data.max()),mean_absolute=float(np.abs(data).mean(dtype=np.float64)))
 q=arrays['q'].reshape(4,128,32,128);k=arrays['k'].reshape(4,128,8,128);v=arrays['v'].reshape(4,128,8,128)
 kc=arrays['cache_k'].reshape(4,8,4096,128);vc=arrays['cache_v'].reshape(4,8,4096,128);out=arrays['output'].reshape(4,128,32,128)
 assert np.array_equal(arrays['slots'],np.tile(np.arange(1920,2048,dtype='<f4'),(4,1))) and np.all(arrays['active']==1)
 assert np.array_equal(kc[:,:,1920:2048,:].view('<u4'),k.transpose(0,2,1,3).view('<u4')) and np.array_equal(vc[:,:,1920:2048,:].view('<u4'),v.transpose(0,2,1,3).view('<u4'))
 assert np.all(kc[:,:,2048:,:]==0) and np.all(vc[:,:,2048:,:]==0)
 for part,name in enumerate(['cache_k','cache_v']):
  actual=hashlib.sha256(arrays[name].view('<u1')).hexdigest();assert actual==state16['layers'][layer][part], 'Captured cache not bound to actual model state';result['source_cache_matches_state16_digests']+=1
 maximum=0.;absolute=0.;points=0
 for row in range(4):
  for t in [0,64,127]:
   visible=1920+t+1
   for head in [0,7,8,31]:
    key=kc[row,head//4,:visible,:].astype(np.float64);value=vc[row,head//4,:visible,:].astype(np.float64)
    scores=np.sum(key*q[row,t,head,:].astype(np.float64),axis=1)/math.sqrt(128)
    probability=np.exp(scores-scores.max());probability/=probability.sum();expected=np.sum(probability[:,None]*value,axis=0)
    error=np.abs(out[row,t,head,:].astype(np.float64)-expected);fraction=error/(1e-5+2e-4*np.abs(expected));maximum=max(maximum,float(fraction.max()));absolute=max(absolute,float(error.max()));points+=128
 result['layers'].append(dict(layer=layer,fp64_values=points,original_fixed_bound_max_fraction=maximum,max_absolute_error=absolute,tensor_statistics=statistics,passed=maximum<=1))
 if maximum>1:failures.append(layer)
assert {p.name for p in (r/'capture2').iterdir()}==expected_files
(r/'capture2-payload-hashes1.json').write_text(json.dumps(payloads,indent=2)+'\n')
result.update(passed=not failures,failed_fixed_bound_layers=failures,fp64_values=sum(x['fp64_values'] for x in result['layers']),fixed_bound='1e-5+2e-4*abs(FP64 reference), unchanged',exported_parent_words=sum(p['parent_elements'] for l in meta['layers'] for p in l['planes']),reserved_bytes=meta['reserved_bytes'],postclose_metal_bytes=es[-1]['memory']['metal_allocated_bytes'],warning='CoreAnalytics context-leak warnings retained; owner counts do not explain them.')
(r/'trained-capture-audit2.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['passed'],'Actual trained attention violates existing fixed FP64 bound'
