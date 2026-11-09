"""Saved-data-only regression audit for the new numerical profile; no inference."""
from pathlib import Path
import copy,hashlib,json,math,re,sys
z=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
load=lambda n:json.loads((z/n).read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def rows(p):return [json.loads(l) for l in p.read_text().splitlines() if l.startswith('{')]
def flat(v):
 if isinstance(v,list):
  return [x for a in v for x in flat(a)]
 return [v]
b=load('bindings1.json');fixture=load('fixture1.json');status=load('model-state-status1.json')
assert status['status']=='collected_pending_audit'
assert [w['mode'] for w in status['waves']]==['off','on']
assert all(w['status']=='complete' and w['returncode']==0 and w['last_event']=='closed' for w in status['waves'])
for name,key in [('worker-state1.py','worker_sha256'),('fixture1.json','fixture_sha256'),('source-audit1.json','source_audit_sha256')]:assert sha(z/name)==b[key],name
assert sha(Path(b['original_control']))==b['original_control_sha256']
for n,h in load('source-audit1.json')['candidate_owned_files_sha256'].items():assert sha(Path(b['source'])/n)==h,n
for mode,tests in [('off',50),('on',51)]:
 assert sha(Path(b['libraries'][mode]['path'])/'libgarnet.dylib')==b['libraries'][mode]['sha256']
 assert sha(z/f'profile-{mode}1.json')==b['libraries'][mode]['profile_sha256']
 assert load(f'ctest-{mode}1-status.json')['returncode']==0
 assert f'100% tests passed out of {tests}' in (z/f'ctest-{mode}1.stdout').read_text()
control=json.loads(Path(b['original_control']).read_text())
assert sha(Path(b['prior_state_study'])/'provenance1.json')==b['prior_state_provenance_sha256']
assert sha(Path(b['prior_state_study'])/'model-on1.stdout')==b['prior_state_stdout_sha256']
prior=rows(Path(b['prior_state_study'])/'model-on1.stdout')
prior_states=[e for e in prior if e['event']=='diagnostic_full_state']
prior_routes=[e for e in prior if e['event']=='diagnostic_native_kv_hoist']
data={m:rows(z/f'model-{m}1.stdout') for m in ['off','on']}
def verify(d):
 states={};routes={};post={};numeric={};bits={}
 for mode in ['off','on']:
  es=d[mode];assert es and not [e for e in es if e['event']=='error']
  assert len([e for e in es if e['event']=='ready'])==1
  complete=[e for e in es if e['event']=='completed'];assert len(complete)==4
  ends={e['id']:e for e in complete};assert len(ends)==4
  for r in fixture['requests']:
   key='trained-'+str(r['row']);e=ends[key];assert e['prompt_ids']==r['token_ids']==control[key]['prompt_ids']
   assert e['generated_ids']==control[key]['generated_ids'] and len(e['generated_ids'])==8
   assert e['text']==control[key]['text'] and e['cached_tokens']==0 and e['finish_reason']=='length'
  assert es[-1]['event']=='closed' and len([e for e in es if e['event']=='closed'])==1
  mem=es[-1]['memory']
  for k in ['garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets']:assert mem[k]==0,k
  assert mem['metal_allocated_bytes']<=4194304;post[mode]=mem['metal_allocated_bytes']
  rs=[e for e in es if e['event']=='diagnostic_native_kv_hoist'];assert len(rs)==25
  assert rs==prior_routes,'Native input/slot/epoch/commit sequence differs from qualified hoist control'
  cs=[e for e in es if e['event']=='diagnostic_committed_attention'];assert len(cs)==25
  assert [e['invocation'] for e in cs]==list(range(1,26))
  assert sum(e['mps_dispatches'] for e in rs)==512 and sum(e['dispatches'] for e in rs)==1088
  for r,c in zip(rs,cs):
   expected=r['mps_dispatches'] if mode=='on' else 0;assert c['dispatches']==expected
   width=len(r['ids'])//4;owners=sum(any(r['active'][i*width:(i+1)*width]) for i in range(4))
   factor=2 if expected else 4
   assert c['matrices']==r['mps_dispatches']*owners*8*factor
   # These four full owners or sparse scalar tail have no compact-owner MPS span.
   assert c['skipped_matrices']==r['mps_dispatches']*(4-owners)*8*factor
  st=[e for e in es if e['event']=='diagnostic_full_state'];assert [e['invocation'] for e in st]==[1,16,18,25]
  for e in st:
   assert len(e['layers'])==32 and all(len(p)==2 for p in e['layers'])
   assert all(re.fullmatch('[0-9a-f]{64}',h) for p in e['layers'] for h in p) and re.fullmatch('[0-9a-f]{64}',e['hidden'])
   a=flat(e['hidden_values']);assert len(a)==16384 and all(type(x) in (int,float) and math.isfinite(x) for x in a)
  states[mode]=st;routes[mode]=rs
  affine=[e for e in es if e['event']=='native_affine_profile'];assert affine and all(e['misses']==0 for e in affine)
 assert [{k:v for k,v in e.items() if k!='hidden_values'} for e in states['off']]==prior_states,'OFF control full physical state changed'
 for a,c in zip(states['off'],states['on']):
  x=flat(a['hidden_values']);y=flat(c['hidden_values']);err=[abs(v-u) for u,v in zip(x,y)]
  numeric[str(a['invocation'])]=dict(words=len(x),max_absolute_error=max(err),rms_error=math.sqrt(sum(e*e for e in err)/len(err)),max_original_magnitude=max(abs(v) for v in x))
  ah=[h for pair in a['layers'] for h in pair]+[a['hidden']];ch=[h for pair in c['layers'] for h in pair]+[c['hidden']]
  bits[str(a['invocation'])]=dict(equal_digest_pairs=sum(u==v for u,v in zip(ah,ch)),total_pairs=65)
 return dict(postclose_metal_bytes=post,hidden_errors_descriptive_only=numeric,whole_state_bit_identity_descriptive_only=bits,committed_dispatches=dict(off=0,on=512),actual_mps_products={m:sum(e['matrices'] for e in d[m] if e['event']=='diagnostic_committed_attention') for m in d})
native_line=re.search(r'Native committed KV: (\d+) cells, (\d+) complete output words, (\d+) original-bound FP64 values, maximum fraction ([0-9.eE+-]+);.* PASS', (z/'ctest-on1-details.log').read_text())
assert native_line and tuple(map(int,native_line.groups()[:3]))==(6,7127040,864) and float(native_line.group(4))<=1
result=verify(data)
checks=[]
mutations={
 'missing_route':lambda d:d['on'].__delitem__(next(i for i,e in enumerate(d['on']) if e['event']=='diagnostic_committed_attention')),
 'wrong_products':lambda d:next(e for e in d['on'] if e['event']=='diagnostic_committed_attention').update(matrices=4096),
 'wrong_ids':lambda d:next(e for e in d['on'] if e['event']=='completed').update(generated_ids=[0]*8),
 'wrong_epoch':lambda d:next(e for e in d['on'] if e['event']=='diagnostic_native_kv_hoist').update(epochs=[0]*4),
 'hidden_nan':lambda d:next(e for e in d['on'] if e['event']=='diagnostic_full_state').update(hidden_values=[float('nan')]*16384),
 'owner_leak':lambda d:d['on'][-1]['memory'].update(garnet_live_matrix_owners=1),
 'off_state_drift':lambda d:next(e for e in d['off'] if e['event']=='diagnostic_full_state').update(hidden='0'*64),
 'extra_error':lambda d:d['on'].insert(-1,dict(event='error',error='injected')),
}
for label,fn in mutations.items():
 d=copy.deepcopy(data);fn(d)
 try:verify(d)
 except (AssertionError,KeyError,ValueError,TypeError):checks.append(label)
 else:raise AssertionError('Corrupt saved data admitted: '+label)
result.update(passed=True,scope=b['scope'],completed_requests_per_engine=4,generated_tokens_per_engine=32,all_ids_and_text_match_saved_native_control=True,body_calls=25,hoisted_commits_per_engine=1088,original_off_state_exact_to_prior=True,negative_controls_rejected=checks,native_tests=dict(on=51,off=50),numeric_primitive=dict(shapes=6,complete_output_words=7127040,FP64_reference_values=864,fixed_bound='1e-5+2e-4*abs(reference)',maximum_fraction_of_bound=float(native_line.group(4))),full_model_numerical_oracle=False,cached_return_tested=False,quality_qualified=False,serving_speed_measured=False,goal_complete=False,warning='CoreAnalytics context-leak warnings retained in both logs; bounded native owners do not explain them.')
(z/'model-state-audit1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
