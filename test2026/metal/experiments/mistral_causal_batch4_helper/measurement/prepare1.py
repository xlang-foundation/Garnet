"""Derive timed bursts from the qualified causal B4 worker; no inference edits."""
from pathlib import Path
import hashlib,json,difflib
R=Path(__file__).resolve().parent;Q=R.parent/'mistral-causal-batch4-helper1'
read=lambda p:json.loads(p.read_text())
def save(n,v):
 p=R/n;assert not p.exists(),n;p.write_text(json.dumps(v,indent=2)+'\n')
assert read(Q/'productive-audit1.json')['passed']
inputs=read(Q/'productive-input1.json')
for job in inputs['jobs']:job['limit']=128
inputs.update(correctness_only=False,performance_claim='Native request burst measurement only; no HTTP comparison admission.')
save('input1.json',inputs)
base=(Q/'productive_worker1.py').read_text();s=base;changes=[]
def edit(a,b):
 global s
 assert s.count(a)==1,(a[:100],s.count(a));s=s.replace(a,b);changes.append(dict(old=a,new=b))
edit('"""Untimed native C4 scheduler with real batched helper drafts and exact target verification."""','"""Native C4 burst timing; cold plus two warm waves; all helper/target work included."""')
edit('target_tokenizer=garnet.tokenizer();', 'wave_records=[]\ntarget_tokenizer=garnet.tokenizer();')
start=s.index('def save(done):');end=s.index('def helper_epochs(labels):')
old=s[start:end]
new="""def save(done):
 with open(output,'w') as f:json.dump(dict(completed=done,mode=mode,inputs=inputs,backend=garnet.backend_info(),library=garnet.native_library_path(),prepared=prepared,waves=wave_records,cleanup=cleanup,scope=__doc__),f,indent=2)
def capture(stage):pass
def helper_state():return None
"""
edit(old,new)
old="""  for number,item in enumerate(inputs['jobs']):
   lease=draft.session.claim_request_row(number+1);assert lease[0]==number;helper_leases[item['label']]=lease
"""
edit(old,'')
start=s.index(" for item in inputs['jobs']:\n  assert target_tokenizer.encode")
end=s.index('\nfinally:',start)
old=s[start:end];body=old
body=body.replace(" for item in inputs['jobs']:\n  assert target_tokenizer.encode(item['rendered'])==item['prompt_ids'];identity=", " for item in inputs['jobs']:\n  identity=")
assert body!=old
body=body.replace("job['generated'].append(token);job['finished']=", "job['generated'].append(token);job['first_token_monotonic']=time.monotonic();job['finished']=",1)
body=body.replace("      if job['finished']:del jobs[q['id']]", "      if job['finished']:job['finished_monotonic']=time.monotonic();del jobs[q['id']]")
body=body.replace("      job=jobs.pop(q['id'])", "      job=jobs.pop(q['id']);job['finished_monotonic']=time.monotonic()")
body=body.replace("    print('productive4_decode',mode,[len(j['generated']) for j in all_jobs.values()],flush=True)", '')
body=body.replace('  save(False);assert len(ticks)<100', '  assert len(ticks)<256')
prefix=""" for item in inputs['jobs']:assert target_tokenizer.encode(item['rendered'])==item['prompt_ids']
 for wave in range(3):
  jobs={};all_jobs={};ticks=[];snapshots=[];helper_calls=[];helper_rounds=[]
  if draft:
   for number,item in enumerate(inputs['jobs']):
    lease=draft.session.claim_request_row(wave*4+number+1);assert lease[0]==number;helper_leases[item['label']]=lease
  began=time.monotonic()
"""
suffix="""
  ended=time.monotonic();wave_records.append(dict(wave=wave,phase='cold' if wave==0 else 'warm',began_monotonic=began,ended_monotonic=ended,jobs=list(all_jobs.values()),ticks=ticks,helper_calls=helper_calls,helper_rounds=helper_rounds,aggregate_tokens_per_second=sum(len(j['generated']) for j in all_jobs.values())/(ended-began),per_request=[dict(label=j['label'],native_ttft_seconds=j['first_token_monotonic']-began,decode_tokens_per_second=(len(j['generated'])-1)/(j['finished_monotonic']-j['first_token_monotonic'])) for j in all_jobs.values()]))
  print('native4_wave',mode,wave,round(wave_records[-1]['aggregate_tokens_per_second'],3),flush=True)
"""
new=prefix+'\n'.join(' '+line if line else '' for line in body.splitlines())+suffix
edit(old,new)
compile(s,str(R/'worker1.py'),'exec');(R/'worker1.py').write_text(s)
(R/'measurement-instrumentation1.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/productive_worker1.py',tofile='b/worker1.py')))
save('preparation1.json',dict(passed=True,base_worker=str(Q/'productive_worker1.py'),base_sha256=hashlib.sha256(base.encode()).hexdigest(),worker_sha256=hashlib.sha256(s.encode()).hexdigest(),changes=changes,inference_source_unchanged=True,diagnostic_full_state_hashes_disabled=True,per_tick_file_writes_disabled=True,native_burst_only=True,http_comparison_admitted=False,no_comparator_execution=True,goal_complete=False))
print('Prepared three-wave timed worker from qualified algorithm.')
