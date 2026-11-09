# SPDX-License-Identifier: Apache-2.0
"""Audit optimized direct encoder diagnostics. Args RESEARCH_DIR NEW_OUTPUT_JSON.
Checks exact full-model states/IDs/routes, raw calibrated intervals and coverage.
Unions account for overlaps; no summation of encoder durations as elapsed time.
"""
import sys,json,math,copy,hashlib,collections,pathlib

def union_ms(intervals,scale):
 total=0;end=None
 for a,b in sorted(intervals):
  if end is None or a>end:total+=b-a;end=b
  elif b>end:total+=b-end;end=b
 return total*scale

def audit(off,on,events):
 assert off['completed'] and on['completed'] and off['mode']=='off' and on['mode']=='on'
 assert off['prepared']['graphs']==on['prepared']['graphs']==224
 assert off['prepared']['plans']==on['prepared']['plans']==3
 assert len(off['cases'])==len(on['cases'])==2
 expected=[];ids=0
 for a,b in zip(off['cases'],on['cases']):
  for k in ['id','prompt_ids','after_prefill','generated_ids','after_decode']:assert a[k]==b[k],('parity',k)
  assert len(b['after_prefill']['kv'])==len(b['after_decode']['kv'])==32
  for key in ['after_prefill','after_decode']:
   for hashes in b[key]['kv']:
    assert len(hashes)==2 and all(len(h)==64 for h in hashes)
   assert len(b[key]['hidden'])==64
  assert len(a['frames'])==len(b['frames']) and len(a['decode'])==len(b['decode'])==7
  for x,y in zip(a['frames'],b['frames']):
   for k in ['offset','chunk','active_tokens']:assert x[k]==y[k]
   for k in ['packed','affine','misses','partials']:assert x['stats'][k]==y['stats'][k]
   assert y['stats']['packed']==(160 if y['active_tokens']>=128 else 0) and y['stats']['misses']==0
   expected.append((b['id'],'prefill',y['offset'],y['stats']))
  for x,y in [(a['head'],b['head'])]+list(zip(a['decode'],b['decode'])):
   for k in ['packed','affine','misses','partials']:assert x[k]==y[k]
  expected.append((b['id'],'head',0,b['head']))
  for i,y in enumerate(b['decode']):expected.append((b['id'],'decode',i,y))
  assert len(b['generated_ids'])==4 and all(len(g)==8 for g in b['generated_ids']);ids+=sum(map(len,b['generated_ids']))
 for d in [off,on]:
  assert d['cleanup']['metal_allocated_bytes']<=4*1024*1024
  for k in ['garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets']:assert d['cleanup'][k]==0
 assert len(events)==len(expected)
 reports=[];total_encoders=0
 for e,(case,phase,offset,stats) in zip(events,expected):
  assert e['diagnostic_only'] is True and e['mps_encoders_observed'] is False
  assert e['partial_submissions']==stats['partials'] and abs(e['gpu_ms']-stats['gpu_ms'])<1e-4
  assert e['cpu1']>e['cpu0']>0 and e['gpu1']>e['gpu0']>0
  scale=(e['cpu1']-e['cpu0'])/(e['gpu1']-e['gpu0'])/1e6
  assert 0<len(e['rows'])<=2048
  spans=[];sites=collections.defaultdict(list);commands=[]
  for r in e['rows']:
   assert e['gpu0']<=r['start']<=r['end']<=e['gpu1'] and r['start']>0
   assert math.isfinite(r['ms']) and r['ms']>=0
   assert abs((r['end']-r['start'])*scale-r['ms'])<max(1e-6,r['ms']*1e-7)
   assert 0<=r['command']<=e['partial_submissions'];commands.append(r['command'])
   assert r['site'].startswith(('Encode','TryEncode'))
   spans.append((r['start'],r['end']));sites[r['site']].append(spans[-1])
  assert commands==sorted(commands) and sorted(set(commands))==list(range(e['partial_submissions']+1))
  covered=union_ms(spans,scale);span=(max(b for a,b in spans)-min(a for a,b in spans))*scale
  assert 0<covered<=span+1e-6 and span<=e['gpu_ms']+1.0
  assert e['gpu_ms']<=(e['cpu1']-e['cpu0'])/1e6+1.0
  total_encoders+=len(spans)
  reports.append(dict(case=case,phase=phase,offset=offset,gpu_ms=e['gpu_ms'],direct_union_ms=covered,coverage_fraction=covered/e['gpu_ms'],site_unions_ms={k:union_ms(v,scale) for k,v in sites.items()},encoders=len(spans)))
 return dict(passed=True,scope=__doc__,generated_ids_equal=ids,all_layer_kv_and_hidden_equal=True,total_batches=len(reports),total_direct_encoders=total_encoders,batches=reports,limitations=['Diagnostic counter overhead present; not serving performance','MPS-owned encoders and command gaps not attributed','Site interval unions may overlap each other; do not add as elapsed time'])

if __name__=='__main__':
 root=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);assert not out.exists()
 off=json.load(open(root/'off1/results.json'));on=json.load(open(root/'on1/results.json'))
 assert (root/'off1.status').read_text().strip()==(root/'on1.status').read_text().strip()=='0'
 assert 'metal_gpu_counters ' not in (root/'off1.log').read_text()
 text=(root/'on1.log').read_text();assert 'metal_op_profile ' not in text
 events=[json.loads(line.split('metal_gpu_counters ',1)[1]) for line in text.splitlines() if line.startswith('metal_gpu_counters ')]
 result=audit(off,on,events);rejected=[]
 def reject(label,mutate):
  a,b,c=copy.deepcopy(off),copy.deepcopy(on),copy.deepcopy(events);mutate(a,b,c)
  try:audit(a,b,c)
  except (AssertionError,KeyError,IndexError,TypeError,ValueError):rejected.append(label)
  else:raise AssertionError('accepted corruption '+label)
 reject('generated_token',lambda a,b,c:b['cases'][0]['generated_ids'][0].__setitem__(0,-1))
 reject('kv_hash',lambda a,b,c:b['cases'][0]['after_prefill']['kv'][31].__setitem__(1,'0'*64))
 reject('missing_batch',lambda a,b,c:c.pop())
 reject('missing_encoder',lambda a,b,c:c[0].__setitem__('rows',[]))
 reject('reversed_interval',lambda a,b,c:c[0]['rows'][0].__setitem__('end',c[0]['rows'][0]['start']-1))
 reject('timestamp_outside_clock',lambda a,b,c:c[0]['rows'][0].__setitem__('end',c[0]['gpu1']+1))
 reject('wrong_duration_units',lambda a,b,c:c[0]['rows'][0].__setitem__('ms',c[0]['rows'][0]['ms']*1000+1))
 reject('changed_submission_count',lambda a,b,c:c[0].__setitem__('partial_submissions',999))
 reject('false_mps_coverage',lambda a,b,c:c[0].__setitem__('mps_encoders_observed',True))
 reject('missing_diagnostic_label',lambda a,b,c:c[0].__setitem__('diagnostic_only',False))
 reject('route_change',lambda a,b,c:b['cases'][0]['frames'][0]['stats'].__setitem__('packed',0))
 result['rejected_corruptions']=rejected
 result['files_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'off1/results.json',root/'on1/results.json',root/'off1.log',root/'on1.log',root/'library1/libgarnet.dylib',pathlib.Path(__file__)]}
 out.write_text(json.dumps(result,indent=2)+'\n');print('GPU counter audit PASS',result['total_batches'],result['total_direct_encoders'],len(rejected),'corruptions')
