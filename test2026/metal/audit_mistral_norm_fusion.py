# SPDX-License-Identifier: Apache-2.0
"""Audit optional norm fusion: full-model state, body routes and native FP64 oracle.
Args RESEARCH_DIR NEW_OUTPUT. Diagnostic timings only; no HTTP/quality admission.
Head-only snapshots retain legacy last-body norm counters and are not coverage.
"""
import json,copy,hashlib,sys,re,statistics
from pathlib import Path

def validate(control,candidates,fixture):
 assert control['completed'] and control['mode']=='control' and len(candidates)==2
 assert len(control['cases'])==len(fixture['cases'])==2
 report=[]
 for c in candidates:
  assert c['completed'] and c['mode']=='candidate' and len(c['cases'])==2
  for k in ['graphs','plans','layout']:assert c['prepared'][k]==control['prepared'][k]
  for feature in ['residual_norm','packed_prefill_norm']:assert feature in c['prepared']['backend'] and feature not in control['prepared']['backend']
  assert c['prepared']['memory']['metal_allocated_bytes']==control['prepared']['memory']['metal_allocated_bytes']==9346269184
  for a,b,f in zip(control['cases'],c['cases'],fixture['cases']):
   assert a['id']==b['id']==f['id'] and a['prompt_ids']==b['prompt_ids']==f['prompts']
   for key in ['after_prefill','after_decode','generated_ids']:assert a[key]==b[key],key
   for key in ['after_prefill','after_decode']:assert len(b[key]['kv'])==32 and all(len(x)==2 and all(len(h)==64 for h in x) for x in b[key]['kv']) and len(b[key]['hidden'])==64
   assert len(a['frames'])==len(b['frames']) and len(a['decode'])==len(b['decode'])==7
   for x,y in zip(a['frames'],b['frames']):
    for key in ['offset','chunk','active_tokens']:assert x[key]==y[key]
    for key in ['packed','affine','misses','partials','hidden_bindings']:assert x['stats'][key]==y['stats'][key]
    assert (x['stats']['residual_norm'],x['stats']['packed_norm'])==(0,0)
    assert (y['stats']['residual_norm'],y['stats']['packed_norm'])==(0,32)
   for x,y in zip(a['decode'],b['decode']):
    for key in ['packed','affine','misses','partials','hidden_bindings']:assert x[key]==y[key]
    assert (x['residual_norm'],x['packed_norm'])==(0,0) and (y['residual_norm'],y['packed_norm'])==(32,0)
   assert all(len(x)==8 for x in b['generated_ids']) and len(b['generated_ids'])==4
   report.append(dict(case=b['id'],control_prefill_seconds=a['prefill_diagnostic_seconds'],candidate_prefill_seconds=b['prefill_diagnostic_seconds'],prefill_ratio=a['prefill_diagnostic_seconds']/b['prefill_diagnostic_seconds'],control_decode_gpu_ms=statistics.median(x['gpu_ms'] for x in a['decode']),candidate_decode_gpu_ms=statistics.median(x['gpu_ms'] for x in b['decode'])))
 for d in [control]+candidates:
  for k in ['garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets']:assert d['cleanup'][k]==0
  assert d['cleanup']['metal_allocated_bytes']<=4194304
 return report

if __name__=='__main__':
 r=Path(sys.argv[1]);out=Path(sys.argv[2]);assert not out.exists()
 names=['control1','candidate1','candidate2']
 for label in names:assert (r/(label+'.status')).read_text().strip()=='0'
 docs=[json.load(open(r/label/'results.json')) for label in names];f=json.load(open(r.parent/'mistral-packed-http1/fixture1.json'));result=validate(docs[0],docs[1:],f)
 assert (r/'ctest2.status').read_text().strip()=='0'
 text=(r/'build1/Testing/Temporary/LastTest.log').read_text();m=re.search(r'norm_fusion_oracle packed_cases=(\d+) residual_cases=(\d+) fp64_words=(\d+)',text);assert m and tuple(map(int,m.groups()))==(480,120,124653520)
 rejected=[]
 def reject(label,mutate):
  a,b,ff=copy.deepcopy((docs[0],docs[1:],f));mutate(a,b,ff)
  try:validate(a,b,ff)
  except (AssertionError,KeyError,IndexError,TypeError,ValueError):rejected.append(label)
  else:raise AssertionError('accepted '+label)
 reject('generated-token',lambda a,b,f:b[0]['cases'][0]['generated_ids'][0].__setitem__(0,-1))
 reject('last-layer-kv',lambda a,b,f:b[1]['cases'][1]['after_decode']['kv'][31].__setitem__(1,'0'*64))
 reject('hidden',lambda a,b,f:b[0]['cases'][0]['after_prefill'].__setitem__('hidden','0'*64))
 reject('missing-prefill-fusion',lambda a,b,f:b[0]['cases'][0]['frames'][0]['stats'].__setitem__('packed_norm',0))
 reject('missing-decode-fusion',lambda a,b,f:b[1]['cases'][1]['decode'][0].__setitem__('residual_norm',0))
 reject('quant-route',lambda a,b,f:b[0]['cases'][0]['frames'][0]['stats'].__setitem__('packed',0))
 reject('prepared-layout',lambda a,b,f:b[0]['prepared']['layout'][0].__setitem__('groups',0))
 reject('changed-fixture',lambda a,b,f:f['cases'][0]['prompts'][0].__setitem__(0,-1))
 reject('owner-leak',lambda a,b,f:b[0]['cleanup'].__setitem__('garnet_live_matrix_owners',1))
 paths=[r/label/'results.json' for label in names]+[r/'library1/libgarnet.dylib',r/'ctest2.log',r/'build1/Testing/Temporary/LastTest.log',Path(__file__)]
 out.write_text(json.dumps(dict(passed=True,scope=__doc__,full32_layer_state_and_ids_equal=True,candidate_generated_ids_equal=128,native_fp64_normalized_words=124653520,packed_operator_cases=480,residual_operator_cases=120,ctests=32,body_prefill_fusions_per_command=32,body_decode_fusions_per_command=32,samples=result,rejected_corruptions=rejected,no_serving_or_quality_admission=True,files_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}),indent=2)+'\n');print('norm fusion audit PASS',len(rejected),'corruptions; no serving gain admission')
