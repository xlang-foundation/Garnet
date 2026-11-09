# SPDX-License-Identifier: Apache-2.0
"""Audit actual-model profile memory reuse, all-layer state and token parity.
Args RESEARCH_DIR NEW_OUTPUT_JSON. Allocation snapshots are not peaks/paging proof.
No model quality or serving performance qualification from this diagnostic.
"""
import json,sys,copy,hashlib
from pathlib import Path

def validate(a,b,ia,ib,fixture):
 assert a['completed'] and b['completed'] and a['mode']=='control' and b['mode']=='candidate'
 for key in ['graphs','plans']:assert a['prepared'][key]==b['prepared'][key]==(224 if key=='graphs' else 3)
 for feature in ['prefill_lifetime_reuse','prefill_alias_groups','native_prefill_hidden_pool','chunk_attention_full_write_scratch']:
  assert feature not in a['prepared']['backend'] and feature in b['prepared']['backend']
 assert len(a['cases'])==len(b['cases'])==len(fixture['cases'])==2
 checked=0
 for x,y,f in zip(a['cases'],b['cases'],fixture['cases']):
  assert x['id']==y['id']==f['id'] and x['prompt_ids']==y['prompt_ids']==f['prompts']
  for k in ['after_prefill','generated_ids','after_decode']:assert x[k]==y[k],k
  for key in ['after_prefill','after_decode']:
   assert len(y[key]['kv'])==32 and all(len(v)==2 and all(len(h)==64 for h in v) for v in y[key]['kv']) and len(y[key]['hidden'])==64
  assert len(x['frames'])==len(y['frames'])
  for u,v in zip(x['frames'],y['frames']):
   for k in ['offset','chunk','active_tokens']:assert u[k]==v[k]
   for k in ['packed','affine','misses','partials']:assert u['stats'][k]==v['stats'][k]
   assert u['stats']['hidden_bindings']==0 and v['stats']['hidden_bindings']==32
  assert len(y['generated_ids'])==4 and all(len(g)==8 for g in y['generated_ids']);checked+=sum(map(len,y['generated_ids']))
 for results,inventory in [(a,ia),(b,ib)]:
  assert results['prepared']['memory']['metal_allocated_bytes']==inventory['device_snapshot']['metal_allocated_bytes']
  assert inventory['unique_allocated_bytes']==sum(x['allocated_bytes'] for x in inventory['buffers'])
  for k in ['garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets']:assert results['cleanup'][k]==0
  assert results['cleanup']['metal_allocated_bytes']<=4194304
 layout=b['prepared']['layout'];assert len(layout)==224 and len(set(x['key'] for x in layout))==224
 assert sum(x['groups'] for x in layout)==224 and sum(x['slots'] for x in layout)==448
 roles={}
 for ref in ib['references']:roles.setdefault(ref['role'],set()).add(ref['buffer_id'])
 pool=roles['shared_prefill_scratch_pool'];states=roles['private_state1']|roles['private_state2'];weights=roles['weight_record'];hidden=roles['prefill_hidden_pingpong']
 assert len(pool)==91 and len(hidden)==14 and len(roles['private_state1'])==len(roles['private_state2'])==32
 assert not pool&(states|weights|hidden) and not hidden&(states|weights)
 assert len({r['tag'] for r in ib['references'] if r['role']=='slot' and r['buffer_id'] in pool and r['resident_mask_writer']})==224
 assert sum(ib['buffers'][i]['length_bytes'] for i in states)==4*1024**3
 assert sum(ib['buffers'][i]['length_bytes'] for i in roles['packed_prefill_half_input'])==14680064
 old=a['prepared']['memory']['metal_allocated_bytes'];new=b['prepared']['memory']['metal_allocated_bytes'];assert 0<new<old<19069665280
 return dict(passed=True,generated_ids_equal=checked,full32_layer_kv_hidden_equal=True,prepared_control_bytes=old,prepared_candidate_bytes=new,saved_bytes=old-new,saved_fraction=(old-new)/old,alias_groups=224,alias_slots=448,shared_pool_buffers=91,hidden_pingpong_buffers=14)

if __name__=='__main__':
 r=Path(sys.argv[1]);out=Path(sys.argv[2]);assert not out.exists()
 for name in ['control1','candidate1']:assert (r/(name+'.status')).read_text().strip()=='0'
 a=json.load(open(r/'control1/results.json'));b=json.load(open(r/'candidate1/results.json'));ia=json.load(open(r/'control1/prepared-inventory.json'));ib=json.load(open(r/'candidate1/prepared-inventory.json'));fixture=json.load(open(r.parent/'mistral-packed-http1/fixture1.json'))
 result=validate(a,b,ia,ib,fixture);rejected=[]
 def reject(label,mutate):
  items=copy.deepcopy([a,b,ia,ib,fixture]);mutate(*items)
  try:validate(*items)
  except (AssertionError,KeyError,IndexError,TypeError,ValueError):rejected.append(label)
  else:raise AssertionError('accepted '+label)
 reject('token',lambda a,b,ia,ib,f:b['cases'][0]['generated_ids'][0].__setitem__(0,-1))
 reject('last-layer-kv',lambda a,b,ia,ib,f:b['cases'][1]['after_decode']['kv'][31].__setitem__(1,'0'*64))
 reject('hidden',lambda a,b,ia,ib,f:b['cases'][0]['after_prefill'].__setitem__('hidden','0'*64))
 reject('alias-layout',lambda a,b,ia,ib,f:b['prepared']['layout'][0].__setitem__('slots',0))
 reject('missing-hidden-bindings',lambda a,b,ia,ib,f:b['cases'][0]['frames'][0]['stats'].__setitem__('hidden_bindings',0))
 reject('route',lambda a,b,ia,ib,f:b['cases'][0]['frames'][0]['stats'].__setitem__('packed',0))
 reject('wrong-memory',lambda a,b,ia,ib,f:b['prepared']['memory'].__setitem__('metal_allocated_bytes',1))
 reject('wrong-inventory-total',lambda a,b,ia,ib,f:ib.__setitem__('unique_allocated_bytes',0))
 reject('wrong-fixture',lambda a,b,ia,ib,f:f['cases'][0]['prompts'][0].__setitem__(0,-1))
 reject('leak',lambda a,b,ia,ib,f:b['cleanup'].__setitem__('garnet_live_matrix_owners',1))
 result.update(scope=__doc__,rejected_corruptions=rejected)
 paths=[r/'control1/results.json',r/'candidate1/results.json',r/'control1/prepared-inventory.json',r/'candidate1/prepared-inventory.json',r/'library1/libgarnet.dylib',Path(__file__)]
 result['files_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};out.write_text(json.dumps(result,indent=2)+'\n');print('profile memory audit PASS',result['saved_bytes'],len(rejected),'corruptions')
