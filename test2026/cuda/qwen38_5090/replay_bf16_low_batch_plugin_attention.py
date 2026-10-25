"""Strict wrapper forwarding replay against closed same-kernel GPU observations."""
import hashlib,json

def records(raw):
 result=[]
 for line in raw.splitlines():
  if not line.startswith(b'{'):continue
  item=json.loads(line)
  if 'event' in item or 'status' in item:result.append(item)
 return result

def compare(parent,parent_binary,observed,observed_binary):
 def require(ok,message):
  if not ok:raise ValueError(message)
 canonical=lambda item:json.dumps(item,sort_keys=True,separators=(',',':'),allow_nan=False)
 parent_final={'status':'bf16_low_batch_overlay_collected_needs_fp64_replay','cases':44,'witness_bytes':119040*4,'performance_admitted':False,'model_qualified':False}
 require(len(parent)==90 and canonical(parent[-1])==canonical(parent_final),'Closed parent corpus differs')
 require(parent[0].get('sm')==120 and parent[0].get('uuid')=='GPU-265ee66c-9b79-cd3b-0c63-dd6942ddb5e6','Parent GPU differs')
 require(len(parent_binary)==119040*4,'Parent witness size differs')
 require(len(observed)==90,'Wrapper observation stream incomplete')
 for i,(a,b) in enumerate(zip(parent[:-1],observed[:-1])):require(canonical(a)==canonical(b),'Wrapper case/identity/control differs at record '+str(i))
 final={**parent_final,'status':'bf16_low_batch_plugin_collected_needs_fp64_replay','host_plugin_enqueue_calls':87,'negative_calls_use_direct_kernel':True}
 require(canonical(observed[-1])==canonical(final),'Wrapper final admission/counters differ')
 require(observed_binary==parent_binary,'Wrapper selected scalar bytes differ from qualified kernel')
 cases=parent[2:-1:2];require(len(cases)==44 and sum(c['rejected_calls'] for c in cases)==104,'Parent case/negative count differs')
 require(sum(c['graph_replay_exact'] is True for c in cases)==3 and sum(c['prefix_cache_exact'] is True for c in cases)==40,'Parent graph/prefix count differs')
 return {'status':'b1_plugin_forwarding_observations_match_qualified_kernel','cases':44,'selected_fp32_scalar_outputs_exact':119040,'witness_sha256':hashlib.sha256(observed_binary).hexdigest(),'host_plugin_enqueue_calls':87,'graph_replays_exact':3,'cache_equivalence_cases_exact':40,'direct_kernel_negative_calls':104,'wrapper_negative_controls_in_this_stream':0,'model_qualified':False,'performance_admitted':False,'scope':'Exact selected witness and complete raw observation record parity against the closed same-kernel operator corpus. Requires separate archive/source/parent proof validation and CPU wrapper negative contracts. No whole-model, broad-quality, 4K or speed admission.'}
