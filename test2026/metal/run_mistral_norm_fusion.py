# SPDX-License-Identifier: Apache-2.0
"""Compiled packed/residual normalization fusion actual-model parity check.
LIB SOURCE FIXTURE NEW_OUTPUT control|candidate. Serial fresh processes, frozen inputs.
No HTTP/quality/speed admission; MPS-owned encoders are deliberately unobserved.
"""
import sys,os,json,time
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet,tensor
from xModel.mistral.text_7b.metal_decoder import MistralMetalDecoder
with open(sys.argv[3]) as f:fixture=json.load(f)
out=sys.argv[4];mode=sys.argv[5];assert mode in ('control','candidate') and not os.path.exists(out);os.mkdir(out)
assert garnet.native_library_path()==sys.argv[1]+'/libgarnet.dylib'
assert os.getenv('GARNET_METAL_PROFILE_OPS') is None
with open(fixture['model']+'/config.json') as f:config=json.load(f)
decoder=None;records=[];prepared=None;cleanup=None

def f32(v,s):return tensor.tensor(v,shape=s,dtype=tensor.float32)
def witness():
 return dict(kv=[[decoder.io.verification_f32_sha256(x) for x in decoder.session.snapshot(i)] for i in range(32)],hidden=decoder.io.verification_f32_sha256(decoder.session.hidden()))
def stats():
 return dict(gpu_ms=decoder.session.gpu_ms,packed=decoder.session.packed_prefill_dispatches,affine=decoder.session.affine_profile_dispatches,misses=decoder.session.affine_profile_misses,partials=decoder.session.prefill_partial_submissions,hidden_bindings=decoder.session.prefill_hidden_pool_bindings,residual_norm=decoder.session.residual_norm_dispatches,packed_norm=decoder.session.packed_prefill_norm_dispatches)
def save(done):
 with open(out+'/results.json','w') as f:json.dump(dict(completed=done,scope=__doc__,mode=mode,prepared=prepared,cases=records,cleanup=cleanup,library=garnet.native_library_path()),f,indent=2)
try:
 decoder=MistralMetalDecoder(fixture['model']+'/model.safetensors',config,4,4096,resident=True);decoder.prepare_prefill(128)
 assert 'prefill_alias_groups' in garnet.backend_info()
 assert 'native_prefill_hidden_pool' in garnet.backend_info()
 assert ('packed_prefill_norm' in garnet.backend_info())==(mode=='candidate')
 assert ('residual_norm' in garnet.backend_info())==(mode=='candidate')
 graphs=[dict(tag='decode-'+str(i),object=x['graph']) for i,x in enumerate(decoder.layers)]
 graphs+=[dict(tag='prefill-'+k,object=v) for k,v in decoder.prefill_graphs.items()];graphs.append(dict(tag='head',object=decoder.head))
 with open(out+'/prepared-inventory.json','w') as f:f.write(garnet.memory_inventory_json(graphs,decoder.session))
 layout=[dict(key=k,reused_bytes=g.shared_prefill_scratch_bytes,groups=g.shared_prefill_alias_groups ,slots=g.shared_prefill_alias_slots ) for k,g in decoder.prefill_graphs.items()]
 prepared=dict(backend=garnet.backend_info(),layout=layout,graphs=len(decoder.prefill_graphs),plans=decoder.session.packed_prefill_plans,memory=json.loads(garnet.memory_stats_json()));assert prepared['graphs']==224 and prepared['plans']==3
 base=[l['graph'] for l in decoder.layers]
 print('norm_fusion_ready',mode,flush=True)
 for case in fixture['cases']:
  prompts=case['prompts'];lengths=[len(p) for p in prompts];inputs=[]
  for offset in range(0,max(lengths),128):
   chunk=2
   while chunk<min(128,max(lengths)-offset):chunk*=2
   slots=[offset+t if offset+t<len(p) else 0 for p in prompts for t in range(chunk)]
   active=[int(offset+t<len(p)) for p in prompts for t in range(chunk)];ids=[p[offset+t] if offset+t<len(p) else 0 for p in prompts for t in range(chunk)]
   hidden=decoder.io.stack_tokens([decoder.io.embedding([ids[row*chunk+t] for row in range(4)]) for t in range(chunk)])
   inputs.append(dict(chunk=chunk,offset=offset,active=active,hidden=hidden,position=f32([v for v in slots for _ in range(3)],[4,chunk,3]),slots=f32(slots,[4,chunk]),flags=f32(active,[4,chunk])))
  decoder.reset();epochs=[decoder.session.claim_request_row(100+i)[1] for i in range(4)];frames=[]
  start=time.monotonic()
  for x in inputs:
   decoder.session.run_leased(decoder.prefill_profile_graphs(x['chunk']),x['hidden'],x['position'],x['slots'],x['flags'],-1,epochs)
   st=stats();assert st['misses']==0 and st['packed']==(160 if sum(x['active'])>=128 else 0)
   assert st['packed_norm']==(32 if mode=='candidate' else 0) and st['residual_norm']==0
   frames.append(dict(offset=x['offset'],chunk=x['chunk'],active_tokens=sum(x['active']),stats=st))
  prediction=list(decoder.session.predict_rows(epochs));head=stats();seconds=time.monotonic()-start
  after_prefill=witness();generated=[[p] for p in prediction];positions=list(lengths);decode=[]
  for tick in range(7):
   prediction=list(decoder.session.run_leased(base,decoder.io.embedding([g[-1] for g in generated]),f32([v for p in positions for v in [p,p,p]],[4,3]),f32(positions,[4]),f32([1]*4,[4]),0,epochs))
   st=stats();assert st['misses']==0 and st['packed']==0
   assert st['residual_norm']==(32 if mode=='candidate' else 0) and st['packed_norm']==0;decode.append(st)
   for row in range(4):generated[row].append(prediction[row]);positions[row]+=1
  records.append(dict(id=case['id'],prompt_ids=prompts,frames=frames,head=head,prefill_diagnostic_seconds=seconds,after_prefill=after_prefill,generated_ids=generated,decode=decode,after_decode=witness()));save(False)
  print('norm_fusion_case',mode,case['id'],flush=True)
  del inputs
finally:
 if decoder is not None:decoder.close()
 garnet.shutdown();cleanup=json.loads(garnet.memory_stats_json())
 for k in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):assert cleanup[k]==0
 assert cleanup['metal_allocated_bytes']<=4*1024*1024
 save(len(records)==len(fixture['cases']))
assert len(records)==len(fixture['cases'])
