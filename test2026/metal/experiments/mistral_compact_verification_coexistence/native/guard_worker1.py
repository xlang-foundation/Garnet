"""Exercise default-off combination guards and enabled preparation lifetime bounds."""
import sys,json,os
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet
from xModel.mistral.text_7b.metal_decoder import MistralMetalDecoder
with open(sys.argv[3]) as f:inputs=json.load(f)
output=sys.argv[4];mode=sys.argv[5];assert mode in ('off-forward','off-reverse','on-owned','on-capacity') and not os.path.exists(output)
with open(inputs['target_model']+'/config.json') as f:config=json.load(f)
decoder=None;report=dict(completed=False,mode=mode,rejections=[],scope=__doc__)
def state():
 result=[]
 for layer in range(32):
  pair=decoder.session.snapshot(layer);result.append([[decoder.io.verification_f32_sha256(decoder.io.rows(p,row,1)) for p in pair] for row in range(4)])
 hidden=decoder.session.diagnostic_final_hidden();return dict(kv=result,hidden=[decoder.io.verification_f32_sha256(decoder.io.rows(hidden,row,1)) for row in range(4)])
def reject(label,fn):
 before=state();failed=False
 try:fn()
 except Exception as error:failed=True;message=str(error)
 assert failed
 after=state();assert before==after;report['rejections'].append(dict(label=label,error=message,before=before,after=after))
try:
 enabled='native_compact_verify_b4t1_optional_v1' in garnet.backend_info();assert enabled==mode.startswith('on-')
 capacity=2048 if mode=='on-capacity' else 4096
 decoder=MistralMetalDecoder(inputs['target_model']+'/model.safetensors',config,4,capacity);decoder.prepare_prefill(128)
 if mode=='off-forward':
  decoder.prepare_compact_prefill(128);reject('verification_after_compact_default_off',lambda:decoder.prepare_verification(1))
 elif mode=='off-reverse':
  decoder.prepare_verification(1);reject('compact_after_verification_default_off',lambda:decoder.prepare_compact_prefill(128))
 elif mode=='on-capacity':
  decoder.prepare_compact_prefill(128);reject('unsupported_capacity',lambda:decoder.prepare_verification(1))
 else:
  decoder.prepare_compact_prefill(512);decoder.prepare_verification(1);lease=decoder.session.claim_request_row(1)
  reject('late_verification_after_claim',lambda:decoder.session.prepare_verification(decoder.verification_graphs,decoder.verification_head))
  graphs=[decoder.compact_prefill_graphs['1-128-'+str(i)] for i in range(32)];fallback=decoder.prefill_profile_graphs(128)
  reject('late_compact_after_claim',lambda:decoder.session.prepare_compact_prefill(graphs,fallback));assert decoder.session.release_request_row(lease[0],lease[1])
 report['completed']=True
finally:
 if decoder is not None:decoder.close()
 garnet.shutdown();report['cleanup']=json.loads(garnet.memory_stats_json())
 for k in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):assert report['cleanup'][k]==0
 assert report['cleanup']['metal_allocated_bytes']<=4194304
 with open(output,'w') as f:json.dump(report,f,indent=2)
assert report['completed'];print('compact_verification_guards PASS',mode,len(report['rejections']),flush=True)
