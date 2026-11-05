# SPDX-License-Identifier: Apache-2.0
"""Serialized last-chunk GPU operator diagnostic, not production timing.

XLang args: LIB SOURCE FIXTURE PAIRED_CONTROL NEW_OUTPUT. Native PROFILE_OPS
inserts waits and disables several fusions. Durations identify work categories
under that diagnostic policy and must not be presented as a production trace.
"""
import sys,os,json
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet,tensor
from xModel.mistral.text_7b.metal_decoder import MistralMetalDecoder
with open(sys.argv[3]) as f:fixture=json.load(f)
with open(fixture['model']+'/config.json') as f:config=json.load(f)
with open(sys.argv[4]) as f:control=json.load(f)
assert control['completed'] and not os.path.exists(sys.argv[5]);decoder=None;records=[];closed=None;finished=False
prompt=next(c for c in fixture['cases'] if c['id']=='context-2048')['prompt_ids'][:2048]
def f32(v,s):return tensor.tensor(v,shape=s,dtype=tensor.float32)
try:
 os.environ['GARNET_METAL_PROFILE_OPS']='0';decoder=MistralMetalDecoder(fixture['model']+'/model.safetensors',config,4,4096,resident=True);decoder.prepare_compact_prefill(512)
 for chunk in (128,512):
  decoder.session.reset();epochs=[]
  for row in range(4):
   lease=decoder.session.claim_request_row(row+1);assert lease[0]==row;epochs.append(lease[1])
  for offset in range(0,2048,chunk):
   hidden=decoder.io.stack_tokens([decoder.io.embedding([prompt[offset+t],0,0,0]) for t in range(chunk)])
   slots=[offset+t if row==0 else 0 for row in range(4) for t in range(chunk)];active=[int(row==0) for row in range(4) for t in range(chunk)]
   last=offset+chunk==2048
   if last:print('diagnostic_begin',json.dumps(dict(chunk=chunk,offset=offset,serialized_diagnostic=True)),flush=True)
   os.environ['GARNET_METAL_PROFILE_OPS']='1' if last else '0'
   decoder.session.run_leased(decoder.prefill_profile_graphs(chunk),hidden,f32([s for s in slots for _ in range(3)],[4,chunk,3]),f32(slots,[4,chunk]),f32(active,[4,chunk]),-1,epochs)
   if last:print('diagnostic_end',chunk,flush=True)
  os.environ['GARNET_METAL_PROFILE_OPS']='0';predicted=decoder.session.predict_rows([epochs[0],0,0,0]);expected=next(v for v in control['groups'][0]['variants'] if v['chunk']==chunk)['predictions'];assert predicted==expected
  sample=json.loads(garnet.memory_stats_json());assert sample['metal_allocated_bytes']<=sample['metal_recommended_working_set_bytes']-256*1024*1024
  records.append(dict(chunk=chunk,predictions=predicted,expected_predictions=expected,memory=sample))
 finished=True
finally:
 os.environ['GARNET_METAL_PROFILE_OPS']='0'
 if decoder is not None:decoder.close()
 garnet.shutdown();closed=json.loads(garnet.memory_stats_json())
 for k in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):assert closed[k]==0
 assert closed['metal_allocated_bytes']<=4*1024*1024
 with open(sys.argv[5],'x') as f:json.dump(dict(completed=finished,records=records,cleanup=closed,scope=__doc__,library=garnet.native_library_path()),f,indent=2)
assert finished
