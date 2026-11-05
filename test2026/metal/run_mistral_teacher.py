# SPDX-License-Identifier: Apache-2.0
"""Native teacher-forced logits. XLang args: LIB SOURCE REFERENCE OUTPUT."""
import sys,json,os
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet
from xModel.mistral.text_7b.metal_decoder import MistralMetalDecoder
with open(sys.argv[3]) as stream:reference=json.load(stream)
assert not os.path.exists(sys.argv[4]),'new evidence output required'
folder=reference['model'];case=reference['case']
with open(folder+'/config.json') as stream:config=json.load(stream)
tokenizer=garnet.tokenizer();tokenizer.open(folder)
assert tokenizer.encode(case['rendered'])==case['prompt_ids']
decoder=None;records=[];closed=None
try:
 decoder=MistralMetalDecoder(folder+'/model.safetensors',config,1,4096,resident=True)
 decoder.prepare_prefill(16)
 for chunk in (0,16):
  decoder.reset();tokens=case['prompt_ids'];logits=[]
  if chunk:
   hidden=[decoder.io.embedding([token]) for token in tokens]
   decoder.prefill_sequence(hidden,[[[i]*3 for i in range(len(tokens))]],predict=False)
  else:
   for token in tokens:decoder.step([token],predict=False)
  for step in range(reference['steps']):
   hidden=decoder.session.hidden()
   if chunk and step==0:
    hidden=decoder.io.visual_token(hidden,len(tokens)-sum(decoder.prefill_chunk_plan[:-1])-1)
   logits.append(decoder.head.run([hidden]).tolist())
   if step+1<reference['steps']:decoder.step([reference['forced_ids'][step]],predict=False)
  records.append(dict(chunk=chunk,logits=logits,slots=decoder.slots))
  print('mistral_teacher_mode',chunk,reference['steps'],flush=True)
finally:
 if decoder is not None:decoder.close()
 garnet.shutdown();closed=json.loads(garnet.memory_stats_json())
 for key in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):
  assert closed[key]==0,(key,closed[key])
 with open(sys.argv[4],'w') as stream:
  json.dump(dict(completed=len(records)==2,reference=sys.argv[3],records=records,cleanup=closed,
     backend=garnet.backend_info(),scope='Teacher-forced logits diagnosis; no quality/performance admission'),stream,indent=2)
assert len(records)==2
