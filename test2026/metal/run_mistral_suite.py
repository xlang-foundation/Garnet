# SPDX-License-Identifier: Apache-2.0
"""Native C++ tokenizer/Metal coding and context checks, not a speed benchmark.

XLang arguments: LIB SOURCE FIXTURE REFERENCE OUTPUT KIND CAPACITY CHUNK
KIND is coding or context. A zero chunk selects ordinary token-wise prefill.
"""
import sys, json, os, time
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2])
import garnet
from xModel.mistral.text_7b.metal_decoder import MistralMetalDecoder

fixture_path=sys.argv[3];reference_root=sys.argv[4];output=sys.argv[5]
kind=sys.argv[6];capacity=int(sys.argv[7]);chunk=int(sys.argv[8])
assert kind in ('coding','context') and capacity in (2048,4096) and chunk in (0,2,4,8,16)
with open(fixture_path) as stream:fixture=json.load(stream)
cases=[case for case in fixture['cases'] if case['kind']==kind]
assert cases and all(case['prompt_tokens']+case['max_tokens']<=capacity for case in cases)
assert not os.path.exists(output),'new evidence directory required'
os.mkdir(output)
folder=fixture['model']
with open(folder+'/config.json') as stream:config=json.load(stream)
tokenizer=garnet.tokenizer();tokenizer.open(folder)
for case in cases:assert tokenizer.encode(case['rendered'])==case['prompt_ids'],'native prompt IDs differ'
decoder=None;closed=None;records=[]
def save(completed):
 with open(output+'/status.json','w') as stream:
  json.dump(dict(completed=completed,scope='Bounded coding/context collection; quality and parity require separate audit',
    fixture=fixture_path,reference=reference_root,backend=garnet.backend_info(),capacity=capacity,
    batch=1,chunk=chunk,records=records,cleanup=closed),stream,indent=2)
try:
 began=time.monotonic()
 decoder=MistralMetalDecoder(folder+'/model.safetensors',config,1,capacity,resident=True)
 if chunk:decoder.prepare_prefill(chunk)
 prepared_count=len(decoder.prefill_graphs) if chunk else 0
 print('mistral_suite_ready',json.dumps(dict(capacity=capacity,chunk=chunk,
       initialization_seconds=time.monotonic()-began,memory=json.loads(garnet.memory_stats_json()))),flush=True)
 for case in cases:
  decoder.reset();tokens=case['prompt_ids'];started=time.monotonic()
  if chunk:
   hidden=[decoder.io.embedding([token]) for token in tokens]
   positions=[[[i]*3 for i in range(len(tokens))]]
   prediction=decoder.prefill_sequence(hidden,positions)
   assert decoder.prefill_compile_ms==0 and len(decoder.prefill_graphs)==prepared_count
  else:
   for i,token in enumerate(tokens):
    prediction=decoder.step([token],predict=i==len(tokens)-1)
    if (i+1)%128==0:print('mistral_suite_prefill',case['id'],i+1,flush=True)
  prefill_seconds=time.monotonic()-started
  last_hidden=decoder.session.hidden()
  if chunk:
   last_active=len(tokens)-sum(decoder.prefill_chunk_plan[:-1])-1
   last_hidden=decoder.io.visual_token(last_hidden,last_active)
  first_logits=decoder.head.run([last_hidden]).tolist()
  generated=[]
  for step in range(case['max_tokens']):
   generated.append(prediction[0])
   if prediction[0] in fixture['stops']:break
   if step+1<case['max_tokens']:prediction=decoder.step(prediction)
   if (step+1)%128==0:print('mistral_suite_decode',case['id'],step+1,flush=True)
  with open(reference_root+'/'+case['id']+'/request-0.result.json') as stream:reference=json.load(stream)
  common=0
  for a,b in zip(generated,reference['generated_ids']):
   if a!=b:break
   common+=1
  task_folder=output+'/'+case['id'];os.mkdir(task_folder)
  result=dict(status='completed',finish_reason='stop' if generated[-1] in fixture['stops'] else 'length',
    text=tokenizer.decode_ex(generated,True),generated_ids=generated,prompt_ids=tokens,
    elapsed_seconds=time.monotonic()-started,prefill_seconds=prefill_seconds,
    full_reference_token_match=generated==reference['generated_ids'],matching_prefix_tokens=common,
    memory=json.loads(garnet.memory_stats_json()),slots=decoder.slots,first_logits=first_logits)
  assert decoder.slots==[len(tokens)+len(generated)-1],'unexpected KV position after generation'
  with open(task_folder+'/request-0.result.json','w') as stream:json.dump(result,stream,indent=2)
  record=dict(id=case['id'],prompt_tokens=len(tokens),output_tokens=len(generated),
    finish_reason=result['finish_reason'],full_reference_token_match=result['full_reference_token_match'],
    matching_prefix_tokens=common)
  records.append(record);save(False);print('mistral_suite_case',json.dumps(record),flush=True)
finally:
 if decoder is not None:decoder.close()
 garnet.shutdown();closed=json.loads(garnet.memory_stats_json())
 for key in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):
  assert closed[key]==0,(key,closed[key])
 save(len(records)==len(cases))
 print('mistral_suite_closed',json.dumps(closed),flush=True)
assert len(records)==len(cases),'incomplete Mistral suite'
