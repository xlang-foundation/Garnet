"""Actual Mistral7B native functional check against frozen greedy token IDs.

This is not a serving/performance benchmark. Run after reference engine exits.
"""
# SPDX-License-Identifier: Apache-2.0
import sys,json,time,os
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet
from xModel.mistral.text_7b.metal_decoder import MistralMetalDecoder
with open(sys.argv[3]) as stream:reference=json.load(stream)
if os.path.exists(sys.argv[4]):raise ValueError('new evidence output required')
folder=reference['model'];batch=int(sys.argv[5]) if len(sys.argv)>5 else 1
short_first=int(sys.argv[6]) if len(sys.argv)>6 else reference['max_tokens']
with open(folder+'/config.json') as stream:config=json.load(stream)
tokenizer=garnet.tokenizer();tokenizer.open(folder)
tokens=tokenizer.encode(reference['rendered'])
assert tokens==reference['prompt_ids'],'native prompt IDs differ from reference'
decoder=None;records=[];started=time.monotonic();closed=None
def save(passed):
 with open(sys.argv[4],'w') as stream:json.dump(dict(passed=passed,backend=garnet.backend_info(),prompt_ids=tokens,reference=sys.argv[3],records=records,cleanup=closed,scope='Small functional text generation only; no serving/coding/performance admission'),stream,indent=2)
try:
 decoder=MistralMetalDecoder(folder+'/model.safetensors',config,batch,256,resident=True)
 assert decoder.prepare_prefill(16)
 prepared_profile_count=len(decoder.prefill_graphs)
 prepared=json.loads(garnet.memory_stats_json())
 print('mistral_native_ready',json.dumps(dict(initialization_seconds=time.monotonic()-started,prepared_memory=prepared)),flush=True)
 for chunk in (0,16):
  decoder.reset()
  if chunk:
   decoder.enable_prefill(chunk)
   hidden=[decoder.io.embedding([token]*batch) for token in tokens]
   positions=[[[i]*3 for i in range(len(tokens))] for row in range(batch)]
   prediction=decoder.prefill_sequence(hidden,positions)
  else:
   for i,token in enumerate(tokens):
    prediction=decoder.step([token]*batch,predict=i==len(tokens)-1)
    if i%8==0:print('mistral_prefill_progress',i+1,flush=True)
  last_hidden=decoder.session.hidden()
  if chunk:
   last_active=len(tokens)-sum(decoder.prefill_chunk_plan[:-1])-1
   last_hidden=decoder.io.visual_token(last_hidden,last_active)
  first_logits=decoder.head.run([last_hidden]).tolist()
  generated_rows=[[] for row in range(batch)];active=[1 for row in range(batch)]
  budgets=[short_first if row==0 else reference['max_tokens'] for row in range(batch)]
  for i in range(reference['max_tokens']):
   for row in range(batch):
    if active[row]:
     generated_rows[row].append(prediction[row])
     if prediction[row] in reference['stops'] or len(generated_rows[row])==budgets[row]:active[row]=0
   if not any(active):break
   if i+1<reference['max_tokens']:prediction=decoder.step([prediction[row] if active[row] else 0 for row in range(batch)],active=active)
   if i%8==0:print('mistral_generation_progress',chunk,i+1,flush=True)
  equal=all(generated_rows[row]==reference['generated_ids'][:budgets[row]] for row in range(batch))
  assert len(decoder.prefill_graphs)==prepared_profile_count,'request compiled an unprepared Mistral graph family'
  if chunk:assert decoder.prefill_compile_ms==0,'request compiled a Mistral prefill graph'
  records.append(dict(chunk=chunk,batch=batch,generated_ids=generated_rows[0],generated_rows=generated_rows,budgets=budgets,full_reference_token_match=equal,text=tokenizer.decode_ex(generated_rows[0],True),first_logits=first_logits,prepared_memory=prepared))
  save(False)
  print('mistral_functional_result',json.dumps(dict(chunk=chunk,batch=batch,full_reference_token_match=equal,tokens=[len(ids) for ids in generated_rows],text=records[-1]['text'])),flush=True)
finally:
 if decoder is not None:
  decoder.close();decoder.io.close()
 garnet.shutdown();closed=json.loads(garnet.memory_stats_json())
 for key in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):assert closed[key]==0,(key,closed[key])
 save(len(records)==2 and all(r['full_reference_token_match'] for r in records))
 print('mistral_native_closed',json.dumps(closed),flush=True)
assert len(records)==2 and all(r['full_reference_token_match'] for r in records),'Mistral greedy reference token mismatch'
