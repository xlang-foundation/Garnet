# SPDX-License-Identifier: Apache-2.0
"""Collect new-protocol Mistral text completions using unchanged native kernels.

Args LIB SOURCE FIXTURE NEW_OUTPUT. All eight original coding user messages,
1024/EOS2 budgets; explicit upstream v3 framing. Old baselines are not controls
for this changed protocol. Quality is evaluated later with the unchanged suite.
"""
import json,os,sys,time
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet
from xModel.mistral.text_7b.metal_decoder import MistralMetalDecoder
with open(sys.argv[3]) as f:fixture=json.load(f)
output=sys.argv[4];assert not os.path.exists(output);os.mkdir(output)
assert garnet.native_library_path()==sys.argv[1]+'/libgarnet.dylib'
assert fixture['protocol']=='mistral_v3_upstream_single_user_text_v1' and fixture['capacity']==4096 and fixture['stops']==[2]
with open(fixture['model']+'/config.json') as f:config=json.load(f)
tokenizer=garnet.tokenizer();tokenizer.open(fixture['model'])
for c in fixture['cases']:assert tokenizer.encode_ex(c['rendered'],False)==c['prompt_ids']
decoder=None;records=[];prepared=None;cleanup=None
def save(completed):
 with open(output+'/status.json','w') as f:json.dump(dict(completed=completed,library=garnet.native_library_path(),backend=garnet.backend_info(),fixture=sys.argv[3],records=records,prepared=prepared,cleanup=cleanup,scope=__doc__),f,indent=2)
try:
 decoder=MistralMetalDecoder(fixture['model']+'/model.safetensors',config,1,4096)
 decoder.prepare_prefill(128);preparation_compile_ms=decoder.prefill_compile_ms;decoder.prefill_compile_ms=0
 graphs=len(decoder.prefill_graphs);prepared=dict(graphs=graphs,prefill_preparation_compile_ms=preparation_compile_ms,memory=json.loads(garnet.memory_stats_json()))
 print('protocol_ready',prepared['memory']['metal_allocated_bytes'],flush=True)
 for case in fixture['cases']:
  decoder.reset();start=time.monotonic();ids=case['prompt_ids']
  predicted=decoder.prefill_sequence([decoder.io.embedding([token]) for token in ids],[[[i]*3 for i in range(len(ids))]])
  prefill=time.monotonic()-start;generated=[predicted[0]];began=time.monotonic()
  while generated[-1] not in fixture['stops'] and len(generated)<case['max_tokens']:
   generated.append(decoder.step([generated[-1]])[0])
  elapsed=time.monotonic()-began
  assert decoder.slots==[len(ids)+len(generated)-1] and decoder.prefill_compile_ms==0 and len(decoder.prefill_graphs)==graphs
  state=[[decoder.io.verification_f32_sha256(part) for part in decoder.session.snapshot(i)] for i in range(32)]
  hidden=decoder.io.verification_f32_sha256(decoder.session.hidden())
  result=dict(status='completed',finish_reason='stop' if generated[-1] in fixture['stops'] else 'length',text=tokenizer.decode_ex(generated,True),
   generated_ids=generated,prompt_ids=ids,prefill_seconds=prefill,decode_seconds=elapsed,slots=decoder.slots,
   state_sha256=state,hidden_sha256=hidden,memory=json.loads(garnet.memory_stats_json()),protocol=fixture['protocol'])
  os.mkdir(output+'/'+case['id'])
  with open(output+'/'+case['id']+'/request-0.result.json','w') as f:json.dump(result,f,indent=2)
  records.append(dict(case_id=case['id'],output_tokens=len(generated),finish_reason=result['finish_reason']));save(False)
  print('protocol_case',case['id'],len(generated),result['finish_reason'],flush=True)
finally:
 if decoder is not None:decoder.close()
 garnet.shutdown();cleanup=json.loads(garnet.memory_stats_json())
 for k in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):assert cleanup[k]==0
 assert cleanup['metal_allocated_bytes']<=4*1024*1024
 save(len(records)==len(fixture['cases']))
assert len(records)==8
