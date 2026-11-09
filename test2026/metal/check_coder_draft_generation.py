# SPDX-License-Identifier: Apache-2.0
"""Standalone native packed coder draft ordinary/chunk16 greedy observations.

Args LIB SOURCE MLX_REFERENCE OUTPUT. FP32 native decode/state is compared
with the saved MLX FP16-state reference; retain divergences, never force IDs.
Short64-token limits do not establish coding quality or whole-target speed.
"""
import json,os,sys
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet
from xModel.qwen.metal_dense.decoder import DenseMetalDecoder
with open(sys.argv[3]) as f:reference=json.load(f)
inputs=reference['inputs'];output=sys.argv[4];assert not os.path.exists(output)
with open(inputs['model']+'/config.json') as f:config=json.load(f)
decoder=None;rows=[];cleanup=None;completed=False;prepared=None
tokenizer=garnet.tokenizer();tokenizer.open(inputs['model'])
try:
 decoder=DenseMetalDecoder(inputs['model']+'/model.safetensors',config,1,512)
 assert decoder.io.checkpoint_format=='mlx_qwen2_affine_int4_g64_fp16'
 decoder.enable_prefill(16)
 decoder.prefill_compile_ms=0
 for width in (2,4,8,16):decoder._resident_prefill_graphs(width)
 decoder.prefill_compile_ms=0;graphs=len(decoder.prefill_graphs);prepared=json.loads(garnet.memory_stats_json())
 for case,witness in zip(inputs['cases'],reference['records']):
  assert case['id']==witness['case_id'];ids=tokenizer.encode_ex(case['rendered'],False);assert ids==witness['prompt_ids']
  for chunk in (0,16):
   decoder.reset()
   if chunk:
    prediction=decoder.prefill_sequence([decoder.io.embedding([t]) for t in ids],[[[i]*3 for i in range(len(ids))]])
   else:
    for i,token in enumerate(ids):prediction=decoder.step([token],predict=i+1==len(ids))
   generated=[prediction[0]]
   while generated[-1] not in inputs['stops'] and len(generated)<case['max_tokens']:generated.append(decoder.step([generated[-1]])[0])
   assert decoder.slots==[len(ids)+len(generated)-1] and decoder.prefill_compile_ms==0 and len(decoder.prefill_graphs)==graphs
   common=0
   for a,b in zip(generated,witness['generated_ids']):
    if a!=b:break
    common+=1
   rows.append(dict(case_id=case['id'],chunk=chunk,prompt_ids=ids,generated_ids=generated,text=tokenizer.decode_ex(generated,True),exact_mlx_ids=generated==witness['generated_ids'],common_prefix_tokens=common,
    slots=decoder.slots,state_sha256=[[decoder.io.verification_f32_sha256(p) for p in decoder.session.snapshot(i)] for i in range(config['num_hidden_layers'])]))
   print('native_coder_draft',case['id'],chunk,len(generated),'MLX_equal',generated==witness['generated_ids'],'prefix',common,flush=True)
 completed=True
finally:
 if decoder is not None:decoder.close()
 garnet.shutdown();cleanup=json.loads(garnet.memory_stats_json())
 for key in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):assert cleanup[key]==0
 assert cleanup['metal_allocated_bytes']<=4*1024*1024
 with open(output,'w') as f:json.dump(dict(completed=completed,records=rows,prepared=prepared,cleanup=cleanup,scope=__doc__,library=garnet.native_library_path()),f,indent=2)
assert completed and len(rows)==16
