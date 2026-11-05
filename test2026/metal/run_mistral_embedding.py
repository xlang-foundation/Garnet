"""Exercise real native IO embeddings and reject unimplemented FP16 kernels."""
# SPDX-License-Identifier: Apache-2.0
import sys,json
sys.path.insert(0,sys.argv[1])
import garnet,tensor
with open(sys.argv[2]) as stream:fixture=json.load(stream)
io=garnet.tensor_io();checks=0;words=0
try:
 assert io.open(fixture['checkpoint'])
 assert io.checkpoint_format=='mlx_mistral_affine_int4_g64_fp16'
 for count in (1,2,4,8):
  rows=fixture['rows'][:count];actual=io.embedding(rows).tolist()
  expected=io.read_f32(fixture['expected'],[len(fixture['rows']),fixture['width']]).tolist()[:count*fixture['width']]
  assert actual==expected,'native IO embedding differs from MLX FP16 dequantization'
  checks+=1;words+=len(actual)
 for row in (-1,fixture['vocab_size']):
  failed=False
  try:io.embedding([row])
  except Exception:failed=True
  assert failed;checks+=1
 T=garnet.tensor();symbol=tensor.input('fp16_projection_rejection',shape=[1,fixture['width']],dtype=tensor.float32)
 @T.fusion(name='fp16_projection_rejection',boundary='required')
 def head(hidden):
  factory=T.unary_op
  return hidden*factory('qwen_text_head',weight_name='lm_head.weight',bf16_weight=1)
 graph=garnet.compiled_graph()
 try:
  failed=False
  try:graph.build(head(symbol),[symbol],fixture['checkpoint'])
  except Exception as error:
   failed='FP16 kernel profile is required' in str(error)
  assert failed,'FP16 coefficients reached the BF16 Metal engine'
  checks+=1
 finally:graph.close()
finally:io.close()
garnet.shutdown();memory=json.loads(garnet.memory_stats_json())
for key in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):assert memory[key]==0,(key,memory[key])
print('mistral_embedding',json.dumps(dict(passed=True,checks=checks,fp32_words=words,cleanup=memory)))
