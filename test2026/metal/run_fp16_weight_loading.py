"""Compare native FP16 checkpoint loading to exactly expanded FP32 weights."""
# SPDX-License-Identifier: Apache-2.0
import json,sys
sys.path.insert(0,sys.argv[2])
import garnet,tensor
with open(sys.argv[1]) as stream:fixture=json.load(stream)
T=garnet.tensor();checks=0;words=0
for case in fixture['cases']:
 symbol=tensor.input('fp16_loader',shape=case['shape'],dtype=tensor.float32)
 @T.fusion(name='fp16_loader_norm',boundary='required')
 def forward(hidden):
  factory=T.unary_op
  return hidden*factory('qwen_text_norm',weight_name='fp16.weight',eps=1e-5)
 @T.fusion(name='fp32_loader_norm',boundary='required')
 def control(hidden):
  factory=T.unary_op
  return hidden*factory('qwen_text_norm',weight_name='fp32.weight',eps=1e-5)
 graph=garnet.compiled_graph();reference=garnet.compiled_graph()
 try:
  graph.build(forward(symbol),[symbol],case['checkpoint'])
  reference.build(control(symbol),[symbol],case['checkpoint'])
  inputs=garnet.tensor_io().read_f32(case['input'],case['shape'])
  actual=graph.run([inputs]).tolist();expected=reference.run([inputs]).tolist()
  assert actual==expected,'FP16 loading changed an exactly representable value'
  checks+=1;words+=len(actual)
 finally:graph.close();reference.close()
garnet.shutdown();memory=json.loads(garnet.memory_stats_json())
for key in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):
 assert memory[key]==0,(key,memory[key])
print('fp16_weight_loading',json.dumps(dict(passed=True,cases=checks,fp32_words=words,cleanup=memory)))
