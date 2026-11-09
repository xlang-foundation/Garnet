# SPDX-License-Identifier: Apache-2.0
"""Native packed-Qwen2/Qwen3 IO and actual-weight projection checks; not timings.
Args: LIB SOURCE FIXTURE OUTPUT. CPU reference preparation runs separately.
"""
import sys,json,os,math
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet,tensor
with open(sys.argv[3]) as stream:fixture=json.load(stream)
assert not os.path.exists(sys.argv[4]);assert garnet.native_library_path()==sys.argv[1]+'/libgarnet.dylib'
io=garnet.tensor_io();T=garnet.tensor();records=[];embedding=None;cleanup=None
try:
 io.open(fixture['checkpoint']);assert io.checkpoint_format==fixture.get('checkpoint_format','mlx_qwen3_affine_int4_g64_fp16')
 embedding=io.embedding(fixture['embedding_ids']).tolist();assert embedding==fixture['embedding_expected']
 for row in (-1,fixture.get('vocabulary_size',151936)):
  rejected=False
  try:io.embedding([row])
  except Exception:rejected=True
  assert rejected
 for index,cell in enumerate(fixture['cells']):
  symbol=tensor.input('packed_qwen3_projection_'+str(index),shape=[cell['rows'],cell['inner']],dtype=tensor.float32)
  @T.fusion(name='packed_qwen3_reference_projection',boundary='required')
  def project(hidden):return hidden*T.unary_op('qwen_text_head',weight_name=cell['weight'],bf16_weight=1)
  graph=garnet.compiled_graph()
  try:
   graph.build(project(symbol),[symbol],fixture['checkpoint']);actual=graph.run([io.read_f32(cell['input'],[cell['rows'],cell['inner']])]).tolist()
   assert len(actual)==cell['rows']*cell['outputs'];sample=[];worst=0
   for row in range(cell['rows']):
    values=[]
    for j,col in enumerate(cell['columns']):
     value=actual[row*cell['outputs']+col];expected=cell['expected'][row][j];bound=2e-6+4e-6*cell['absolute'][row][j]
     ratio=abs(value-expected)/bound;worst=max(worst,ratio)
     assert math.isfinite(value) and ratio<=1,('FP64 projection mismatch',cell['weight'],cell['rows'],row,col,value,expected,bound)
     values.append(value)
    sample.append(values)
   records.append(dict(cell=index,weight=cell['weight'],rows=cell['rows'],outputs=cell['outputs'],actual=sample,worst_bound_ratio=worst))
   print('packed_projection',index,cell['weight'],cell['rows'],'PASS',flush=True)
  finally:graph.close()
finally:
 io.close();garnet.shutdown();cleanup=json.loads(garnet.memory_stats_json())
 for key in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):assert cleanup[key]==0
 assert cleanup['metal_allocated_bytes']<=4*1024*1024
 with open(sys.argv[4],'w') as stream:json.dump(dict(completed=len(records)==len(fixture['cells']),records=records,embedding=embedding,cleanup=cleanup,backend=garnet.backend_info(),library=garnet.native_library_path(),scope=__doc__),stream,indent=2)
assert len(records)==len(fixture['cells'])
