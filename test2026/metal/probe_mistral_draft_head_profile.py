# SPDX-License-Identifier: Apache-2.0
"""Check compact proposal head outputs against exact selected full-head logits.

Args LIB SOURCE ASSETS OUTPUT. Serial operator evidence, not full-model speed.
"""
import sys,json,os,math,time
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet,tensor
from xModel.qwen.metal_dense.layers import T,norm
lib,source,assets,output=sys.argv[1:];assert not os.path.exists(output)
assert garnet.native_library_path()==lib+'/libgarnet.dylib'
with open(assets+'/assets-manifest.json') as f:manifest=json.load(f)
with open(manifest['source']+'/config.json') as f:config=json.load(f)
graphs=[];records=[];cleanup=None;checked=0
try:
 for batch in (1,2):
  io=garnet.tensor_io();io.open(manifest['source']+'/model.safetensors')
  for name,checkpoint,weight in [('full',manifest['source'],'model.embed_tokens.weight')]+[(p['label'],p['root'],'lm_head.weight') for p in manifest['profiles']]:
   symbol=tensor.input('draft_head_'+name+'_'+str(batch),shape=[batch,1024],dtype=tensor.float32)
   @T.fusion(name='draft_head_profile_probe',boundary='required')
   def fn(hidden):return norm(hidden,'model.norm',config)*T.unary_op('qwen_text_head',weight_name=weight,bf16_weight=1)
   graph=garnet.compiled_graph();graph.build(fn(symbol),[symbol],checkpoint+'/model.safetensors');graphs.append(graph)
  group=graphs[-3:]
  for sample in range(3):
   values=[math.sin((i+1)*(.071+sample*.013))*(.01 if sample==0 else 1.) for i in range(batch*1024)]
   hidden=tensor.tensor(values,shape=[batch,1024],dtype=tensor.float32)
   full=group[0].run([hidden]).tolist();assert len(full)==batch*config['vocab_size']
   timings=[[],[],[]];expected_tokens=[]
   for index,profile in enumerate(manifest['profiles']):
    with open(profile['root']+'/output-token-map.json') as f:mapping=json.load(f)['token_ids']
    result=group[index+1].run([hidden]);compact=result.tolist();assert len(compact)==batch*len(mapping)
    expected=[full[row*config['vocab_size']+token] for row in range(batch) for token in mapping]
    assert compact==expected and all(math.isfinite(v) for v in compact),'compact logits differ from selected full head'
    raw=io.argmax(result);mapped=[mapping[i] for i in raw]
    oracle=[max(mapping,key=lambda token:full[row*config['vocab_size']+token]) for row in range(batch)]
    assert mapped==oracle,'output token map or stable argmax differs'
    checked+=len(compact);expected_tokens.append(mapped)
   for rep in range(-3,15):
    for order in range(3):
     index=(order+rep+3)%3
     if (rep+3)%2:index=2-index
     start=time.monotonic();group[index].run([hidden]);wall=time.monotonic()-start
     if rep>=0:timings[index].append(dict(native_ms=group[index].last_inference_ms,wall_seconds=wall))
   records.append(dict(batch=batch,sample=sample,words=batch*sum(p['head_rows'] for p in manifest['profiles']),mapped_tokens=expected_tokens,timings=timings))
   print('compact_head_probe',batch,sample,'PASS',flush=True)
  io.close()
finally:
 for graph in graphs:graph.close()
 garnet.shutdown();cleanup=json.loads(garnet.memory_stats_json())
 for k in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):assert cleanup[k]==0
 with open(output,'x') as f:json.dump(dict(completed=len(records)==6,checked_words=checked,records=records,cleanup=cleanup,library=garnet.native_library_path(),scope=__doc__),f,indent=2)
assert len(records)==6
