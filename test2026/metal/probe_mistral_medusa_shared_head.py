# SPDX-License-Identifier: Apache-2.0
"""Check residual-only mode, unchanged old modes, and shared four-row head."""
import sys,json,os,math
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet,tensor
from xModel.mistral.text_7b.metal_decoder import MistralMetalDecoder
from xModel.mistral.text_7b.medusa_shared_draft import SharedMedusaDraft
from xModel.qwen.metal_dense.layers import T
lib,source,model,aux,reference,output=sys.argv[1:];assert not os.path.exists(output)
assert garnet.native_library_path()==lib+'/libgarnet.dylib'
with open(model+'/config.json') as f:config=json.load(f)
decoder=None;draft=None;captures=[];plain=[];single=None;two=None;records=[];invalid=[];before=None;ready=None;cleanup=None
owners=('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets')
def snapshot():return json.loads(garnet.memory_stats_json())
def build(index,features,shape,peer=None):
 symbol=tensor.input('residual_test_'+str(index)+'_'+str(features),shape=shape,dtype=tensor.float32)
 @T.fusion(name='residual_test',boundary='required')
 def fn(hidden):return hidden*T.unary_op('medusa_residual_head',head_index=index,emit_features=features)
 graph=garnet.compiled_graph()
 try:
  if peer is None:graph.build(fn(symbol),[symbol],aux+'/model.safetensors')
  else:graph.build_shared(fn(symbol),[symbol],aux+'/model.safetensors',peer)
 except:graph.close();raise
 return graph
try:
 decoder=MistralMetalDecoder(model+'/model.safetensors',config,1,256,resident=True)
 before=snapshot();draft=SharedMedusaDraft(decoder,aux);ready=snapshot()
 assert ready['garnet_live_affine_weight_owners']==before['garnet_live_affine_weight_owners'],'duplicated target quantized weights'
 assert ready['garnet_live_bf16_weight_owners']==before['garnet_live_bf16_weight_owners']+3,'residual route loaded full vocabulary heads'
 for i in range(3):
  captures.append(build(i,1,[1,4096],draft.features[i]));plain.append(build(i,0,[1,4096],captures[i]))
 symbol=tensor.input('single_shared_head',shape=[1,4096],dtype=tensor.float32)
 @T.fusion(name='single_shared_head',boundary='required')
 def project(hidden):return hidden*T.unary_op('qwen_text_head',weight_name='lm_head.weight',bf16_weight=1)
 single=garnet.compiled_graph();single.build_shared(project(symbol),[symbol],decoder.checkpoint,decoder.head)
 two=build(0,2,[2,4096],draft.features[0])
 for label in ['synthetic-small','synthetic-normal','trained-eight-token-hidden']:
  with open(reference+'/'+label+'-input.json') as f:values=json.load(f)
  hidden=tensor.tensor(values,shape=[1,4096],dtype=tensor.float32)
  normalized,features,logits=draft.logits(hidden);n=normalized.tolist();packed=logits.tolist()
  with open(reference+'/'+label+'-normalized.json') as f:assert n==json.load(f),'changed RMS normalization'
  expected=[];old_tokens=[]
  for i in range(3):
   capture=captures[i].run([normalized]).tolist();raw=plain[i].run([normalized]);feature=features[i].tolist()
   assert feature==capture[:4096] and raw.tolist()==capture[4096:],'new mode changes old math'
   with open(reference+'/'+label+'-head'+str(i)+'.json') as f:old=json.load(f)
   assert old['features']==feature and old['logits']==raw.tolist(),'changed pinned old operator output'
   old_tokens.append(decoder.io.argmax(raw)[0])
  for value in [normalized]+features:expected.extend(single.run([value]).tolist())
  assert packed==expected,'packed target projection differs from four scalar projections'
  anchor=decoder.io.argmax(decoder.head.run([hidden]))[0];tokens=decoder.io.argmax(logits)
  assert tokens[0]==anchor and draft.propose(hidden,anchor)==tokens[1:],'known-token anchor changed'
  doubled=two.run([tensor.tensor(n+n,shape=[2,4096],dtype=tensor.float32)]).tolist()
  assert doubled==features[0].tolist()*2,'bounded two-row feature mode differs'
  records.append(dict(label=label,shared_tokens=tokens,raw_head_tokens=old_tokens,feature_words=3*4096,packed_head_words=len(packed)))
  print('shared_probe_case',label,tokens,flush=True)
 for index,mode,shape in [(-1,2,[1,4096]),(4,2,[1,4096]),(0,3,[1,4096]),(0,-1,[1,4096]),(0,2,[3,4096]),(0,2,[1,4095]),(0,2,[1,1,4096])]:
  prev=snapshot();bad=None;rejected=False
  try:bad=build(index,mode,shape)
  except Exception:rejected=True
  if bad is not None:bad.close()
  now=snapshot();assert rejected and all(prev[k]==now[k] for k in owners),'invalid profile or owner leak'
  invalid.append(dict(index=index,mode=mode,shape=shape))
finally:
 for graph in captures+plain:graph.close()
 if two is not None:two.close()
 if single is not None:single.close()
 if draft is not None:draft.close()
 if decoder is not None:decoder.close()
 garnet.shutdown();cleanup=snapshot();assert all(cleanup[k]==0 for k in owners)
 with open(output,'x') as f:json.dump(dict(completed=len(records)==3,records=records,invalid=invalid,before=before,prepared_shared=ready,cleanup=cleanup,backend=garnet.backend_info(),library=garnet.native_library_path(),scope=__doc__),f,indent=2)
assert len(records)==3 and len(invalid)==7
print('shared_head_reference PASS samples=3 packed_words=393216 residual_words=36864 invalid=7 owners=0',flush=True)
