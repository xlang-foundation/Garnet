# SPDX-License-Identifier: Apache-2.0
"""Save native Medusa operator outputs for an independent offline FP64 oracle.

Args: MODEL MEDUSA LIB SOURCE FIXTURE NEW_OUTPUT. No serving/speed, trained
draft acceptance or target compatibility claim follows from this operator test.
"""
import json,sys,os,math
sys.path.insert(0,sys.argv[3]);sys.path.insert(0,sys.argv[4])
import garnet,tensor
from xModel.mistral.text_7b.metal_decoder import MistralMetalDecoder
from xModel.mistral.text_7b.medusa_draft import MistralMedusaDraft
from xModel.qwen.metal_dense.layers import T
model,aux,output=sys.argv[1],sys.argv[2],sys.argv[6]
assert garnet.native_library_path()==sys.argv[3]+'/libgarnet.dylib'
assert not os.path.exists(output);os.mkdir(output)
with open(model+'/config.json') as stream:config=json.load(stream)
with open(sys.argv[5]) as stream:fixture=json.load(stream)
decoder=None;draft=None;captures=[];records=[];closed=None
def write(name,value):
 with open(output+'/'+name,'x') as stream:json.dump(value,stream)
try:
 decoder=MistralMetalDecoder(model+'/model.safetensors',config,1,256,resident=True)
 draft=MistralMedusaDraft(decoder,aux,3)
 for index in range(3):
  symbol=tensor.input('medusa_capture_'+str(index),shape=[1,4096],dtype=tensor.float32)
  @T.fusion(name='medusa_capture_'+str(index),boundary='required')
  def capture(hidden):return hidden*T.unary_op('medusa_residual_head',head_index=index,emit_features=1)
  graph=garnet.compiled_graph();captures.append(graph)
  graph.build_shared(capture(symbol),[symbol],aux+'/model.safetensors',draft.heads[index])
 owners=('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets')
 invalid=[]
 for label,shape,index,features,checkpoint in [
  ('negative-head',[1,4096],-1,0,aux+'/model.safetensors'),('head-four',[1,4096],4,0,aux+'/model.safetensors'),
  ('wrong-width',[1,4095],0,0,aux+'/model.safetensors'),('wrong-rank',[1,1,4096],0,0,aux+'/model.safetensors'),
  ('oversized-batch',[3,4096],0,0,aux+'/model.safetensors'),('wrong-feature-mode',[1,4096],0,2,aux+'/model.safetensors'),
  ('target-as-draft',[1,4096],0,0,model+'/model.safetensors')]:
  before=json.loads(garnet.memory_stats_json());bad=garnet.compiled_graph();rejected=False
  symbol=tensor.input('bad_medusa',shape=shape,dtype=tensor.float32)
  @T.fusion(name='bad_medusa',boundary='required')
  def project(hidden):return hidden*T.unary_op('medusa_residual_head',head_index=index,emit_features=features)
  try:bad.build(project(symbol),[symbol],checkpoint)
  except Exception:rejected=True
  bad.close();after=json.loads(garnet.memory_stats_json())
  assert rejected and all(before[key]==after[key] for key in owners),('invalid Medusa profile/owners',label)
  invalid.append(label)
 prepared=json.loads(garnet.memory_stats_json())
 assert prepared['metal_allocated_bytes']<=prepared['metal_recommended_working_set_bytes']-256*1024*1024
 samples=[('synthetic-small',[.01*math.sin(i*.17) for i in range(4096)]),
          ('synthetic-normal',[math.sin(i*.13)+.3*math.cos(i*.23) for i in range(4096)])]
 for token in fixture['cases'][0]['prompt_ids'][:8]:decoder.step([token],active=[1])
 samples.append(('trained-eight-token-hidden',decoder.session.hidden().tolist()))
 for label,values in samples:
  hidden=tensor.tensor(values,shape=[1,4096],dtype=tensor.float32)
  normalized,logits=draft.logits(hidden)
  write(label+'-input.json',hidden.tolist());write(label+'-normalized.json',normalized.tolist())
  predictions=[]
  for index in range(3):
   actual=logits[index].tolist();combined=captures[index].run([normalized]).tolist()
   assert len(actual)==32768 and len(combined)==4096+32768 and combined[4096:]==actual,'capture changed head logits'
   write(label+'-head'+str(index)+'.json',dict(features=combined[:4096],logits=actual))
   predictions.append(decoder.io.argmax(logits[index])[0])
  assert draft.propose(hidden)==[predictions]
  records.append(dict(label=label,predictions=predictions));print('medusa_probe_case',label,predictions,flush=True)
finally:
 for graph in captures:graph.close()
 if draft is not None:draft.close()
 if decoder is not None:decoder.close()
 garnet.shutdown();closed=json.loads(garnet.memory_stats_json())
 for key in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):assert closed[key]==0,(key,closed[key])
 write('status.json',dict(completed=len(records)==3,records=records,prepared=prepared if len(records) else None,
  cleanup=closed,invalid_profiles_rejected=invalid if len(records) else [],backend=garnet.backend_info(),scope=__doc__))
assert len(records)==3
