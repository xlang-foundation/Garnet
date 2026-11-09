# SPDX-License-Identifier: Apache-2.0
"""Dense rewind rejects hybrid state; existing native hybrid snapshots still work.
Args: LIB SOURCE HYBRID_FIXTURE OUTPUT. Base steps only: no dense prefill claim.
"""
import sys,json,os
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet,tensor
from xModel.qwen35.vl_0_8b.decoder_layer import T,Qwen35LinearDecoderLayer,Qwen35FullDecoderLayer
with open(sys.argv[3]) as stream:f=json.load(stream)
c=f['config'];width=c['hidden_size'];base=[];head=None;session=garnet.decoder_session();io=garnet.tensor_io();passed=False
shapes_by_layer=[[[4,width],[4,29,4],[4,3,7,5],[4]],[[4,width],[4,1,8,6],[4,1,8,6],[4,3],[4],[4]]]
def val(values,shape):return tensor.tensor(values,shape=shape,dtype=tensor.float32)
def state():return [[io.verification_f32_sha256(v) for v in session.snapshot(i)] for i in range(2)]
def reject(epoch):
 before=state();failed=False
 try:session.reconcile_dense_tokens(0,epoch,[1,2])
 except Exception as error:failed='excludes recurrent' in str(error)
 assert failed and state()==before
try:
 for layer,shapes in enumerate(shapes_by_layer):
  symbols=[tensor.input('hybrid_guard_'+str(layer)+'_'+str(i),shape=s,dtype=tensor.float32) for i,s in enumerate(shapes)]
  if layer==0:
   @T.fusion(name='hybrid_guard_linear',boundary='required')
   def forward(h,a,b,active):return Qwen35LinearDecoderLayer(h,a,b,active,c,0)
  else:
   @T.fusion(name='hybrid_guard_full',boundary='required')
   def forward(h,a,b,p,s,active):return Qwen35FullDecoderLayer(h,a,b,p,s,active,c,1,8)
  graph=garnet.compiled_graph();graph.build(forward(*symbols),symbols,f['checkpoint']);base.append(graph)
 symbol=tensor.input('hybrid_guard_head',shape=[4,width],dtype=tensor.float32)
 @T.fusion(name='hybrid_guard_head',boundary='required')
 def project(h):return h*T.binary_op('add')*h
 head=garnet.compiled_graph();head.build(project(symbol),[symbol],f['checkpoint']);session.build(base,head)
 session.configure_token_prefix(['hybrid-fixture','F32','tokens','text_zero_based_3axis_v1','B4C8','text'],2,1024*1024)
 leases=[session.claim_request_row(i+1) for i in range(4)];epochs=[x[1] for x in leases];reject(epochs[0])
 prediction=session.run_tokens(base,val([.1]*(4*width),[4,width]),val([0.]*12,[4,3]),val([0.]*4,[4]),val([1.,0.,0.,0.],[4]),0,epochs,[1,0,0,0])
 before=state();reject(epochs[0]);assert session.cache_token_prefix(0,epochs[0])
 assert session.release_request_row(0,epochs[0]);epochs[0]=session.claim_request_row(999)[1]
 assert session.restore_token_prefix(0,epochs[0],[1,2])==1 and state()==before
 assert session.predict_rows([epochs[0],0,0,0])[0]==prediction[0];passed=True
finally:
 session.close()
 for graph in base:graph.close()
 if head is not None:head.close()
 io.close();garnet.shutdown();cleanup=json.loads(garnet.memory_stats_json())
 for key in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):assert cleanup[key]==0
 assert cleanup['metal_allocated_bytes']<=4*1024*1024
 with open(sys.argv[4],'w') as stream:json.dump(dict(completed=passed,hybrid_rejections=2,restored_state=before if passed else None,cleanup=cleanup,scope=__doc__),stream,indent=2)
assert passed;print('hybrid guard and existing snapshot restore PASS',flush=True)
