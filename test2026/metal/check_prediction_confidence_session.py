# SPDX-License-Identifier: Apache-2.0
"""Real packed Qwen2/Qwen3 confidence: exact greedy/KV/hidden, row epochs, shared heads.
Args LIB SOURCE MODEL OUTPUT. Correctness only, no stopping/performance claim.
"""
import sys,json,os,math
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet,tensor
from xModel.qwen.metal_dense.decoder import DenseMetalDecoder
with open(sys.argv[3]+'/config.json') as f:config=json.load(f)
width=config['hidden_size'];vocab=config['vocab_size'];layers=config['num_hidden_layers'];family=config['model_type'];assert family in ('qwen2','qwen3')
assert not os.path.exists(sys.argv[4]);decoder=None;reference=garnet.decoder_session();records=[];rejected=[];cleanup=None;completed=False
identity=[family+'-real-confidence','fp16g64','pinned-tokenizer','text_zero_based_3axis_v1','B2C64','text']
def val(v,shape):return tensor.tensor(v,shape=shape,dtype=tensor.float32)
def states(owner):return [[io.verification_f32_sha256(p) for p in owner.snapshot(i)] for i in range(layers)]
def control(ids,slots,flags,epochs):return [base,io.embedding(ids),val([float(p) for p in slots for a in range(3)],[2,3]),val([float(p) for p in slots],[2]),val([float(f) for f in flags],[2]),0,epochs,ids]
def reject(label,fn):
 before=states(session);failed=False
 try:fn()
 except Exception:failed=True
 assert failed and states(session)==before,label
 rejected.append(label)
def expected(hidden):
 logits=decoder.head.run([val(hidden,[2,width])]).tolist();wanted=[]
 for row in range(2):
  scores=logits[row*vocab:(row+1)*vocab];index=max(range(len(scores)),key=lambda i:scores[i]);best=scores[index];second=max(scores[:index]+scores[index+1:]);wanted.append([index,best,second,best-second])
 return wanted
try:
 decoder=DenseMetalDecoder(sys.argv[3]+'/model.safetensors',config,2,64);session=decoder.session;io=decoder.io;base=[p['graph'] for p in decoder.layers]
 reference.build(base,decoder.head)
 reject('unconfigured',lambda:session.predict_rows_confidence([0,0]))
 reject('before-token-prefix',lambda:session.configure_prediction_confidence())
 session.configure_token_prefix(identity,1,1024*1024);reference.configure_token_prefix(identity,1,1024*1024)
 assert session.configure_prediction_confidence();reject('repeated-configuration',lambda:session.configure_prediction_confidence())
 epochs=[session.claim_request_row(i+1)[1] for i in range(2)];ref_epochs=[reference.claim_request_row(i+1)[1] for i in range(2)]
 reject('uninitialized-row',lambda:session.predict_rows_confidence(epochs))
 slots=[0,0];last=[None,None]
 for index,(ids,flags) in enumerate([([1,2],[1,1]),([41,42],[1,1]),([151643,0],[1,0]),([0,151645],[0,1]),([300,1000],[1,1])]):
  values=session.run_tokens_confidence(*control(ids,slots,flags,epochs));ordinary=reference.run_tokens(*control(ids,slots,flags,ref_epochs))
  assert states(session)==states(reference),'confidence changed full KV'
  assert io.verification_f32_sha256(session.hidden())==io.verification_f32_sha256(reference.hidden()),'confidence changed hidden'
  for row in range(2):
   if flags[row]:last[row]=io.rows(session.hidden(),row,1).tolist()
  wanted=expected(last[0]+last[1])
  for row in range(2):
   assert values[row]==(wanted[row] if flags[row] else []),'confidence differs from full CPU logits'
   if flags[row]:assert values[row][0]==ordinary[row]
  # Another owner/standalone head has overwritten shared head storage. Fresh
  # prediction must use this session's private per-row hidden states again.
  fresh=session.predict_rows_confidence(epochs);plain=reference.predict_rows(ref_epochs)
  assert fresh==wanted and [p[0] for p in fresh]==plain
  assert states(session)==states(reference)
  records.append(dict(index=index,ids=ids,active=flags,confidence=values,expected=wanted,state=states(session),reference_state=states(reference)))
  for row in range(2):slots[row]+=flags[row]
  print('confidence_session_step',index,'PASS',flush=True)
 args=control([31,32],slots,[1,1],epochs)
 reject('wrong-prediction-index',lambda:session.run_tokens_confidence(*args[:5],1,*args[6:]))
 reject('wrong-rank',lambda:session.run_tokens_confidence(args[0],val([0.]*(2*width),[2,1,width]),*args[2:]))
 reject('stale-epoch',lambda:session.predict_rows_confidence([epochs[0]+100,epochs[1]]))
 reject('empty-selection',lambda:session.predict_rows_confidence([0,0]))
 reject('bad-epoch-type',lambda:session.predict_rows_confidence([True,epochs[1]]))
 query=[1,41,151643];assert session.reconcile_dense_tokens(0,epochs[0],query)==2
 reject('discarded-hidden',lambda:session.predict_rows_confidence(epochs))
 assert session.run_tokens_confidence(*control([151643,0],[2,0],[1,0],epochs))[0]
 # Row1 is preserved by replay of row0.
 assert session.predict_rows_confidence([0,epochs[1]])[1]==records[-1]['expected'][1]
 old_epoch=epochs[0];assert session.release_request_row(0,old_epoch)
 reject('released-epoch',lambda:session.predict_rows_confidence([old_epoch,0]))
 decoder.head.close();reject('closed-head',lambda:session.predict_rows_confidence([0,epochs[1]]))
 completed=True
finally:
 reference.close()
 if decoder is not None:decoder.close()
 garnet.shutdown();cleanup=json.loads(garnet.memory_stats_json())
 for k in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):assert cleanup[k]==0
 assert cleanup['metal_allocated_bytes']<=4*1024*1024
 with open(sys.argv[4],'w') as f:json.dump(dict(completed=completed,records=records,rejected=rejected,cleanup=cleanup,library=garnet.native_library_path(),scope=__doc__),f,indent=2)
assert completed;print('confidence session PASS',len(records),len(rejected),flush=True)
