# SPDX-License-Identifier: Apache-2.0
"""Real packed Qwen3 compact output head: input token domain and exact KV reuse.
Args: LIB SOURCE MODEL OUTPUT. B2/C64, ordinary FP32 decode for the oracle.
No performance or speculative coding-quality claim follows from this test.
"""
import sys,json,os
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet,tensor
from xModel.qwen.metal_dense.decoder import DenseMetalDecoder
with open(sys.argv[3]+'/config.json') as stream:config=json.load(stream)
assert not os.path.exists(sys.argv[4]);decoder=None;reference=garnet.decoder_session();records=[];rejected=[];cleanup=None;finished=False
capacity=64;batch=2;layers=28;heads=8;dim=128
identity=['qwen3-real-affine4','fp16g64','pinned-tokenizer','text_zero_based_3axis_v1','B2C64','text']
def val(values,shape):return tensor.tensor(values,shape=shape,dtype=tensor.float32)
def take(owner):return [owner.snapshot(i) for i in range(layers)]
def hashes(parts):return [[io.verification_f32_sha256(x) for x in layer] for layer in parts]
def step(owner,epochs,ids,slots,active):
 return owner.run_tokens(base,io.embedding(ids),val([float(p) for p in slots for axis in range(3)],[2,3]),
  val([float(p) for p in slots],[2]),val(active,[2]),0,epochs,ids)
def seed(owner,histories):
 owner.reset();leases=[owner.claim_request_row(i+1) for i in range(2)];epochs=[x[1] for x in leases];last=[None,None]
 for t in range(max([len(h) for h in histories]+[0])):
  active=[int(t<len(h)) for h in histories];ids=[h[t] if t<len(h) else 0 for h in histories]
  step(owner,epochs,ids,[t if a else 0 for a in active],active)
  for row in range(2):
   if active[row]:last[row]=io.rows(owner.hidden(),row,1)
 return epochs,last
try:
 decoder=DenseMetalDecoder(sys.argv[3]+'/model.safetensors',config,2,capacity);io=decoder.io;base=[x['graph'] for x in decoder.layers];session=decoder.session
 assert io.checkpoint_format=='mlx_qwen3_affine_int4_g64_fp16'
 with open(sys.argv[3]+'/output-token-map.json') as stream:output_ids=json.load(stream)['token_ids']
 assert config['tie_word_embeddings'] is False and len(output_ids)<151643
 assert max(io.argmax(decoder.head.run([io.embedding([151643,151645])])) )<len(output_ids)
 reference.build(base,decoder.head)
 session.configure_token_prefix(identity,1,1024*1024);reference.configure_token_prefix(identity,1,1024*1024)
 cases=[('high-input-ids',[],[151643,151645,151935]),('high-input-append',[151643,151645],[151643,151645,151935]),
  ('high-input-changed',[151643,151645,151935],[151643,151644]),('fresh',[],[41,42]),('exact',list(range(10,18)),list(range(10,18))),('append',list(range(10,18)),list(range(10,18))+[31,32]),
  ('changed',list(range(10,18)),[10,11,12,13,33,34]),('shorter',list(range(10,18)),[10,11,12,13]),('disjoint',list(range(10,18)),[41,42,43]),
  ('near-capacity',list(range(100,163)),list(range(100,160))+[300,301,302,303])]
 for label,old,query in cases:
  other=list(range(20,28));epochs,last=seed(session,[old,other]);seed(reference,[old,other]);before=take(session);unaffected=hashes(take(reference))
  other_prediction=session.predict_rows([0,epochs[1]])[1]
  common=0
  while common<min(len(old),len(query)) and old[common]==query[common]:common+=1
  expected=common-1 if common==len(query) and common<len(old) else common
  kept=session.reconcile_dense_tokens(0,epochs[0],query);assert kept==expected
  stats=json.loads(session.dense_reconcile_stats_json());assert stats['retained_tokens']==kept and stats['common_tokens']==common and stats['previous_tokens']==len(old)
  assert stats['host_snapshot_bytes']==0 and stats['replay_last_token']==(kept<common)
  wanted_bytes=(2*layers*heads*(len(old)-kept)*dim+config['hidden_size'])*4 if kept<len(old) else 0
  assert stats['zeroed_bytes']==wanted_bytes,(label,stats,wanted_bytes)
  after=take(session);witness=[]
  for layer in range(layers):
   entries=[]
   for part in range(2):
    values=io.rows(before[layer][part],0,1).tolist()
    for group in range(heads):
     for index in range((group*capacity+kept)*dim,(group*capacity+len(old))*dim):values[index]=0.
    wanted=val(values,[1,heads,capacity,dim]);actual=io.rows(after[layer][part],0,1)
    want_hash=io.verification_f32_sha256(wanted);actual_hash=io.verification_f32_sha256(actual);assert want_hash==actual_hash,(label,layer,part)
    survivor=io.verification_f32_sha256(io.rows(after[layer][part],1,1));assert survivor==io.verification_f32_sha256(io.rows(before[layer][part],1,1))
    entries.append(dict(actual=actual_hash,expected=want_hash,survivor=survivor))
   witness.append(entries)
  assert hashes(take(reference))==unaffected,'another session changed'
  assert session.predict_rows([0,epochs[1]])[1]==other_prediction
  if kept<len(old):
   for call in [lambda:session.predict_rows([epochs[0],0]),lambda:session.hidden(),lambda:session.cache_token_prefix(0,epochs[0])]:
    failed=False
    try:call()
    except Exception:failed=True
    assert failed,'discarded hidden exposed'
  for t in range(kept,len(query)):step(session,epochs,[query[t],0],[t,0],[1,0])
  ref_epochs,ref_last=seed(reference,[query,other]);assert hashes(take(session))==hashes(take(reference)),('continued KV differs',label)
  actual_prediction=session.predict_rows(epochs);wanted_prediction=reference.predict_rows(ref_epochs);assert actual_prediction==wanted_prediction
  if kept<len(query):assert io.verification_f32_sha256(io.rows(session.hidden(),0,1))==io.verification_f32_sha256(ref_last[0])
  assert json.loads(session.token_prefix_stats_json())['entries']==0
  records.append(dict(label=label,old=old,query=query,common=common,kept=kept,stats=stats,range_witness=witness,
   continued_state=hashes(take(session)),reference_state=hashes(take(reference)),prediction=actual_prediction,reference_prediction=wanted_prediction))
  print('dense_reconcile',label,'PASS',kept,stats['zeroed_bytes'],flush=True)
 epochs,last=seed(session,[list(range(10,18)),list(range(20,28))])
 invalid=[('empty',lambda:session.reconcile_dense_tokens(0,epochs[0],[])),('over-capacity',lambda:session.reconcile_dense_tokens(0,epochs[0],[1]*65)),
  ('negative',lambda:session.reconcile_dense_tokens(0,epochs[0],[-1])),('bool',lambda:session.reconcile_dense_tokens(0,epochs[0],[True])),
  ('vocabulary',lambda:session.reconcile_dense_tokens(0,epochs[0],[151936])),('overflow-token',lambda:session.reconcile_dense_tokens(0,epochs[0],[2**40])),
  ('wrong-row',lambda:session.reconcile_dense_tokens(2,epochs[0],[1])),('negative-row',lambda:session.reconcile_dense_tokens(-1,epochs[0],[1])),
  ('stale-epoch',lambda:session.reconcile_dense_tokens(0,epochs[0]+100,[1])),('wrong-type',lambda:session.reconcile_dense_tokens(0,epochs[0],'bad'))]
 for label,call in invalid:
  before=hashes(take(session));h=io.verification_f32_sha256(session.hidden());stats=session.dense_reconcile_stats_json();failed=False
  try:call()
  except Exception:failed=True
  assert failed and hashes(take(session))==before and io.verification_f32_sha256(session.hidden())==h and session.dense_reconcile_stats_json()==stats
  rejected.append(label)
 before=hashes(take(session));old_epoch=epochs[0];assert session.release_request_row(0,old_epoch)
 released=hashes(take(session));failed=False
 try:session.reconcile_dense_tokens(0,old_epoch,[1])
 except Exception:failed=True
 assert failed and hashes(take(session))==released;rejected.append('released-epoch')
 epochs,last=seed(session,[list(range(10,18)),list(range(20,28))])
 session.run(base,io.embedding([31,0]),val([8.,8.,8.,0.,0.,0.],[2,3]),val([8.,0.],[2]),val([1.,0.],[2]),0)
 untracked=hashes(take(session));failed=False
 try:session.reconcile_dense_tokens(0,epochs[0],[10,11])
 except Exception:failed=True
 assert failed and hashes(take(session))==untracked;rejected.append('untracked-state')
 epochs,last=seed(session,[list(range(10,18)),list(range(20,28))])
 decoder.layers[0]['graph'].close();before=hashes(take(session));failed=False
 try:session.reconcile_dense_tokens(0,epochs[0],[1])
 except Exception:failed=True
 assert failed and hashes(take(session))==before;rejected.append('closed-owner')
 finished=True
finally:
 reference.close()
 if decoder is not None:decoder.close()
 garnet.shutdown();cleanup=json.loads(garnet.memory_stats_json())
 for key in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):assert cleanup[key]==0
 assert cleanup['metal_allocated_bytes']<=4*1024*1024
 with open(sys.argv[4],'w') as stream:json.dump(dict(completed=finished,records=records,rejected=rejected,cleanup=cleanup,scope=__doc__),stream,indent=2)
assert finished
