"""Native greedy accept/commit vs ordinary decode, including shared-bank swaps."""
import json,sys
sys.path.insert(0,sys.argv[2]);sys.path.insert(0,sys.argv[3])
import garnet,tensor
assert 'native_greedy_verification' in garnet.backend_info()
from xModel.qwen35.vl_0_8b.decoder_layer import T,norm,Qwen35LinearDecoderLayer,Qwen35FullDecoderLayer
from xModel.qwen35.vl_0_8b.prefill_layer import PrefillLayer
with open(sys.argv[1]) as stream:f=json.load(stream)
c=f['config'];w=c['hidden_size'];vocab=c['vocab_size'];io=garnet.tensor_io();io.open(f['checkpoint'])
checks=0;cases=0;coverage=set()
def val(data,shape):return tensor.tensor(data,shape=shape,dtype=tensor.float32)
def symbols(shapes,name):return [tensor.input(name+str(i),shape=s,dtype=tensor.float32) for i,s in enumerate(shapes)]
def state(owner):return [[io.verification_f32_sha256(x) for x in owner.snapshot(i)] for i in range(4)]
def hidden(owner):return io.verification_f32_sha256(owner.hidden())
def equal(a,b,label):
 global checks
 assert a==b,(label,a,b)
 checks+=1
def reject(call,label):
 global checks
 failed=False
 try:call()
 except Exception as error:failed=True;message=str(error)
 assert failed,'Missing rejection: '+label
 checks+=1
 return message
for batch,chunk,capacity in [(1,1,16),(1,2,16),(1,3,16),(1,4,16),(2,2,16),(4,1,16),(1,4,4096)]:
 base=[];packed=[];heads=[];session=garnet.decoder_session();reference=garnet.decoder_session();other=garnet.decoder_session()
 label='B'+str(batch)+'T'+str(chunk)+'C'+str(capacity)
 try:
  for i in range(4):
   states=([[batch,320,4],[batch,4,128,16]] if i%2==0 else [[batch,2,capacity,32]]*2)
   shapes=[[batch,w]]+states+([[batch]] if i%2==0 else [[batch,3],[batch],[batch]])
   inputs=symbols(shapes,'greedy_base_'+label+'_'+str(i)+'_')
   if i%2==0:
    @T.fusion(name='greedy_base_'+label+'_'+str(i),boundary='required')
    def forward(h,a,b,active):return Qwen35LinearDecoderLayer(h,a,b,active,c,i)
   else:
    @T.fusion(name='greedy_base_'+label+'_'+str(i),boundary='required')
    def forward(h,a,b,positions,slots,active):return Qwen35FullDecoderLayer(h,a,b,positions,slots,active,c,i,capacity)
   graph=garnet.compiled_graph();graph.build_bf16(forward(*inputs),inputs,f['checkpoint']);base.append(graph)
   inputs=symbols([[batch,chunk,w]]+states+[[batch,chunk,3],[batch,chunk],[batch,chunk]],'greedy_packed_'+label+'_'+str(i)+'_')
   @T.fusion(name='greedy_packed_'+label+'_'+str(i),boundary='required')
   def verify(h,a,b,positions,slots,active):
    return PrefillLayer(h,a,b,positions,slots,active,c,i,batch,chunk,capacity,'linear_attention' if i%2==0 else 'full_attention',parallel_attention=True,parallel_linear=True,attention_capacity_limit=4096)
   graph=garnet.compiled_graph();graph.build_verification_shared(verify(*inputs),inputs,f['checkpoint'],base[-1]);packed.append(graph)
  for rows in (batch,batch*chunk):
   inputs=symbols([[rows,w]],'greedy_head_'+label+'_'+str(rows)+'_')
   @T.fusion(name='greedy_head_'+label+'_'+str(rows),boundary='required')
   def head(h):return norm(h,'model.language_model.norm',c)*T.unary_op('qwen_text_head',weight_name='lm_head.weight',bf16_weight=1)
   graph=garnet.compiled_graph()
   if heads:graph.build_shared(head(*inputs),inputs,f['checkpoint'],heads[0])
   else:graph.build(head(*inputs),inputs,f['checkpoint'])
   heads.append(graph)
  session.build(base,heads[0]);reference.build(base,heads[0]);other.build(base,heads[0]);session.prepare_verification(packed,heads[1])
  def step(owner,ids,slots,flags):
   return owner.run(base,io.embedding(ids),val([s for s in slots for axis in range(3)],[batch,3]),val(slots,[batch]),val(flags,[batch]),0)
  def seed(owner):
   owner.reset();start=capacity-3-chunk if capacity==4096 else 0
   for k in range(3):step(owner,[(k*17+row+3)%vocab for row in range(batch)],[float(start+k)]*batch,[1.]*batch)
   return start+3
  # Verify immediately, before any ordinary target prediction initializes the
  # session's prediction bank (the trained factory uses predict-false prefill).
  initial_ids=[(11+row+t)%vocab for row in range(batch) for t in range(chunk)]
  initial_hidden=io.stack_tokens([io.embedding([initial_ids[row*chunk+t] for row in range(batch)]) for t in range(chunk)])
  initial_slots=[float(t) for row in range(batch) for t in range(chunk)]
  initial_flags=[1.]+[0.]*(batch*chunk-1)
  expected_first=step(reference,[initial_ids[row*chunk] for row in range(batch)],[0.]*batch,[1.]+[0.]*(batch-1))
  first=session.verify_greedy(initial_hidden,val([s for s in initial_slots for axis in range(3)],[batch,chunk,3]),val(initial_slots,[batch,chunk]),val(initial_flags,[batch,chunk]),initial_ids,[],[chunk]*batch,None)
  equal(first[0],[[expected_first[0]],1,False],label+' fresh prediction bank')
  equal(state(session),state(reference),label+' fresh state parity');cases+=1
  for length_mode in range(chunk+1):
   for requested in range(1,chunk+1):
    for eos_position in (-1,0,chunk-1):
     start=seed(reference);seed(session)
     lengths=[(length_mode+row)%(chunk+1) for row in range(batch)]
     known=[(7+requested*9+row*13)%vocab for row in range(batch)];ideal=[];oracle=[];current=list(known)
     for t in range(chunk):
      ideal.append(list(current));current=step(reference,current,[float(start+t)]*batch,[1.]*batch);oracle.append(list(current))
     proposals=[[ideal[t][row] for t in range(chunk)] for row in range(batch)]
     for row in range(batch):
      if requested<lengths[row]:proposals[row][requested]=(proposals[row][requested]+1)%vocab
     stops=[] if eos_position<0 else [oracle[min(eos_position,chunk-1)][0]]
     # Independent ordinary greedy decision, then consume EXACTLY that prefix.
     expected=[];counts=[];ended=[]
     for row in range(batch):
      emitted=[];stop=False
      for t in range(lengths[row]):
       actual=oracle[t][row];emitted.append(actual)
       if actual in stops:stop=True;break
       if t+1==lengths[row] or actual!=proposals[row][t+1]:break
      expected.append(emitted);counts.append(len(emitted));ended.append(stop)
      if lengths[row]:coverage.add(('eos' if stop else 'prefix')+str(len(emitted)))
     seed(reference)
     for t in range(max(counts)):
      step(reference,[proposals[row][t] for row in range(batch)],[float(start+t)]*batch,[float(t<counts[row]) for row in range(batch)])
     # Rotate the source graph's output banks through another private session.
     # A stale retained S1 pointer would now corrupt that other session's state.
     seed(other);step(other,[(31+row)%vocab for row in range(batch)],[float(start)]*batch,[1.]*batch)
     untouched=state(other)
     ids=[proposals[row][t] for row in range(batch) for t in range(chunk)]
     packed_hidden=io.stack_tokens([io.embedding([proposals[row][t] for row in range(batch)]) for t in range(chunk)])
     slots=[float(start+t) for row in range(batch) for t in range(chunk)]
     flags=[float(t<lengths[row]) for row in range(batch) for t in range(chunk)]
     result=session.verify_greedy(packed_hidden,val([s for s in slots for axis in range(3)],[batch,chunk,3]),val(slots,[batch,chunk]),val(flags,[batch,chunk]),ids,stops,[chunk]*batch,None)
     equal(result,[[expected[row],counts[row],ended[row]] for row in range(batch)],label+' emitted/consumed/EOS')
     equal(state(session),state(reference),label+' all Conv/GDN/KV hashes')
     equal(state(other),untouched,label+' other session unchanged')
     # Ordinary Hidden() returns the last physical step for every row, whereas
     # verified Hidden() retains each row's last committed hidden. Build the
     # independent row-wise oracle with a fresh ordinary pass.
     seed(reference);wanted_hidden=[0.]*(batch*w)
     # Include the seeded last hidden for rows with no submitted work.
     wanted_hidden=list(reference.hidden().tolist())
     for t in range(max(counts)):
      step(reference,[proposals[row][t] for row in range(batch)],[float(start+t)]*batch,[float(t<counts[row]) for row in range(batch)])
      h=reference.hidden().tolist()
      for row in range(batch):
       if t<counts[row]:wanted_hidden[row*w:(row+1)*w]=h[row*w:(row+1)*w]
     equal(hidden(session),io.verification_f32_sha256(val(wanted_hidden,[batch,w])),label+' committed hidden')
     stats=json.loads(session.verification_stats_json())
     equal(stats['target_conv_checkpoints'],2,label+' capture conv')
     equal(stats['target_delta_checkpoints'],2,label+' capture delta')
     equal(stats['target_kv_backups'],4,label+' cache backups')
     equal(stats['target_attention_dispatches'],2,label+' attention')
     equal(stats['target_affine_misses'],0,label+' prepared kernels')
     equal(stats['borrowed_s1_banks'],4,label+' borrowed S1 banks')
     if any(counts[row]<lengths[row] for row in range(batch)):
      equal(stats['commit_replay_layers'],2,label+' state-only replay')
      equal(stats['commit_kv_restores'],4,label+' rejected KV restore')
     if max(counts)<chunk and capacity!=4096:
      # Continue ordinary decode to prove accepted state can be used immediately.
      next_ids=[expected[row][-1] if counts[row] else known[row] for row in range(batch)]
      actual=step(session,next_ids,[float(start+counts[row]) for row in range(batch)],[1.]*batch)
      wanted=step(reference,next_ids,[float(start+counts[row]) for row in range(batch)],[1.]*batch)
      equal(actual,wanted,label+' continued predictions');equal(state(session),state(reference),label+' continued all states')
     cases+=1
  before=state(session);hbefore=hidden(session)
  reject(lambda:session.verify_greedy(packed_hidden,val([0.]*(batch*chunk*3),[batch,chunk,3]),val([0.]*(batch*chunk),[batch,chunk]),val([1.]*(batch*chunk),[batch,chunk]),ids,[],[0]*batch,None),label+' budget')
  equal(state(session),before,label+' invalid budget unchanged');equal(hidden(session),hbefore,label+' invalid hidden unchanged')
  if 'native_request_row_leases' in garnet.backend_info():
   session.reset()
   session.configure_token_prefix(['synthetic-target','affine4g64','fixture-tokenizer','text_zero_based_3axis_v1',label,'text'],4,8*1024*1024)
   leases=[session.claim_request_row(100+row) for row in range(batch)];epochs=[x[1] for x in leases]
   flags=val([1.]+[0.]*(batch*chunk-1),[batch,chunk]);sl=val([float(t) for row in range(batch) for t in range(chunk)],[batch,chunk]);pos=val([float(t) for row in range(batch) for t in range(chunk) for axis in range(3)],[batch,chunk,3])
   args=[packed_hidden,pos,sl,flags,ids,[],[chunk]*batch]
   before=state(session);bad=list(epochs);bad[0]+=100
   reject(lambda:session.verify_greedy(*args,bad),label+' stale lease');equal(state(session),before,label+' stale lease unchanged')
   reject(lambda:session.verify_greedy(*args,None),label+' missing lease');equal(state(session),before,label+' missing lease unchanged')
   if batch>1:
    bad=list(epochs);bad[-1]+=100
    reject(lambda:session.verify_greedy(*args,bad),label+' stale inactive lease');equal(state(session),before,label+' inactive stale unchanged')
   # Derive a fresh greedy chain at history length zero, then corrupt draft1.
   reference.reset();known=[9+row for row in range(batch)];proposals=[];current=list(known)
   for t in range(chunk):
    proposals.append(list(current));current=step(reference,current,[float(t)]*batch,[1.]*batch)
    if t==0:first=list(current)
   if chunk>1:proposals[1][0]=(first[0]+1)%vocab
   ids=[proposals[t][row] for row in range(batch) for t in range(chunk)]
   ph=io.stack_tokens([io.embedding(proposals[t]) for t in range(chunk)])
   flags=val([1.]*chunk+[0.]*(batch*chunk-chunk),[batch,chunk]);args=[ph,pos,sl,flags,ids,[],[chunk]*batch]
   result=session.verify_greedy(*args,epochs)
   equal(result[0],[[first[0]],1,False],label+' tracked rejection result')
   assert session.cache_token_prefix(0,epochs[0])
   assert session.release_request_row(0,epochs[0]);renewed=session.claim_request_row(999);equal(renewed[0],0,label+' reused row')
   epochs[0]=renewed[1]
   query=[proposals[t][0] for t in range(chunk)]+[17]
   equal(session.restore_token_prefix(0,epochs[0],query),1,label+' only committed input cached')
   selected=[0]*batch;selected[0]=epochs[0]
   prediction=session.predict_rows(selected)
   equal(prediction[0],first[0],label+' restored last hidden prediction')
   # Nonfinite target outputs poison the session instead of exposing full draft
   # state; reset must restore usable banks, including borrowed source outputs.
   bad_hidden=val([float('nan')]*(batch*chunk*w),[batch,chunk,w])
   sl1=val([float(1+t) for row in range(batch) for t in range(chunk)],[batch,chunk])
   pos1=val([float(1+t) for row in range(batch) for t in range(chunk) for axis in range(3)],[batch,chunk,3])
   active1=val([1.]+[0.]*(batch*chunk-1),[batch,chunk]);untouched=state(other)
   reject(lambda:session.verify_greedy(bad_hidden,pos1,sl1,active1,ids,[],[chunk]*batch,epochs),label+' nonfinite target')
   reject(lambda:session.snapshot(0),label+' poisoned snapshot');reject(lambda:session.hidden(),label+' poisoned hidden')
   equal(state(other),untouched,label+' failed transaction other private state')
   session.reset();reference.reset()
   equal(step(session,known,[0.]*batch,[1.]*batch),step(reference,known,[0.]*batch,[1.]*batch),label+' reset recovery prediction')
   equal(state(session),state(reference),label+' reset recovery all state')
   # Changed prepared graphs reject before execution.
   before=state(session);base[0].close()
   message=reject(lambda:session.verify_greedy(ph,pos,sl,active1,ids,[],[chunk]*batch,None),label+' closed source')
   assert 'graph/source changed' in message,message
   equal(state(session),before,label+' closed source unchanged')
   inputs=symbols([[batch,w],[batch,320,4],[batch,4,128,16],[batch]],'greedy_rebuilt_'+label+'_')
   @T.fusion(name='greedy_rebuilt_'+label,boundary='required')
   def rebuilt(h,a,b,flags):return Qwen35LinearDecoderLayer(h,a,b,flags,c,0)
   base[0].build_bf16(rebuilt(*inputs),inputs,f['checkpoint'])
   message=reject(lambda:session.verify_greedy(ph,pos,sl,active1,ids,[],[chunk]*batch,None),label+' rebuilt source')
   assert 'graph/source changed' in message,message
   equal(state(session),before,label+' rebuilt source unchanged')
   heads[1].close()
   message=reject(lambda:session.verify_greedy(ph,pos,sl,active1,ids,[],[chunk]*batch,None),label+' closed prepared head')
   assert 'head changed' in message,message
   equal(state(session),before,label+' closed head unchanged')
   packed[0].close()
   reject(lambda:session.verify_greedy(ph,pos,sl,active1,ids,[],[chunk]*batch,None),label+' closed verifier')
   equal(state(session),before,label+' closed verifier unchanged')
  print('native_greedy_profile PASS',label,'cases',cases,'checks',checks,flush=True)
 finally:
  session.close();reference.close();other.close()
  for graph in packed+base+heads:graph.close()
io.close();garnet.shutdown();owners=json.loads(garnet.memory_stats_json())
for key in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):equal(owners[key],0,key)
print('native_greedy_verification PASS',json.dumps(dict(cases=cases,checks=checks,coverage=sorted(coverage),owners=owners,model_speed_claim=False)),flush=True)
