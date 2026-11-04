"""Native MTP transfer/correction vs standalone MTP fed ordinary target states."""
import json,sys
sys.path.insert(0,sys.argv[2]);sys.path.insert(0,sys.argv[3])
import garnet,tensor
from xModel.qwen35.vl_0_8b.decoder_layer import T,norm,Qwen35LinearDecoderLayer,Qwen35FullDecoderLayer
from xModel.qwen35.vl_0_8b.prefill_layer import PrefillLayer
from xModel.qwen38.vl_27b.metal_mtp import MetalMTPProgram
with open(sys.argv[1]) as stream:f=json.load(stream)
c=f['config'];w=c['hidden_size'];io=garnet.tensor_io();io.open(f['checkpoint']);checks=0;cases=0
def val(data,shape):return tensor.tensor(data,shape=shape,dtype=tensor.float32)
def syms(shapes,name):return [tensor.input(name+str(i),shape=s,dtype=tensor.float32) for i,s in enumerate(shapes)]
def equal(a,b,label):
 global checks
 assert a==b,(label,a,b);checks+=1
def reject(call,label):
 global checks
 failed=False
 try:call()
 except Exception:failed=True
 assert failed,'Missing rejection: '+label;checks+=1
def hashes(parts):return [io.verification_f32_sha256(x) for x in parts]
for batch,chunk,capacity in [(1,4,16),(2,2,16),(4,1,16),(1,4,4096)]:
 base=[];verify=[];heads=[];extra=[];session=garnet.decoder_session();reference=garnet.decoder_session();mtp=None;oracle=None
 label='B'+str(batch)+'T'+str(chunk)+'C'+str(capacity)
 try:
  for i in range(4):
   states=([[batch,320,4],[batch,4,128,16]] if i%2==0 else [[batch,2,capacity,32]]*2)
   shapes=[[batch,w]]+states+([[batch]] if i%2==0 else [[batch,3],[batch],[batch]])
   inputs=syms(shapes,'bridge_base_'+label+'_'+str(i)+'_')
   if i%2==0:
    @T.fusion(name='bridge_base_'+label+'_'+str(i),boundary='required')
    def body(h,a,b,active):return Qwen35LinearDecoderLayer(h,a,b,active,c,i)
   else:
    @T.fusion(name='bridge_base_'+label+'_'+str(i),boundary='required')
    def body(h,a,b,positions,slots,active):return Qwen35FullDecoderLayer(h,a,b,positions,slots,active,c,i,capacity)
   g=garnet.compiled_graph();g.build_bf16(body(*inputs),inputs,f['checkpoint']);base.append(g)
   inputs=syms([[batch,chunk,w]]+states+[[batch,chunk,3],[batch,chunk],[batch,chunk]],'bridge_verify_'+label+'_'+str(i)+'_')
   @T.fusion(name='bridge_verify_'+label+'_'+str(i),boundary='required')
   def body(h,a,b,positions,slots,active):return PrefillLayer(h,a,b,positions,slots,active,c,i,batch,chunk,capacity,'linear_attention' if i%2==0 else 'full_attention',parallel_attention=True,parallel_linear=True,attention_capacity_limit=4096)
   g=garnet.compiled_graph();g.build_verification_shared(body(*inputs),inputs,f['checkpoint'],base[-1]);verify.append(g)
  for rows in (batch,batch*chunk):
   inputs=syms([[rows,w]],'bridge_head_'+label+'_'+str(rows)+'_')
   @T.fusion(name='bridge_head_'+label+'_'+str(rows),boundary='required')
   def head(h):return norm(h,'model.language_model.norm',c)*T.unary_op('qwen_text_head',weight_name='lm_head.weight',bf16_weight=1)
   g=garnet.compiled_graph()
   if heads:g.build_shared(head(*inputs),inputs,f['checkpoint'],heads[0])
   else:g.build(head(*inputs),inputs,f['checkpoint'])
   heads.append(g)
  inputs=syms([[batch,w]],'bridge_projection_'+label+'_')
  @T.fusion(name='bridge_projection_'+label,boundary='required')
  def projection(h):return h*T.unary_op('qwen_text_head',weight_name='lm_head.weight',bf16_weight=1)
  projection_head=garnet.compiled_graph();projection_head.build_shared(projection(*inputs),inputs,f['checkpoint'],heads[0]);extra.append(projection_head)
  @T.fusion(name='bridge_normalize_'+label,boundary='required')
  def normalize(h):return norm(h,'model.language_model.norm',c)
  normalizer=garnet.compiled_graph();normalizer.build_shared(normalize(*inputs),inputs,f['checkpoint'],heads[0]);extra.append(normalizer)
  session.build(base,heads[0]);reference.build(base,heads[0]);session.prepare_verification(verify,heads[1])
  mtp=MetalMTPProgram(sys.argv[4],c,batch,capacity,chunk);oracle=MetalMTPProgram(sys.argv[4],c,batch,capacity,chunk,source=mtp)
  capture_chunks=[1] if chunk==1 else [1,chunk]
  if batch==1 and capacity==16:capture_chunks.append(8)
  session.prepare_mtp(mtp.graph,projection_head,capture_chunks)
  def stack_ids(rows):return io.stack_tokens([io.embedding([rows[row][j] for row in range(batch)]) for j in range(chunk)])
  def target_step(owner,ids,slot):return owner.run(base,io.embedding(ids),val([float(slot)]*(batch*3),[batch,3]),val([float(slot)]*batch,[batch]),val([1.]*batch,[batch]),0)
  def ctl(starts,lengths):
   slots=[float(starts[row]+j) for row in range(batch) for j in range(chunk)];flags=[float(j<lengths[row]) for row in range(batch) for j in range(chunk)]
   return val([s+1 for s in slots for axis in range(3)],[batch,chunk,3]),val(slots,[batch,chunk]),val(flags,[batch,chunk])
  def oracle_step(normalized,embeddings,controls,lengths):
   global key,value,last,predicted
   output=oracle.run([normalized,embeddings,key,value]+list(controls));cache=2*capacity*32
   key=io.slice(output,chunk*w,[batch,2,capacity,32]);value=io.slice(output,chunk*w+cache,[batch,2,capacity,32]);values=io.slice(output,0,[batch,chunk*w]).tolist();old=last.tolist()
   for row in range(batch):
    if lengths[row]:old[row*w:(row+1)*w]=values[row*chunk*w+(lengths[row]-1)*w:row*chunk*w+lengths[row]*w]
   last=val(old,[batch,w]);predicted=io.argmax(projection_head.run([last]));return [predicted[row] if lengths[row] else -1 for row in range(batch)]
  def oracle_hidden_step(hidden,embeddings,controls,lengths):return oracle_step(io.stack_tokens([hidden]+[io.zeros([batch,w])]*(chunk-1)),embeddings,controls,lengths)
  def compare(label):
   snap=session.mtp_snapshot();equal(hashes(snap[:3]),hashes([key,value,last]),label+' private caches/last')
   equal(io.argmax(projection_head.run([snap[2]])),predicted,label+' shared head')
  for requested in range(1,chunk+1):
   session.reset();reference.reset();key=io.zeros([batch,2,capacity,32]);value=io.zeros([batch,2,capacity,32]);last=io.zeros([batch,w]);predicted=[0]*batch
   for slot in range(3):
    ids=[3+slot*17+row for row in range(batch)];next_ids=[3+(slot+1)*17+row if slot<2 else 7+row for row in range(batch)]
    target_step(session,ids,slot);target_step(reference,ids,slot);normalized=normalizer.run([reference.hidden()]);embeddings=stack_ids([[next_ids[row]]+[0]*(chunk-1) for row in range(batch)]);controls=ctl([slot]*batch,[1]*batch)
    equal(session.mtp_teacher(embeddings,*controls,0,None),oracle_hidden_step(normalized,embeddings,controls,[1]*batch),label+' seed teacher');compare(label+' seed')
   base_key=key;base_value=value;base_last=last;base_pred=list(predicted)
   for j in range(max(0,chunk-1)):
    embeddings=stack_ids([[13+j+row]+[0]*(chunk-1) for row in range(batch)]);controls=ctl([3+j]*batch,[1]*batch)
    equal(session.mtp_draft(embeddings,*controls,None),oracle_hidden_step(last,embeddings,controls,[1]*batch),label+' draft');compare(label+' draft')
   # Ordinary target oracle records every normalized candidate state.
   known=[7+row for row in range(batch)];current=list(known);ideal=[];outputs=[];normalized=[]
   for j in range(chunk):ideal.append(list(current));current=target_step(reference,current,3+j);outputs.append(list(current));normalized.append(normalizer.run([reference.hidden()]))
   proposals=[[ideal[j][row] for j in range(chunk)] for row in range(batch)];lengths=[chunk]*batch
   desired=[min(chunk,requested+row) for row in range(batch)]
   if batch>1 and requested==1:lengths[-1]=0;desired[-1]=0
   for row in range(batch):
    if desired[row] and desired[row]<chunk:proposals[row][desired[row]]=(proposals[row][desired[row]]+1)%c['vocab_size']
   controls=ctl([3]*batch,lengths);target_slots=[float(3+j) for row in range(batch) for j in range(chunk)]
   result=session.verify_greedy(stack_ids(proposals),val([s for s in target_slots for axis in range(3)],[batch,chunk,3]),controls[1],controls[2],[x for row in proposals for x in row],[],[chunk]*batch,None)
   counts=[r[1] for r in result];equal(counts,desired,label+' target accepted counts')
   # Replay the teacher oracle from the committed pre-draft cache, never its
   # tentative suffix. Native correction must restore all rejected old slots.
   key=base_key;value=base_value;last=base_last;predicted=base_pred
   embeddings=stack_ids([result[row][0]+[0]*(chunk-counts[row]) for row in range(batch)]);controls=ctl([3]*batch,counts)
   packed_normalized=io.stack_tokens(normalized)
   before=hashes(session.mtp_snapshot())
   if min(counts)<chunk:
    wrong=ctl([3]*batch,[chunk]*batch);reject(lambda:session.mtp_teacher(embeddings,*wrong,0,None),label+' rejected target source');equal(hashes(session.mtp_snapshot()),before,label+' rejected metadata unchanged')
   expected=oracle_step(packed_normalized,embeddings,controls,counts)
   equal(session.mtp_teacher(embeddings,*controls,0,None),expected,label+' correction predictions');compare(label+' correction')
   equal(json.loads(session.mtp_stats_json())['committed'],[3+n for n in counts],label+' corrected native length')
   cases+=1
  if batch==1 and capacity==16:
   # Wider prepared target capture is consumed in native four-token windows.
   wide=[]
   for i in range(4):
    states=([[batch,320,4],[batch,4,128,16]] if i%2==0 else [[batch,2,capacity,32]]*2)
    inputs=syms([[1,8,w]]+states+[[1,8,3],[1,8],[1,8]],'bridge_wide_'+str(i)+'_')
    @T.fusion(name='bridge_wide_'+str(i),boundary='required')
    def body(h,a,b,p,s,active):return PrefillLayer(h,a,b,p,s,active,c,i,1,8,capacity,'linear_attention' if i%2==0 else 'full_attention',parallel_attention=True,parallel_linear=True,attention_capacity_limit=4096)
    g=garnet.compiled_graph();g.build_shared(body(*inputs),inputs,f['checkpoint'],base[i]);extra.append(g);wide.append(g);session.prepare_prefill(g,i)
   session.reset();reference.reset();key=io.zeros([1,2,capacity,32]);value=io.zeros([1,2,capacity,32]);last=io.zeros([1,w]);predicted=[0]
   ids=[3+i*7 for i in range(8)];embeddings=io.stack_tokens([io.embedding([x]) for x in ids]);positions=val([float(j) for j in range(8) for axis in range(3)],[1,8,3]);slots=val([float(j) for j in range(8)],[1,8]);flags=val([1.]*8,[1,8])
   prediction=session.run(wide,embeddings,positions,slots,flags,7);equal(prediction,reference.run(wide,embeddings,positions,slots,flags,7),label+' wide target prediction')
   normalized=[normalizer.run([io.visual_token(reference.hidden(),j)]) for j in range(8)]
   for offset in (0,4):
    next_ids=ids[offset+1:offset+5]
    if len(next_ids)<4:next_ids.append(prediction[0])
    embedding=stack_ids([next_ids]);controls=ctl([offset],[4]);expected=oracle_step(io.stack_tokens(normalized[offset:offset+4]),embedding,controls,[4])
    equal(session.mtp_teacher(embedding,*controls,offset,None),expected,label+' wide shifted teacher');compare(label+' wide teacher');cases+=1
  # Leases guard auxiliary history as well as target state, including inactive rows.
  session.reset();lease=session.claim_request_row(41);epochs=[0]*batch;epochs[lease[0]]=lease[1]
  flags=[0.]*batch;flags[lease[0]]=1.
  session.run_leased(base,io.embedding([3]*batch),val([0.]*(batch*3),[batch,3]),val([0.]*batch,[batch]),val(flags,[batch]),0,epochs)
  stale_target=list(epochs);stale_target[lease[0]]+=1
  reject(lambda:session.run_leased(base,io.embedding([3]*batch),val([0.]*(batch*3),[batch,3]),val([0.]*batch,[batch]),val([0.]*batch,[batch]),0,stale_target),label+' stale inactive target capture')
  if batch>1:
   unowned=[0.]*batch;unowned[1]=1.
   reject(lambda:session.run_leased(base,io.embedding([3]*batch),val([0.]*(batch*3),[batch,3]),val([0.]*batch,[batch]),val(unowned,[batch]),0,epochs),label+' unowned target capture')
  lengths=[int(x) for x in flags];embedding=stack_ids([[7]+[0]*(chunk-1) for row in range(batch)]);controls=ctl([0]*batch,lengths);before=hashes(session.mtp_snapshot())
  reject(lambda:session.mtp_teacher(embedding,*controls,0,None),label+' missing leases')
  stale=list(epochs);stale[lease[0]]+=1;reject(lambda:session.mtp_teacher(embedding,*controls,0,stale),label+' stale lease')
  equal(hashes(session.mtp_snapshot()),before,label+' lease rejection unchanged')
  session.mtp_teacher(embedding,*controls,0,epochs);equal(json.loads(session.mtp_stats_json())['committed'],lengths,label+' leased teacher')
  session.release_request_row(lease[0],lease[1]);equal(hashes(session.mtp_snapshot()[:3]),hashes([io.zeros([batch,2,capacity,32]),io.zeros([batch,2,capacity,32]),io.zeros([batch,w])]),label+' lease release clears auxiliary')
  # Nonfinite model results poison both paths until full reset; caller metadata does not.
  session.reset();target_step(session,[3]*batch,0);bad=val([float('nan')]*(batch*chunk*w),[batch,chunk,w]);controls=ctl([0]*batch,[1]*batch)
  reject(lambda:session.mtp_teacher(bad,*controls,0,None),label+' nonfinite poison');reject(lambda:session.mtp_snapshot(),label+' poisoned snapshot');reject(lambda:session.hidden(),label+' poisoned target hidden')
  session.reset();target_step(session,[3]*batch,0);session.mtp_teacher(embedding,*controls,0,None);checks+=1
  print('mtp_bridge_profile PASS',label,'cases',cases,'checks',checks,flush=True)
 finally:
  session.close();reference.close()
  if mtp is not None:mtp.close()
  if oracle is not None:oracle.close()
  for graph in base+verify+heads+extra:graph.close()
garnet.shutdown();memory=json.loads(garnet.memory_stats_json())
for name in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):assert memory[name]==0,(name,memory[name])
print('mtp_bridge PASS',json.dumps(dict(cases=cases,checks=checks,owners=memory,model_speed_claim=False)),flush=True)
