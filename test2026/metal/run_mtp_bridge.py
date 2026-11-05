"""Native MTP transfer/correction vs standalone MTP fed ordinary target states."""
import json,os,sys
sys.path.insert(0,sys.argv[2]);sys.path.insert(0,sys.argv[3])
import garnet,tensor
from xModel.qwen35.vl_0_8b.decoder_layer import T,norm,Qwen35LinearDecoderLayer,Qwen35FullDecoderLayer
from xModel.qwen35.vl_0_8b.prefill_layer import PrefillLayer
from xModel.qwen38.vl_27b.metal_mtp import MetalMTPProgram
with open(sys.argv[1]) as stream:f=json.load(stream)
profile_mode='profiles' in sys.argv[5:]
scheduled_mode='scheduled' in sys.argv[5:]
c=f['config'];w=c['hidden_size'];generation=len(sys.argv)>5 and sys.argv[5]=='generation'
if generation:
 # This synthetic fixture alone uses whole-second mtimes so the XLang/Python
 # test bridge can restore timestamps without a nanosecond integer conversion.
 stamp=os.stat(f['checkpoint']);os.utime(f['checkpoint'],(int(stamp.st_mtime),int(stamp.st_mtime)))
io=garnet.tensor_io();io.open(f['checkpoint']);checks=0;cases=0;generation_cases=0;generation_eos=0
def val(data,shape):return tensor.tensor(data,shape=shape,dtype=tensor.float32)
def syms(shapes,name):return [tensor.input(name+str(i),shape=s,dtype=tensor.float32) for i,s in enumerate(shapes)]
def equal(a,b,label):
 global checks
 assert a==b,(label,a,b);checks+=1
def reject(call,label):
 global checks
 failed=False
 try:call()
 except Exception as error:failed=True;message=str(error)
 assert failed,'Missing rejection: '+label;checks+=1
 return message
def hashes(parts):return [io.verification_f32_sha256(x) for x in parts]
for batch,chunk,capacity in [(1,4,16),(2,2,16),(4,1,16),(1,4,4096)]:
 base=[];verify=[];heads=[];extra=[];session=garnet.decoder_session();reference=garnet.decoder_session();mtp=None;oracle=None;profile_programs=[]
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
  if 'native_prefill_hidden_pool' in garnet.backend_info():
   hidden_inventory=json.loads(garnet.memory_inventory_json([],session))
   pooled=[r for r in hidden_inventory['references'] if r['role']=='prefill_hidden_pingpong']
   equal(len(pooled),2 if chunk>1 else 0,label+' two private hidden buffers per compiled chunk')
   equal(len({r['buffer_id'] for r in pooled}),len(pooled),label+' independent hidden pingpong buffers')
  mtp=MetalMTPProgram(sys.argv[4],c,batch,capacity,chunk);oracle=MetalMTPProgram(sys.argv[4],c,batch,capacity,chunk,source=mtp)
  if 'native_mtp_deferred_packs' in garnet.backend_info():
   equal(mtp.graph.mtp_deferred_pack_slots>0,True,label+' native packed defaults deferred')
   equal(oracle.graph.mtp_deferred_pack_slots>0,True,label+' standalone packed defaults deferred')
  capture_chunks=[1] if chunk==1 else [1,chunk]
  if batch==1 and capacity==16:capture_chunks.append(8)
  session.prepare_mtp(mtp.graph,projection_head,capture_chunks)
  if profile_mode:
   if chunk>1:profile_programs.append(MetalMTPProgram(sys.argv[4],c,batch,capacity,1,source=mtp))
   if 8 in capture_chunks:profile_programs.append(MetalMTPProgram(sys.argv[4],c,batch,capacity,8,source=mtp,prefill=True))
   if profile_programs:
    before=json.loads(garnet.memory_stats_json())['metal_allocated_bytes']
    reject(lambda:session.prepare_mtp_profiles([profile_programs[0].graph]*2),label+' duplicate profile')
    equal(json.loads(garnet.memory_stats_json())['metal_allocated_bytes'],before,label+' rejected profile no allocation')
    session.prepare_mtp_profiles([program.graph for program in profile_programs])
    reject(lambda:session.prepare_mtp_profiles([profile_programs[0].graph]),label+' repeated profile preparation')
    inv=json.loads(garnet.memory_inventory_json([{'tag':'primary','object':mtp.graph}]+[{'tag':'profile'+str(i),'object':program.graph} for i,program in enumerate(profile_programs)],session))
    weights={}
    for ref in inv['references']:
     if ref['role']=='weight_record':
      tag=ref['tag']
      if tag not in weights:weights[tag]={}
      if ref['name'] not in weights[tag]:weights[tag][ref['name']]=[]
      weights[tag][ref['name']].append(ref['buffer_id'])
    for tag in weights:equal(weights[tag],weights['primary'],label+' shared profile weights '+tag)

  if generation:
   reject(lambda:session.prepare_mtp_embedding(sys.argv[4]),label+' foreign embedding checkpoint')
   session.prepare_mtp_embedding(f['checkpoint'])
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
   if 'native_mtp_kv_inplace' in garnet.backend_info():equal(json.loads(session.mtp_stats_json())['kv_inplace_updates'],2,label+' private in-place auxiliary K/V')
  if generation:
   for mode in range(4):
    session.reset();reference.reset();key=io.zeros([batch,2,capacity,32]);value=io.zeros([batch,2,capacity,32]);last=io.zeros([batch,w]);predicted=[0]*batch
    epochs=None
    if mode%2:
     epochs=[session.claim_request_row(80+row)[1] for row in range(batch)]
    delta=[.25,2.5,4.75]
    for slot in range(3):
     ids=[3+slot*17+row for row in range(batch)];pos=val([float(slot)+d for row in range(batch) for d in delta],[batch,3]);sl=val([float(slot)]*batch,[batch]);ac=val([1.]*batch,[batch]);embedding=io.embedding(ids)
     known=session.run_leased(base,embedding,pos,sl,ac,0,epochs) if epochs is not None else session.run(base,embedding,pos,sl,ac,0)
     equal(known,reference.run(base,embedding,pos,sl,ac,0),label+' native seed')
     next_ids=[3+(slot+1)*17+row if slot<2 else known[row] for row in range(batch)]
     embedding=stack_ids([[next_ids[row]]+[0]*(chunk-1) for row in range(batch)])
     controls=list(ctl([slot]*batch,[1]*batch));controls[0]=val([float(slot+j+1)+d for row in range(batch) for j in range(chunk) for d in delta],[batch,chunk,3])
     normalized=normalizer.run([reference.hidden()]);wanted=oracle_hidden_step(normalized,embedding,controls,[1]*batch)
     if slot<2:
      equal(session.mtp_teacher_cache(embedding,*controls,0,epochs),[-1]*batch,label+' cache-only teacher');equal(json.loads(session.mtp_stats_json())['head_calls'],0,label+' skipped vocabulary projection')
      reject(lambda:session.mtp_generate(known,[1]*batch,[],epochs),label+' uninitialized draft prediction')
     else:equal(session.mtp_teacher(embedding,*controls,0,epochs),wanted,label+' ready teacher')
     compare(label+' native prefill')
    ref_last_target=reference.hidden().tolist();budgets=[1+row for row in range(batch)] if mode==0 else [2+row for row in range(batch)] if mode==1 else [chunk+2]*batch
    stops=[i for i in range(c['vocab_size']) if i not in known] if mode==3 else []
    before=hashes(session.mtp_snapshot());wrong=list(known);wrong[0]=(wrong[0]+1)%c['vocab_size']
    reject(lambda:session.mtp_generate(wrong,budgets,[],epochs),label+' wrong authoritative known token');equal(hashes(session.mtp_snapshot()),before,label+' generation preflight unchanged')
    reject(lambda:session.mtp_generate(known,[-1]+budgets[1:],[],epochs),label+' negative generation budget')
    reject(lambda:session.mtp_generate(known,[capacity+1]+budgets[1:],[],epochs),label+' excessive context budget')
    reject(lambda:session.mtp_generate(known,budgets,[-1],epochs),label+' invalid generation stop ID')
    reject(lambda:session.mtp_generate(known,budgets,[known[0]],epochs),label+' stopped known token')
    if epochs is not None:
     stale=list(epochs);stale[-1]+=1;inactive=list(budgets);inactive[-1]=0
     reject(lambda:session.mtp_generate(known,inactive,[],stale),label+' stale inactive generation lease')
    stamp=os.stat(f['checkpoint'])
    try:
     os.utime(f['checkpoint'],(stamp.st_atime,stamp.st_mtime+1))
     message=reject(lambda:session.mtp_generate(known,budgets,[],epochs),label+' changed embedding checkpoint')
     assert 'embedding checkpoint changed' in message,message
    finally:os.utime(f['checkpoint'],(stamp.st_atime,stamp.st_mtime))
    equal(hashes(session.mtp_snapshot()),before,label+' all generation preflight failures unchanged')
    remaining=list(budgets);positions=[[3.+d for d in delta] for row in range(batch)];slots=[3]*batch
    while any(remaining):
     bases=list(slots);base_key=key;base_value=value;base_last=last;base_pred=list(predicted)
     # Closing an external helper cannot close the runtime's private mapping.
     io.close();result=session.mtp_generate(known,remaining,stops,epochs);io.open(f['checkpoint']);stats=json.loads(session.mtp_generation_stats_json())
     if 'native_prefill_hidden_pool' in garnet.backend_info():equal(session.prefill_hidden_pool_bindings,4 if chunk>1 else 0,label+' every verifier layer uses hidden pool')
     if profile_mode:equal(stats['draft_profile_chunk'],1,label+' native T1 draft selection')
     proposals=stats['proposals'];lengths=stats['submitted_lengths'];counts=[x[1] for x in result]
     def gctl(starts,sizes,shift):
      s=[float(starts[row]+j) for row in range(batch) for j in range(chunk)];a=[float(j<sizes[row]) for row in range(batch) for j in range(chunk)]
      p=[positions[row][axis]+shift+j for row in range(batch) for j in range(chunk) for axis in range(3)]
      return val(p,[batch,chunk,3]),val(s,[batch,chunk]),val(a,[batch,chunk])
     for row in range(batch):
      if lengths[row]>1:equal(proposals[row*chunk+1],predicted[row],label+' native first draft')
     for j in range(1,chunk-1):
      advance=[int(j+1<lengths[row] and proposals[row*chunk+j] not in stops) for row in range(batch)]
      if any(advance):
       emb=stack_ids([[proposals[row*chunk+j]]+[0]*(chunk-1) for row in range(batch)]);source=last;oracle_hidden_step(source,emb,gctl([bases[row]+j-1 for row in range(batch)],advance,j),advance)
       for row in range(batch):
        if advance[row]:equal(proposals[row*chunk+j+1],predicted[row],label+' native recursive draft')
     # The ordinary oracle consumes only the independent greedy accepted prefix.
     expected=[[] for row in range(batch)];current=list(known);ended=[False]*batch;normalized=[]
     for j in range(max(lengths)):
      active=[float(j<lengths[row] and not ended[row]) for row in range(batch)]
      if not any(active):break
      pred=reference.run(base,io.embedding(current),val([positions[row][axis]+j for row in range(batch) for axis in range(3)],[batch,3]),val([float(bases[row]+j) for row in range(batch)],[batch]),val(active,[batch]),0)
      hidden_values=reference.hidden().tolist();normalized.append(normalizer.run([reference.hidden()]))
      for row in range(batch):
       if active[row]:
        equal(proposals[row*chunk+j],current[row],label+' target accepted input');ref_last_target[row*w:(row+1)*w]=hidden_values[row*w:(row+1)*w];expected[row].append(pred[row]);current[row]=pred[row]
        ended[row]=pred[row] in stops or j+1==lengths[row] or pred[row]!=proposals[row*chunk+j+1]
     equal([x[0] for x in result],expected,label+' native ordinary output IDs');equal(counts,[len(x) for x in expected],label+' native accepted count')
     for i in range(4):equal(hashes(session.snapshot(i)),hashes(reference.snapshot(i)),label+' native target complete state')
     equal(io.verification_f32_sha256(session.hidden()),io.verification_f32_sha256(val(ref_last_target,[batch,w])),label+' native final target hidden')
     key=base_key;value=base_value;last=base_last;predicted=base_pred
     emb=stack_ids([expected[row]+[0]*(chunk-counts[row]) for row in range(batch)]);source=io.stack_tokens(normalized+[io.zeros([batch,w])]*(chunk-len(normalized)))
     oracle_step(source,emb,gctl(bases,counts,1),counts);compare(label+' native corrected auxiliary')
     equal(stats['accepted_drafts'],sum(max(n-1,0) for n in counts),label+' native draft accounting')
     for row in range(batch):
      if counts[row]:known[row]=expected[row][-1]
      generation_eos+=int(result[row][2])
      remaining[row]=0 if result[row][2] else remaining[row]-counts[row];slots[row]+=counts[row];positions[row]=[x+counts[row] for x in positions[row]]
     equal(stats['auxiliary_validation_calls'],int(not any(remaining)),label+' final vocabulary head skipped');generation_cases+=1
    before=hashes(session.mtp_snapshot());equal(session.mtp_generate(known,[0]*batch,stops,epochs),[[[],0,False] for row in range(batch)],label+' idle generation');equal(hashes(session.mtp_snapshot()),before,label+' idle unchanged')
   print('mtp_native_generation_profile PASS',label,'transactions',generation_cases,'checks',checks,flush=True)
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
  if profile_mode and batch==1 and capacity==16:
   wide_oracle=profile_programs[-1]
   for valid in (1,3,7,8):
    session.reset();reference.reset()
    key=io.zeros([1,2,capacity,32]);value=io.zeros([1,2,capacity,32]);last=io.zeros([1,w]);predicted=[0]
    ids=[3+j*7 for j in range(8)];embedding=io.stack_tokens([io.embedding([x]) for x in ids]);p=val([float(j) for j in range(8) for axis in range(3)],[1,8,3]);slots=val([float(j) for j in range(8)],[1,8]);flags=val([float(j<valid) for j in range(8)],[1,8])
    actual=session.run(wide,embedding,p,slots,flags,valid-1);equal(actual,reference.run(wide,embedding,p,slots,flags,valid-1),label+' profile target')
    packed=io.stack_tokens([normalizer.run([io.visual_token(reference.hidden(),j)]) for j in range(8)])
    next_ids=ids[1:valid]+[actual[0]]+[0]*(8-valid);embedding=io.stack_tokens([io.embedding([x]) for x in next_ids]);p=val([float(j+1) for j in range(8) for axis in range(3)],[1,8,3]);ctl_w=[p,slots,flags]
    result=wide_oracle.run([packed,embedding,key,value]+ctl_w);key=io.slice(result,8*w,[1,2,capacity,32]);value=io.slice(result,8*w+2*capacity*32,[1,2,capacity,32]);last=io.slice(result,(valid-1)*w,[1,w]);predicted=io.argmax(projection_head.run([last]))
    before=hashes(session.mtp_snapshot());reject(lambda:session.mtp_teacher(embedding,*ctl_w,1,None),label+' incomplete wide source window');equal(hashes(session.mtp_snapshot()),before,label+' incomplete source unchanged')
    equal(session.mtp_teacher(embedding,*ctl_w,0,None),predicted,label+' wide profile prediction');compare(label+' ragged wide profile')
    equal(json.loads(session.mtp_stats_json())['committed'],[valid],label+' actual wide length');equal(json.loads(session.mtp_stats_json())['profile_chunk'],8,label+' wide profile chosen');cases+=1
  if scheduled_mode:
   for stop_mode in (0,1):
    session.reset();reference.reset();policy=garnet.continuous_scheduler();policy.initialize(session,16);policy.set_prefill_chunk(1)
    reject(lambda:policy.mtp_generate_json([]),label+' unscheduled MTP')
    jobs={};retired=0;frames=0;refreshes=0;stopped=0
    for j in range(batch+2):
     limit=3+j%4;identity=policy.submit(1,limit,0);jobs[identity]=dict(limit=limit,known=0,slot=1,row=-1,generated=1)
    try:
     while jobs:
      tick=json.loads(policy.next_json())
      assert tick['batches'],label+' scheduled progress stalled'
      for plan in tick['batches']:
       if plan['stage']=='prefill':
        flags=[0.]*batch;ids=[0]*batch;epochs=[0]*batch
        for request in plan['requests']:
         row=request['row'];reference.reset_row(row);flags[row]=1.;ids[row]=3+request['id']%41;epochs[row]=request['epoch'];jobs[request['id']]['row']=row
        actual=session.run_leased(base,io.embedding(ids),val([0.]*(batch*3),[batch,3]),val([0.]*batch,[batch]),val(flags,[batch]),-1,epochs)
        first=session.predict_rows(epochs);wanted=reference.run(base,io.embedding(ids),val([0.]*(batch*3),[batch,3]),val([0.]*batch,[batch]),val(flags,[batch]),0)
        embeddings=stack_ids([[first[row] if flags[row] else 0]+[0]*(chunk-1) for row in range(batch)]);controls=ctl([0]*batch,[int(x) for x in flags]);session.mtp_teacher(embeddings,*controls,0,epochs)
        for request in plan['requests']:
         row=request['row'];identity=request['id'];equal(first[row],wanted[row],label+' scheduled first token');jobs[identity]['known']=first[row]
         equal(policy.complete_prefill(identity,first[row],False),2,label+' native prefill ownership')
       else:
        current={request['row']:request for request in plan['requests']};known_set=[jobs[r['id']]['known'] for r in plan['requests']]
        stop_ids=[i for i in range(c['vocab_size']) if i not in known_set] if stop_mode else []
        result=json.loads(policy.mtp_generate_json(stop_ids));frames+=1;refreshes+=int(result['generation']['target_prediction_refreshed'])
        by_row={item['row']:item for item in result['requests']}
        equal(set(by_row),set(current),label+' scheduled result ownership')
        for item in result['requests']:
         request=current[item['row']];job=jobs[item['id']]
         equal((item['id'],item['epoch']),(request['id'],request['epoch']),label+' scheduled lease')
         equal(item['known'],job['known'],label+' authoritative scheduled known')
         equal(item['remaining_before'],job['limit']-job['generated'],label+' native output budget')
        expected={row:[] for row in by_row}
        for step_index in range(max(item['consumed'] for item in result['requests'])):
         ids=[];flags=[];slots=[]
         for row in range(batch):
          item=by_row.get(row);valid=item is not None and step_index<item['consumed'];job=jobs[item['id']] if item is not None else None
          ids.append((job['known'] if step_index==0 else item['emitted'][step_index-1]) if valid else 0);flags.append(float(valid));slots.append(float(job['slot']+step_index) if valid else 0.)
         predictions=reference.run(base,io.embedding(ids),val([slot for slot in slots for axis in range(3)],[batch,3]),val(slots,[batch]),val(flags,[batch]),0)
         for row in by_row:
          if flags[row]:expected[row].append(predictions[row])
        for item in result['requests']:
         row=item['row'];job=jobs[item['id']];equal(item['emitted'],expected[row],label+' scheduled ordinary output IDs')
         equal(item['eos'],item['emitted'][-1] in stop_ids,label+' scheduled EOS')
         job['generated']+=item['consumed'];job['slot']+=item['consumed'];job['known']=item['emitted'][-1]
         equal(item['finished'],item['eos'] or job['generated']==job['limit'],label+' scheduled completion')
         if item['finished']:
          reference.reset_row(row);jobs.pop(item['id']);retired+=1;stopped+=int(item['eos'])
        for layer in range(4):equal(hashes(session.snapshot(layer)),hashes(reference.snapshot(layer)),label+' scheduled target state'+str(layer))
     equal(retired,batch+2,label+' all queued jobs retired');equal(json.loads(policy.status_json())['tracked'],0,label+' no scheduler metadata leak')
     if batch>1 and stop_mode==0:assert refreshes>0,label+' no interleaved prediction refresh'
     if stop_mode:assert stopped>0,label+' no native scheduled EOS'
     cases+=frames
     print('mtp_native_scheduled PASS',label,'stop_mode',stop_mode,'frames',frames,'refreshes',refreshes,'EOS',stopped,'checks',checks,flush=True)
    finally:policy.close()
   session.reset();reference.reset()
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
  session.reset();current=target_step(session,[3]*batch,0);session.mtp_teacher(embedding,*controls,0,None);checks+=1
  if generation:
   before=hashes(session.mtp_snapshot());(profile_programs[0].graph if profile_programs else mtp.graph).close();message=reject(lambda:session.mtp_generate(current,[1]*batch,[],None),label+' closed owner before shallow generation')
   assert ('Native generation MTP profile owner changed' if profile_programs else 'Native generation graph owner changed') in message,message
   equal(hashes(session.mtp_snapshot()),before,label+' closed owner rejection unchanged')
  print('mtp_bridge_profile PASS',label,'cases',cases,'checks',checks,flush=True)
 finally:
  session.close();reference.close()
  for program in profile_programs:program.close()
  if mtp is not None:mtp.close()
  if oracle is not None:oracle.close()
  for graph in base+verify+heads+extra:graph.close()
garnet.shutdown();memory=json.loads(garnet.memory_stats_json())
if generation:assert generation_eos>0,'No actual generated EOS coverage'
for name in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):assert memory[name]==0,(name,memory[name])
print('mtp_bridge PASS',json.dumps(dict(cases=cases,generation_transactions=generation_cases,generation_eos=generation_eos,checks=checks,owners=memory,model_speed_claim=False)),flush=True)
