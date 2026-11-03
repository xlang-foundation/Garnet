"""Native token matching with mixed recurrent/KV snapshots and row reuse.

Arguments: request-row fixture, native build, Garnet source. Compares restored
state and suffix execution with fresh execution; request IDs are never keys.
"""
import sys,json,math
sys.path.insert(0,sys.argv[2]);sys.path.insert(0,sys.argv[3])
import garnet,tensor
from xModel.qwen35.vl_0_8b.decoder_layer import T,Qwen35LinearDecoderLayer,Qwen35FullDecoderLayer
from xModel.qwen35.vl_0_8b.prefill_layer import PrefillLayer
assert 'native_hybrid_token_prefix' in garnet.backend_info()
with open(sys.argv[1]) as stream:f=json.load(stream)
c=f['config'];batch=f['batch'];width=c['hidden_size'];capacity=f['capacity'];chunk=4
assert batch==4 and capacity>=8
state_shapes=[[[batch,29,4],[batch,3,7,5]],[[batch,1,capacity,6],[batch,1,capacity,6]]]
base=[];packed=[];head=None;budgeted=None;rogue=None;session=garnet.decoder_session();fresh=garnet.decoder_session()
identity=['fixture-model','fixture-b4-c8','fixture-tokenizer','text_zero_based_3axis_v1','local-user','text']
def make(values,shape):return tensor.tensor(values,shape=shape,dtype=tensor.float32)
def snapshot(owner):return [[v.tolist() for v in owner.snapshot(i)] for i in range(2)]
def near(a,b):
    assert len(a)==len(b)
    for x,y in zip(a,b):assert math.isfinite(x) and abs(x-y)<=5e-6+5e-5*abs(y),(x,y)
def compare_row(a,b,row):
    for layer in range(2):
        for part in range(2):
            n=len(a[layer][part])//batch
            near(a[layer][part][row*n:(row+1)*n],b[layer][part][row*n:(row+1)*n])
def reject(fn):
    before=snapshot(session);failed=False
    try:fn()
    except Exception:failed=True
    assert failed and snapshot(session)==before

def inputs(rows,starts):
    ids=[];values=[];slots=[];flags=[]
    for row in range(batch):
        for token in range(chunk):
            active=token<len(rows[row]);value=rows[row][token] if active else 0
            ids.append(value);flags.append(1.0 if active else 0.0)
            slots.append(float(starts[row]+token) if active else 0.0)
            values += [float((value*13+column*7)%53-26)/61 for column in range(width)]
    positions=[slot for slot in slots for axis in range(3)]
    return (make(values,[batch,chunk,width]),make(positions,[batch,chunk,3]),make(slots,[batch,chunk]),make(flags,[batch,chunk]),ids)
def run(rows,starts,epochs):
    h,p,s,a,ids=inputs(rows,starts)
    session.run_tokens(packed,h,p,s,a,-1,epochs,ids)
def reference(tokens,row=0):
    fresh.reset()
    for offset in range(0,len(tokens),chunk):
        rows=[[] for _ in range(batch)];rows[row]=tokens[offset:offset+chunk]
        starts=[0]*batch;starts[row]=offset;h,p,s,a,ids=inputs(rows,starts)
        fresh.run(packed,h,p,s,a,-1)
    return snapshot(fresh),fresh.hidden().tolist()
def bank_prediction(hidden,row,final):
    vector=[0.0]*(batch*width);start=(row*chunk+final)*width
    vector[row*width:(row+1)*width]=hidden[start:start+width]
    logits=head.run([make(vector,[batch,width])]).tolist()
    values=logits[row*width:(row+1)*width]
    return max(range(width),key=lambda i:values[i])
try:
    for layer in range(2):
        shapes=[[batch,width]]+state_shapes[layer]+([[batch]] if layer==0 else [[batch,3],[batch],[batch]])
        symbols=[tensor.input('hybrid_base_'+str(layer)+'_'+str(i),shape=s,dtype=tensor.float32) for i,s in enumerate(shapes)]
        if layer==0:
            @T.fusion(name='hybrid_base_linear',boundary='required')
            def forward(h,a,b,active):return Qwen35LinearDecoderLayer(h,a,b,active,c,0)
        else:
            @T.fusion(name='hybrid_base_full',boundary='required')
            def forward(h,a,b,p,s,active):return Qwen35FullDecoderLayer(h,a,b,p,s,active,c,1,capacity)
        graph=garnet.compiled_graph();graph.build(forward(*symbols),symbols,f['checkpoint']);base.append(graph)
        shapes=[[batch,chunk,width]]+state_shapes[layer]+[[batch,chunk,3],[batch,chunk],[batch,chunk]]
        symbols=[tensor.input('hybrid_packed_'+str(layer)+'_'+str(i),shape=s,dtype=tensor.float32) for i,s in enumerate(shapes)]
        kind='linear_attention' if layer==0 else 'full_attention'
        @T.fusion(name='hybrid_packed_layer_'+str(layer),boundary='required')
        def prefill(h,a,b,p,s,active):return PrefillLayer(h,a,b,p,s,active,c,layer,batch,chunk,capacity,kind,parallel_attention=True,parallel_linear=True)
        graph=garnet.compiled_graph();graph.build_shared(prefill(*symbols),symbols,f['checkpoint'],base[-1]);packed.append(graph)
    symbol=tensor.input('hybrid_head',shape=[batch,width],dtype=tensor.float32)
    @T.fusion(name='hybrid_prediction',boundary='required')
    def predict(h):return h * T.binary_op('add') * h
    head=garnet.compiled_graph();head.build(predict(symbol),[symbol],f['checkpoint'])
    session.build(base,head);fresh.build(base,head)
    reject(lambda:session.configure_token_prefix(identity,0,1024))
    bad=list(identity);bad[3]='unknown-position-policy'
    reject(lambda:session.configure_token_prefix(bad,2,1024*1024))
    assert session.configure_token_prefix(identity,2,1024*1024)
    reject(lambda:session.configure_token_prefix(identity,2,1024*1024))
    for layer in range(2):session.prepare_prefill(packed[layer],layer)
    leases=[session.claim_request_row(row+1) for row in range(batch)];epochs=[lease[1] for lease in leases]
    reject(lambda:session.cache_token_prefix(0,epochs[0]))
    a=[11,12,13,14,15,16]
    run([a[:2],[21,22,23],[31,32,33,34],[41]],[0]*batch,epochs)
    first=snapshot(session);assert session.cache_token_prefix(0,epochs[0])
    run([a[2:4],[],[],[]],[2,0,0,0],epochs)
    second=snapshot(session);assert session.cache_token_prefix(0,epochs[0])
    wanted,hidden=reference(a[:4]);compare_row(second,wanted,0)
    expected=bank_prediction(hidden,0,3)
    run([a[4:],[],[],[]],[4,0,0,0],epochs)
    before=snapshot(session);old=epochs[0]
    assert session.release_request_row(0,old)
    lease=session.claim_request_row(99);assert lease[0]==0;epochs[0]=lease[1]
    reject(lambda:session.restore_token_prefix(0,old,a))
    reject(lambda:session.cache_token_prefix(0,old))
    assert session.restore_token_prefix(0,epochs[0],a[:4])==4
    restored=snapshot(session);compare_row(restored,second,0)
    for row in range(1,batch):compare_row(restored,before,row)
    assert session.predict_rows([epochs[0],0,0,0])==[expected,-1,-1,-1]
    reject(lambda:session.restore_token_prefix(0,epochs[0],a))
    assert session.release_request_row(0,epochs[0]);epochs[0]=session.claim_request_row(199)[1]
    assert session.restore_token_prefix(0,epochs[0],a)==4
    compare_row(snapshot(session),second,0)
    changed=a[:4]+[77,78]
    run([changed[4:],[],[],[]],[4,0,0,0],epochs)
    wanted,hidden=reference(changed);compare_row(snapshot(session),wanted,0)
    assert session.predict_rows([epochs[0],0,0,0])[0]==bank_prediction(hidden,0,1)
    assert session.release_request_row(0,epochs[0]);epochs[0]=session.claim_request_row(100)[1]
    assert session.restore_token_prefix(0,epochs[0],a[:2]+[88,89])==2
    compare_row(snapshot(session),first,0)
    run([[88,89],[],[],[]],[2,0,0,0],epochs)
    wanted,hidden=reference(a[:2]+[88,89]);compare_row(snapshot(session),wanted,0)
    # Untracked execution makes a row ineligible; stale slot and token errors
    # must fail before changing any native request state.
    h,p,s,active,ids=inputs([[],[24],[],[]],[0,3,0,0])
    bad=list(ids);bad[chunk]=-1
    reject(lambda:session.run_tokens(packed,h,p,s,active,-1,epochs,bad))
    reject(lambda:session.run_tokens(packed,h,p,s,active,-1,epochs,ids[:-1]))
    h0,p0,s0,a0,i0=inputs([[],[24],[],[]],[0,1,0,0])
    reject(lambda:session.run_tokens(packed,h0,p0,s0,a0,-1,epochs,i0))
    values=p.tolist();values[chunk*3+2]+=1.0
    wrong_position=make(values,[batch,chunk,3])
    reject(lambda:session.run_tokens(packed,h,wrong_position,s,active,-1,epochs,ids))
    session.run_leased(packed,h,p,s,active,-1,epochs)
    reject(lambda:session.cache_token_prefix(1,epochs[1]))
    reject(lambda:session.run_tokens(packed,h,p,s,active,-1,epochs,ids))
    # Cache D's one-token prefix; the older four-token checkpoint is evicted.
    assert session.cache_token_prefix(3,epochs[3])
    stats=json.loads(session.token_prefix_stats_json());assert stats['entries']==2 and stats['evictions']==1
    assert session.release_request_row(0,epochs[0]);epochs[0]=session.claim_request_row(101)[1]
    assert session.restore_token_prefix(0,epochs[0],a)==2
    assert session.release_request_row(0,epochs[0]);epochs[0]=session.claim_request_row(102)[1]
    zero=snapshot(session);assert session.restore_token_prefix(0,epochs[0],[999,11,12])==0 and snapshot(session)==zero
    # A full reset clears live leases but preserves immutable cache owners.
    session.reset();lease=session.claim_request_row(103);assert lease[0]==0
    assert session.restore_token_prefix(0,lease[1],[41,42])==1
    wanted,hidden=reference([41]);compare_row(snapshot(session),wanted,0)
    assert session.predict_rows([lease[1],0,0,0])[0]==bank_prediction(hidden,0,0)
    assert session.clear_token_prefixes();stats=json.loads(session.token_prefix_stats_json())
    assert stats['entries']==0 and stats['accounted_host_bytes']==0 and stats['hits']>=5 and stats['misses']>=1
    # An independent same-shaped graph is not a registered/shared model family.
    rogue=garnet.compiled_graph();rogue.build(prefill(*symbols),symbols,f['checkpoint'])
    reject(lambda:session.prepare_prefill(rogue,1))
    h,p,s,active,ids=inputs([[42],[],[],[]],[1,0,0,0])
    reject(lambda:session.run_tokens([packed[0],rogue],h,p,s,active,-1,[lease[1],0,0,0],ids))
    # A too-small snapshot budget declines capture without breaking inference.
    budgeted=garnet.decoder_session();budgeted.build(base,head)
    assert budgeted.configure_token_prefix(identity,1,1024)
    small=budgeted.claim_request_row(1);small_epochs=[small[1],0,0,0]
    hidden=make([float(i%7)/17 for i in range(batch*width)],[batch,width])
    budgeted.run_tokens(base,hidden,make([0.0]*(batch*3),[batch,3]),make([0.0]*batch,[batch]),make([1.0,0.0,0.0,0.0],[batch]),0,small_epochs,[91,0,0,0])
    assert not budgeted.cache_token_prefix(0,small[1])
    assert json.loads(budgeted.token_prefix_stats_json())['entries']==0
    assert budgeted.predict_rows(small_epochs)[0]>=0
    assert session.cache_token_prefix(0,lease[1])
    assert session.release_request_row(0,lease[1]);replacement=session.claim_request_row(999)
    base[0].close()
    reject(lambda:session.restore_token_prefix(0,replacement[1],[41]))
    print('native_hybrid_token_prefix PASS content matching, longest stored boundary, exact/changed/append hits, recurrent+conv+KV+head parity, no future-KV leakage, A-to-E and B/C/D isolation, stale/untracked/invalid inputs, LRU, miss, reset and clear',flush=True)
finally:
    session.close();fresh.close()
    if budgeted is not None:budgeted.close()
    if rogue is not None:rogue.close()
    for graph in packed:graph.close()
    for graph in base:graph.close()
    if head is not None:head.close()
    garnet.shutdown()
