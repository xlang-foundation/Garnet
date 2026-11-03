"""Variable-length parallel prefill: native bank versus sequential token oracle.

Arguments: request-row vector fixture, native build, Garnet source.
"""
import sys,json,math
sys.path.insert(0,sys.argv[2]);sys.path.insert(0,sys.argv[3])
import garnet,tensor
from xModel.qwen35.vl_0_8b.decoder_layer import T,Qwen35FullDecoderLayer
from xModel.qwen35.vl_0_8b.prefill_layer import PrefillLayer
assert 'row_final_hidden' in garnet.backend_info()
with open(sys.argv[1]) as stream:f=json.load(stream)
c=f['config'];batch=f['batch'];width=c['hidden_size'];capacity=f['capacity'];chunk=4
assert batch==4
cache=[batch,c['num_key_value_heads'],capacity,c['head_dim']]
base=None;packed=None;head=None
parallel=garnet.decoder_session();sequential=garnet.decoder_session()
def make(values,shape):return tensor.tensor(values,shape=shape,dtype=tensor.float32)
def controls(values,slots,flags,wide):
    return (make(values,[batch,chunk,width] if wide else [batch,width]),
            make([float(s) for s in slots for axis in range(3)],[batch,chunk,3] if wide else [batch,3]),
            make([float(s) for s in slots],[batch,chunk] if wide else [batch]),
            make([float(a) for a in flags],[batch,chunk] if wide else [batch]))
def reject(epochs):
    before=[v.tolist() for v in parallel.snapshot(0)];failed=False
    try:parallel.predict_rows(epochs)
    except Exception:failed=True
    assert failed and before==[v.tolist() for v in parallel.snapshot(0)]
try:
    shapes=[[batch,width],cache,cache,[batch,3],[batch],[batch]]
    inputs=[tensor.input('row_bank_base_'+str(i),shape=s,dtype=tensor.float32) for i,s in enumerate(shapes)]
    @T.fusion(name='row_bank_base_attention',boundary='required')
    def decode(hidden,key,value,positions,slots,active):return Qwen35FullDecoderLayer(hidden,key,value,positions,slots,active,c,1,capacity)
    base=garnet.compiled_graph();base.build(decode(*inputs),inputs,f['checkpoint'])
    shapes=[[batch,chunk,width],cache,cache,[batch,chunk,3],[batch,chunk],[batch,chunk]]
    inputs=[tensor.input('row_bank_packed_'+str(i),shape=s,dtype=tensor.float32) for i,s in enumerate(shapes)]
    @T.fusion(name='row_bank_packed_attention',boundary='required')
    def prefill(hidden,key,value,positions,slots,active):return PrefillLayer(hidden,key,value,positions,slots,active,c,1,batch,chunk,capacity,'full_attention',parallel_attention=True)
    packed=garnet.compiled_graph();packed.build_shared(prefill(*inputs),inputs,f['checkpoint'],base)
    symbol=tensor.input('row_bank_head',shape=[batch,width],dtype=tensor.float32)
    @T.fusion(name='row_bank_prediction',boundary='required')
    def predict(hidden):return hidden * T.binary_op('add') * hidden
    head=garnet.compiled_graph();head.build(predict(symbol),[symbol],f['checkpoint'])
    parallel.build([base],head);sequential.build([base],head);parallel.prepare_prefill(packed,0)
    leases=[parallel.claim_request_row(row+1) for row in range(batch)]
    epochs=[lease[1] for lease in leases];reject(epochs)
    expected=[-1]*batch;ready=[False]*batch
    patterns=[[1,1,1,0, 1,0,0,0, 1,1,1,1, 0,0,0,0],
              [1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
              [0,0,0,0, 0,1,0,1, 0,0,0,0, 1,0,0,0]]
    for index,flags in enumerate(patterns):
        values=[float((i*13+index*17)%43-21)/53 for i in range(batch*chunk*width)]
        slots=[(index*chunk+t)%capacity for row in range(batch) for t in range(chunk)]
        h,p,s,a=controls(values,slots,flags,True)
        parallel.run_leased([packed],h,p,s,a,-1,epochs)
        for token in range(chunk):
            selected=[];pos=[];active=[]
            for row in range(batch):
                start=(row*chunk+token)*width;selected+=values[start:start+width]
                pos.append(slots[row*chunk+token]);active.append(flags[row*chunk+token])
            ht,pt,st,at=controls(selected,pos,active,False)
            predicted=sequential.run([base],ht,pt,st,at,0)
            for row in range(batch):
                if active[row]:expected[row]=predicted[row];ready[row]=True
        selected_epochs=[epochs[row] if ready[row] else 0 for row in range(batch)]
        assert parallel.predict_rows(selected_epochs)==expected,(index,parallel.predict_rows(selected_epochs),expected)
        if not ready[3]:reject(epochs)
        reject([0]*batch);reject([-1,0,0,0]);reject(epochs[:-1])
    old=epochs[1];assert parallel.release_request_row(1,old)
    replacement=parallel.claim_request_row(99);assert replacement[0]==1 and replacement[1]!=old
    reject(epochs);epochs[1]=replacement[1];reject(epochs)
    epochs[1]=0;wanted=list(expected);wanted[1]=-1
    assert parallel.predict_rows(epochs)==wanted
    parallel.reset();reject(epochs)
    print('row_final_hidden PASS different final indices, padding and inactive-bank retention, sequential oracle, stale/replaced/uninitialized/invalid epochs, full reset',flush=True)
finally:
    parallel.close();sequential.close()
    if packed is not None:packed.close()
    if base is not None:base.close()
    if head is not None:head.close()
    garnet.shutdown()
