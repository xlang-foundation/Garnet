"""Exercise shared scratch across four mixed layers, sessions and legacy calls.

XLang arguments: existing request-row fixture, native build, source root.
Kernel equations are covered by the independent request-row reference; this
test checks storage changes against execution before scratch sharing.
"""
import sys,json
sys.path.insert(0,sys.argv[2]);sys.path.insert(0,sys.argv[3])
import garnet,tensor
from xModel.qwen35.vl_0_8b.decoder_layer import T,Qwen35LinearDecoderLayer,Qwen35FullDecoderLayer
from xModel.qwen35.vl_0_8b.prefill_layer import PrefillLayer
with open(sys.argv[1]) as stream:f=json.load(stream)
c=f['config'];batch=f['batch'];width=c['hidden_size'];capacity=f['capacity'];chunk=4
assert 'shared_prefill_scratch' in garnet.backend_info()
base=[];packed=[];session=garnet.decoder_session();other=garnet.decoder_session()
state_shapes=[[[batch,29,4],[batch,3,7,5]],[[batch,1,capacity,6],[batch,1,capacity,6]]]
def zeros(shape):
    count=1
    for value in shape:count*=value
    return tensor.tensor([0.0]*count,shape=shape,dtype=tensor.float32)
def snapshot(owner):return [[value.tolist() for value in owner.snapshot(i)] for i in range(4)]
def symbols(shapes,name):return [tensor.input(name+str(i),shape=shape,dtype=tensor.float32) for i,shape in enumerate(shapes)]
try:
    for i in range(4):
        layer=i%2;states=state_shapes[layer]
        shapes=[[batch,width]]+states+([[batch]] if layer==0 else [[batch,3],[batch],[batch]])
        inputs=symbols(shapes,'scratch_base_'+str(i)+'_')
        if layer==0:
            @T.fusion(name='scratch_base_linear_'+str(i),boundary='required')
            def forward(hidden,conv,state,active):return Qwen35LinearDecoderLayer(hidden,conv,state,active,c,0)
        else:
            @T.fusion(name='scratch_base_attention_'+str(i),boundary='required')
            def forward(hidden,key,value,positions,slots,active):return Qwen35FullDecoderLayer(hidden,key,value,positions,slots,active,c,1,capacity)
        graph=garnet.compiled_graph();graph.build(forward(*inputs),inputs,f['checkpoint']);base.append(graph)
        shapes=[[batch,chunk,width]]+states+[[batch,chunk,3],[batch,chunk],[batch,chunk]]
        inputs=symbols(shapes,'scratch_packed_'+str(i)+'_')
        kind='linear_attention' if layer==0 else 'full_attention'
        @T.fusion(name='scratch_packed_layer_'+str(i),boundary='required')
        def prefill(hidden,a,b,positions,slots,active):
            return PrefillLayer(hidden,a,b,positions,slots,active,c,layer,batch,chunk,capacity,kind,
                                parallel_attention=True,parallel_linear=True)
        graph=garnet.compiled_graph();graph.build_shared(prefill(*inputs),inputs,f['checkpoint'],base[-1]);packed.append(graph)
    session.build(base,None);other.build(base,None)
    hidden=tensor.tensor([float((i%17)-8)/37 for i in range(batch*chunk*width)],shape=[batch,chunk,width],dtype=tensor.float32)
    slots=tensor.tensor([float(t) for row in range(batch) for t in range(chunk)],shape=[batch,chunk],dtype=tensor.float32)
    positions=tensor.tensor([float(t) for row in range(batch) for t in range(chunk) for axis in range(3)],shape=[batch,chunk,3],dtype=tensor.float32)
    active=tensor.tensor([1.,1.,1.,0.,1.,0.,0.,0.,1.,1.,1.,1.,0.,0.,0.,0.],shape=[batch,chunk],dtype=tensor.float32)
    session.run(packed,hidden,positions,slots,active,-1)
    wanted_hidden=session.hidden().tolist();wanted_state=snapshot(session)
    legacy=[]
    for i,graph in enumerate(packed):
        states=state_shapes[i%2]
        legacy.append(graph.run([hidden,zeros(states[0]),zeros(states[1]),positions,slots,active]).tolist())
    session.reset()
    for i,graph in enumerate(packed):session.prepare_prefill(graph,i)
    assert sum(graph.shared_prefill_scratch_bytes for graph in packed)>0
    session.run(packed,hidden,positions,slots,active,-1)
    assert session.hidden().tolist()==wanted_hidden and snapshot(session)==wanted_state
    other.run(packed,hidden,positions,slots,active,-1)
    assert other.hidden().tolist()==wanted_hidden and snapshot(other)==wanted_state
    assert snapshot(session)==wanted_state
    rejected=False
    try:other.prepare_prefill(packed[0],0)
    except Exception:rejected=True
    assert rejected and snapshot(session)==wanted_state and snapshot(other)==wanted_state
    for i in (3,1,2,0):
        states=state_shapes[i%2]
        assert packed[i].run([hidden,zeros(states[0]),zeros(states[1]),positions,slots,active]).tolist()==legacy[i]
        assert snapshot(session)==wanted_state and snapshot(other)==wanted_state
    session.close();other.close()
    # Pool lifetime is owned by the graphs, not a dangling session reference.
    for i in (0,2,1,3):
        states=state_shapes[i%2]
        assert packed[i].run([hidden,zeros(states[0]),zeros(states[1]),positions,slots,active]).tolist()==legacy[i]
    print('shared_prefill_scratch PASS mixed4 layers, padded/inactive rows, exact pre-share parity, private sessions, foreign-owner rejection, standalone calls and pool lifetime; reused_bytes',sum(graph.shared_prefill_scratch_bytes for graph in packed),flush=True)
finally:
    session.close();other.close()
    for graph in packed+base:graph.close()
    garnet.shutdown()
