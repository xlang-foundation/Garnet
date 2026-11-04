"""Prefill pack retirement: standalone restoration and two-session isolation.

Arguments: persisted small request-row fixture, native build, Garnet source.
No model timing or performance claim. Exercises the actual native pack path.
"""
import sys,json,math
sys.path.insert(0,sys.argv[2]);sys.path.insert(0,sys.argv[3])
import garnet,tensor
from xModel.qwen35.vl_0_8b.decoder_layer import T,Qwen35LinearDecoderLayer,Qwen35FullDecoderLayer
from xModel.qwen35.vl_0_8b.prefill_layer import PrefillLayer
assert 'retired_nonresident_packs' in garnet.backend_info()
with open(sys.argv[1]) as stream:f=json.load(stream)
c=f['config'];batch=f['batch'];width=c['hidden_size'];capacity=f['capacity'];chunk=2
def f32(values,shape):return tensor.tensor(values,shape=shape,dtype=tensor.float32)
def near(a,b):
    assert len(a)==len(b)
    for x,y in zip(a,b):assert math.isfinite(x) and abs(x-y)<=5e-6+5e-5*abs(y),(x,y)
def states(owner):return [value.tolist() for value in owner.snapshot(0)]
def collect(owner):
    parts=[owner.hidden().tolist()]+states(owner);result=[]
    for row in range(batch):
        for part in parts:result+=part[row*(len(part)//batch):(row+1)*(len(part)//batch)]
    return result
try:
    for layer,kind in enumerate(['linear_attention','full_attention']):
        shapes=([[batch,width],[batch,29,4],[batch,3,7,5],[batch]] if layer==0 else
            [[batch,width],[batch,1,capacity,6],[batch,1,capacity,6],[batch,3],[batch],[batch]])
        base=garnet.compiled_graph();graph=garnet.compiled_graph()
        session=garnet.decoder_session();other=garnet.decoder_session()
        try:
            symbols=[tensor.input('pack_base_'+str(layer)+'_'+str(i),shape=shape,dtype=tensor.float32) for i,shape in enumerate(shapes)]
            if layer==0:
                @T.fusion(name='pack_base_linear',boundary='required')
                def forward(h,s1,s2,a):return Qwen35LinearDecoderLayer(h,s1,s2,a,c,layer)
            else:
                @T.fusion(name='pack_base_full',boundary='required')
                def forward(h,s1,s2,p,s,a):return Qwen35FullDecoderLayer(h,s1,s2,p,s,a,c,layer,capacity)
            base.build(forward(*symbols),symbols,f['checkpoint']);session.build([base],None);other.build([base],None)
            chunk_shapes=[[batch,chunk,width],shapes[1],shapes[2],[batch,chunk,3],[batch,chunk],[batch,chunk]]
            symbols=[tensor.input('pack_chunk_'+str(layer)+'_'+str(i),shape=shape,dtype=tensor.float32) for i,shape in enumerate(chunk_shapes)]
            @T.fusion(name='pack_chunk',boundary='required')
            def prefill(h,s1,s2,p,s,a):return PrefillLayer(h,s1,s2,p,s,a,c,layer,batch,chunk,capacity,kind)
            graph.build_shared(prefill(*symbols),symbols,f['checkpoint'],base)
            session.prepare_prefill(graph,0)
            assert graph.nonresident_pack_slots>0
            if 'retired_prefill_hidden_inputs' in garnet.backend_info():assert graph.retired_prefill_hidden_input_slots==1
            for repeat in range(3):
                session.reset();other.reset()
                # The previous standalone call rehydrates defaults. Explicitly
                # retire them again before testing the next restoration cycle.
                session.prepare_prefill(graph,0)
                h=f32([float((i*7+repeat*3)%19-9)/23 for i in range(batch*chunk*width)],[batch,chunk,width])
                flags=[float((row+t+repeat)%3!=0) for row in range(batch) for t in range(chunk)]
                slots=[float(t) if flags[row*chunk+t] else 0.0 for row in range(batch) for t in range(chunk)]
                p=f32([value for value in slots for axis in range(3)],[batch,chunk,3])
                s=f32(slots,[batch,chunk]);a=f32(flags,[batch,chunk]);state=session.snapshot(0)
                inputs=[h,state[0],state[1],p,s,a];host=[value.tolist() for value in inputs]
                before=graph.lazy_default_rehydrated_bytes;actual=graph.run(inputs).tolist()
                assert graph.lazy_default_rehydrated_bytes>before
                session.run([graph],h,p,s,a,-1);near(actual,collect(session))
                saved=states(session);hidden=session.hidden().tolist()
                other.run([graph],h,p,s,a,-1);near(actual,collect(other))
                assert saved==states(session) and hidden==session.hidden().tolist()
                near(actual,graph.run(inputs).tolist())
                assert host==[value.tolist() for value in inputs] and saved==states(session)
            print('nonresident_pack_reference PASS',kind,'retired_slots',graph.nonresident_pack_slots,'repeats',3,flush=True)
        finally:session.close();other.close();graph.close();base.close()
finally:garnet.shutdown()
