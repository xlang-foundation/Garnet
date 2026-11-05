"""Compare chunk cache execution with sequential decode, including overwrites.

XLang arguments: request-row fixture, native build, source. Uses real full
attention kernels and the same weights; independent row math is tested apart.
"""
import sys,json,math
sys.path.insert(0,sys.argv[2]);sys.path.insert(0,sys.argv[3])
import garnet,tensor
from xModel.qwen35.vl_0_8b.decoder_layer import T,Qwen35FullDecoderLayer
from xModel.qwen35.vl_0_8b.prefill_layer import PrefillLayer
with open(sys.argv[1]) as stream:f=json.load(stream)
c=f['config'];batch=f['batch'];width=c['hidden_size'];capacity=f['capacity'];chunk=4
cache=[batch,c['num_key_value_heads'],capacity,c['head_dim']]
base=None;packed=None;parallel=garnet.decoder_session();sequential=garnet.decoder_session()
enabled='chunk_kv_inplace' in garnet.backend_info()
def make(values,shape):return tensor.tensor(values,shape=shape,dtype=tensor.float32)
def snap(owner):return [value.tolist() for value in owner.snapshot(0)]
def near(a,b):
    assert len(a)==len(b)
    for x,y in zip(a,b):assert math.isfinite(x) and abs(x-y)<=5e-6+5e-5*abs(y),(x,y)
def controls(values,slots,flags,packed_shape):
    positions=[float(slot) for slot in slots for axis in range(3)]
    return (make(values,[batch,chunk,width] if packed_shape else [batch,width]),
            make(positions,[batch,chunk,3] if packed_shape else [batch,3]),
            make([float(v) for v in slots],[batch,chunk] if packed_shape else [batch]),
            make([float(v) for v in flags],[batch,chunk] if packed_shape else [batch]))
try:
    shapes=[[batch,width],cache,cache,[batch,3],[batch],[batch]]
    inputs=[tensor.input('chunk_cache_base_'+str(i),shape=s,dtype=tensor.float32) for i,s in enumerate(shapes)]
    @T.fusion(name='chunk_cache_base_attention',boundary='required')
    def decode(hidden,key,value,positions,slots,active):return Qwen35FullDecoderLayer(hidden,key,value,positions,slots,active,c,1,capacity)
    base=garnet.compiled_graph();base.build(decode(*inputs),inputs,f['checkpoint'])
    shapes=[[batch,chunk,width],cache,cache,[batch,chunk,3],[batch,chunk],[batch,chunk]]
    inputs=[tensor.input('chunk_cache_packed_'+str(i),shape=s,dtype=tensor.float32) for i,s in enumerate(shapes)]
    @T.fusion(name='chunk_cache_packed_attention',boundary='required')
    def prefill(hidden,key,value,positions,slots,active):
        return PrefillLayer(hidden,key,value,positions,slots,active,c,1,batch,chunk,capacity,'full_attention',parallel_attention=True)
    packed=garnet.compiled_graph();packed.build_shared(prefill(*inputs),inputs,f['checkpoint'],base)
    parallel.build([base],None);sequential.build([base],None)
    if enabled:parallel.prepare_prefill(packed,0)
    # Populate old slots first: a later chunk overwrite must not leak backward
    # into an earlier query's attention over this old cache.
    for slot in range(4):
        values=[float((i*7+slot*11)%31-15)/41 for i in range(batch*width)]
        h,p,s,a=controls(values,[slot]*batch,[1]*batch,False)
        parallel.run([base],h,p,s,a,-1);sequential.run([base],h,p,s,a,-1)
    cases=[
        ([4,5,6,7,4,5,0,0,0,0,0,0,1,0,1,3],
         [1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1]),
        ([0,0,1,0,0,2,0,3,4,5,2,6,0,0,0,0],
         [1,1,1,1,0,1,0,1,1,1,1,1,0,0,0,0])]
    for index,(slots,flags) in enumerate(cases):
        values=[float((i*13+index*17)%43-21)/53 for i in range(batch*chunk*width)]
        h,p,s,a=controls(values,slots,flags,True)
        initial=parallel.snapshot(0);host_before=[v.tolist() for v in initial]
        legacy=packed.run([h,initial[0],initial[1],p,s,a]).tolist()
        assert host_before==[v.tolist() for v in initial]
        parallel.run([packed],h,p,s,a,-1)
        assert parallel.kv_inplace_updates==(2 if enabled else 0)
        expected=[0.0]*(batch*chunk*width)
        for token in range(chunk):
            selected=[];positions=[];active=[]
            for row in range(batch):
                start=(row*chunk+token)*width;selected+=values[start:start+width]
                positions.append(slots[row*chunk+token]);active.append(flags[row*chunk+token])
            ht,pt,st,at=controls(selected,positions,active,False)
            sequential.run([base],ht,pt,st,at,-1);output=sequential.hidden().tolist()
            for row in range(batch):
                start=(row*chunk+token)*width;expected[start:start+width]=output[row*width:(row+1)*width]
        near(parallel.hidden().tolist(),expected)
        actual=snap(parallel);wanted=snap(sequential)
        for x,y in zip(actual,wanted):near(x,y)
        # Standalone packed output is row-wise hidden + complete K + complete V.
        collected=[];columns=len(actual[0])//batch
        for row in range(batch):
            collected+=expected[row*chunk*width:(row+1)*chunk*width]
            for part in actual:collected+=part[row*columns:(row+1)*columns]
        near(legacy,collected)
    print('chunk_kv_inplace PASS append, old-cache overwrites, duplicate slots, inactive/reactivated rows, sequential and standalone parity; enabled',enabled,flush=True)
finally:
    parallel.close();sequential.close()
    if packed is not None:packed.close()
    if base is not None:base.close()
    garnet.shutdown()
