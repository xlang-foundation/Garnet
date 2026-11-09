# SPDX-License-Identifier: Apache-2.0
"""BF16 SDPA versus scalar reference, GQA, mask, scale, and long sequence."""
import math
import os
import struct
import sys
sys.path.insert(0,os.getcwd())
import garnet
import tensor
source=os.path.join(os.path.dirname(__file__),"attention_compute.py")
cache=os.path.abspath(sys.argv[1])
def bf16(value):
    bits=struct.unpack("<I",struct.pack("<f",value))[0]
    bits=(bits+0x7fff+((bits>>16)&1))&0xffff0000
    return struct.unpack("<f",struct.pack("<I",bits))[0]
def load(entry,shapes):
    return garnet.load_model(source,runtime_mode="compiled_xmodel",entry_function=entry,
        input_shapes=shapes,input_dtypes=["float32"]*len(shapes),cache_dir=os.path.join(cache,entry),
        backend="tensorrt",precision="bf16",compile={"builder_workspace_mb":64,"builder_optimization_level":0})
for entry,kv_heads,scale,masked in [("attention",4,1/math.sqrt(64),False),("unscaled",4,1.0,False),("masked_gqa",2,0.25,True)]:
    batch,qs,ks,heads,width=2,5,7,4,64
    shapes=[[batch,qs,heads,width],[batch,ks,kv_heads,width],[batch,ks,kv_heads,width]]
    data=[[0.6*math.sin(i*0.173+offset) for i in range(math.prod(shape))] for offset,shape in enumerate(shapes)]
    q,k,v=[[bf16(x) for x in values] for values in data]
    masks=[-10000.0 if j%ks==ks-1 else 0.125*(j%ks) for j in range(batch*qs*ks)]
    if masked:
        shapes.append([batch,1,qs,ks]);data.append(masks)
    expected=[]
    for b in range(batch):
        for row in range(qs):
            for head in range(heads):
                kh=head//(heads//kv_heads)
                scores=[]
                for col in range(ks):
                    dot=sum(q[((b*qs+row)*heads+head)*width+d]*k[((b*ks+col)*kv_heads+kh)*width+d] for d in range(width))
                    scores.append(dot*scale+(masks[(b*qs+row)*ks+col] if masked else 0))
                probabilities=[math.exp(score-max(scores)) for score in scores]
                total=sum(probabilities)
                for d in range(width):
                    expected.append(bf16(sum(p*v[((b*ks+col)*kv_heads+kh)*width+d] for col,p in enumerate(probabilities))/total))
    model=load(entry,shapes)
    try:
        assert model.runtime_status()["ready"],model.runtime_status()
        inputs=[tensor.tensor(values,shape=shape,dtype=tensor.float32) for values,shape in zip(data,shapes)]
        result=model.forward({"inputs":inputs})
        assert result["status"]=="ok",result
        host=garnet.tensor_to_cpu(result["output"])
        assert list(host.shape)==shapes[0]
        actual=host.tolist()
        largest=max(abs(a-b) for a,b in zip(actual,expected))
        assert all(math.isfinite(x) for x in actual) and largest<0.008,(entry,largest)
        print("BF16 attention scalar parity passed",entry,"max error",largest,flush=True)
    finally: model.release_runtime()
for entry in ("invalid_rounding","invalid_dtype"):
    model=load(entry,[[1,5,4,64],[1,7,4,64],[1,7,4,64]])
    try: assert not model.runtime_status()["ready"],entry
    finally: model.release_runtime()
shape=[1,8192,2,64]
model=load("long_sequence",[shape]*3)
try:
    assert model.runtime_status()["ready"],model.runtime_status()
    zeros=tensor.tensor([0.0]*math.prod(shape),shape=shape,dtype=tensor.float32)
    values=tensor.tensor([0.25]*math.prod(shape),shape=shape,dtype=tensor.float32)
    result=model.forward({"inputs":[zeros,zeros,values]})
    assert result["status"]=="ok",result
    host=garnet.tensor_to_cpu(result["output"])
    assert list(host.shape)==shape,host.shape
    assert all(abs(value-0.25)<0.000001 for value in host.tolist())
    print("BF16 attention 8192-token uniform reference passed",flush=True)
finally: model.release_runtime()
print("BF16 shared attention compute checks passed",flush=True)
