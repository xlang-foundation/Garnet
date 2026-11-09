# SPDX-License-Identifier: Apache-2.0
"""Independent nonuniform SDPA means with 128-wide fused FP16 heads."""
import math,os,sys
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','..')))
import garnet,tensor
from xModel.generation.runtime import load,execute
root=os.path.abspath(sys.argv[1]);queries=513;keys=129;heads=2;dim=128
# The graph cache fingerprints Python dependencies. Give this backend test a
# separate cache after native lowering edits so stale engines cannot mask them.
backend_source=os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','..','src','backends','tensorrt','tensor_primitives.cpp'))
root=os.path.join(root,str(os.path.getmtime(backend_source)))
q=[float((i%3-1)*(h+1)) if d==0 else 0.0 for i in range(queries) for h in range(heads) for d in range(dim)]
k=[i/128 if d==0 else 0.0 for i in range(keys) for h in range(heads) for d in range(dim)]
v=[i/128+h/32+d/1024 for i in range(keys) for h in range(heads) for d in range(dim)]
inputs=[tensor.tensor(data,shape=shape,dtype=tensor.float32) for data,shape in [(q,[1,queries,heads,dim]),(k,[1,keys,heads,dim]),(v,[1,keys,heads,dim])]]
source=os.path.join(os.path.dirname(__file__),'fixtures','fused_fp16_attention_model.py')
for masked in (False,True):
    count=97 if masked else keys;expected=[]
    for i in range(queries):
        for h in range(heads):
            weights=[math.exp((i%3-1)*(h+1)*j/(128*math.sqrt(dim))) for j in range(count)]
            mean=sum(p*j/128 for j,p in enumerate(weights))/sum(weights)
            expected += [mean+h/32+d/1024 for d in range(dim)]
    values=inputs
    if masked:
        expected[:heads*dim]=[0.0]*(heads*dim)
        values=inputs+[tensor.tensor([-float('inf')]*keys+([0.0]*count+[-10000.0]*(keys-count))*(queries-1),shape=[1,1,queries,keys],dtype=tensor.float32)]
    model=load(source,'Masked' if masked else 'Fused',os.path.join(root,'masked' if masked else 'unmasked'),[list(x.shape) for x in values])
    try:actual=garnet.tensor_to_cpu(execute(model,values)).tolist()
    finally:model.release_runtime()
    assert all(math.isfinite(x) for x in actual)
    error=max(abs(a-b) for a,b in zip(actual,expected));assert error<6e-4,(masked,error)
    print('Fused FP16 attention independent mean reference',masked,'max_abs',error,flush=True)

# Four query heads share two KV heads. Their distinct value offsets make
# accidental broadcasting of the centering term observable.
queries=17;keys=33;heads=4;kv_heads=2
q=[float(h+1) if d==0 else 0.0 for i in range(queries) for h in range(heads) for d in range(dim)]
k=[j/32 if d==0 else 0.0 for j in range(keys) for h in range(kv_heads) for d in range(dim)]
v=[j/32+h/8+d/1024 for j in range(keys) for h in range(kv_heads) for d in range(dim)]
values=[tensor.tensor(data,shape=shape,dtype=tensor.float32) for data,shape in [(q,[1,queries,heads,dim]),(k,[1,keys,kv_heads,dim]),(v,[1,keys,kv_heads,dim])]]
expected=[]
for i in range(queries):
    for h in range(heads):
        weights=[math.exp((h+1)*j/(32*math.sqrt(dim))) for j in range(keys)]
        mean=sum(p*j/32 for j,p in enumerate(weights))/sum(weights)
        expected += [mean+(h//2)/8+d/1024 for d in range(dim)]
model=load(source,'Fused',os.path.join(root,'grouped'),[list(x.shape) for x in values])
try:actual=garnet.tensor_to_cpu(execute(model,values)).tolist()
finally:model.release_runtime()
error=max(abs(a-b) for a,b in zip(actual,expected));assert error<6e-4,error
print('Fused FP16 grouped-query centering independent reference max_abs',error,flush=True)
