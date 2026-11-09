# SPDX-License-Identifier: Apache-2.0
"""Closed-form nonuniform attention across the 512+1 query boundary."""
import math,os,sys
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','..')))
import garnet,tensor
from xModel.generation.runtime import load,execute
root=os.path.abspath(sys.argv[1]);queries=513;keys=129;heads=2;dim=4
q=[float((i%3-1)*(h+1)) if d==0 else 0.0 for i in range(queries) for h in range(heads) for d in range(dim)]
k=[i/128 if d==0 else 0.0 for i in range(keys) for h in range(heads) for d in range(dim)]
v=[i/128+h/32+d/1024 for i in range(keys) for h in range(heads) for d in range(dim)]
inputs=[tensor.tensor(data,shape=shape,dtype=tensor.float32) for data,shape in [(q,[1,queries,heads,dim]),(k,[1,keys,heads,dim]),(v,[1,keys,heads,dim])]]
expected=[]
for i in range(queries):
    for h in range(heads):
        # Only the first Q/K coordinate is nonzero, so every output is an
        # independently calculated exponential-weighted mean of a key ramp.
        weights=[math.exp((i%3-1)*(h+1)*j/256) for j in range(keys)]
        mean=sum(p*j/128 for j,p in enumerate(weights))/sum(weights)
        expected += [mean+h/32+d/1024 for d in range(dim)]
model=load(os.path.join(os.path.dirname(__file__),'fixtures','query_tiled_attention_model.py'),'Tiled',root,[list(x.shape) for x in inputs])
try:value=garnet.tensor_to_cpu(execute(model,inputs))
finally:model.release_runtime()
assert list(value.shape)==[1,queries,heads,dim]
error=max(abs(a-b) for a,b in zip(value.tolist(),expected));assert error<3e-6,error
print('Query-tiled FP32 attention passed independent weighted-mean reference across 512+1 queries; maximum error',error,flush=True)
