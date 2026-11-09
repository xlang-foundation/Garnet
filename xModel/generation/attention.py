# SPDX-License-Identifier: Apache-2.0
"""Memory-bounded FP32 attention expressed entirely inside a compiled graph."""
from . import tensor_ops as O
T=O.T
@T.fusion(role='fp32_attention_query_tile',atomic=True)
def tile(query,key,value,mask,scale):return O.attention(query,key,value,mask=mask,scale=scale)

def query_tiled_attention(query,key,value,queries,chunk=128,mask=None,scale=None):
    if queries<1 or chunk<1:raise ValueError('attention queries and tile size must be positive')
    result=None
    # These loops expand graph nodes at compile time. Model inference remains
    # one native engine call; no script controls individual query tiles.
    for start in range(0,queries,chunk):
        count=min(chunk,queries-start);part=O.slice(query,1,start,count)
        part=tile(part,key,value,O.slice(mask,2,start,count) if mask is not None else None,scale)
        result=part if result is None else O.concat(result,part,1)
    return result
