# SPDX-License-Identifier: Apache-2.0
# garnet-dependency: ../../../../xModel/generation/attention.py
from xModel.generation.attention import query_tiled_attention
from xModel.generation import tensor_ops as O
T=O.T
@T.fusion(name='query_tile_boundary_reference',boundary='required')
def Tiled(q,k,v):return query_tiled_attention(q,k,v,q.shape[1],chunk=512)
