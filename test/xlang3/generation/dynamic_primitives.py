# SPDX-License-Identifier: Apache-2.0
from xModel.generation import tensor_ops as O
T=O.T

def body(x,cosine,sine,length):
    q=O.rotary_split_half(O.reshape(x,[1,length,2,16]),cosine,sine)
    q=O.rms_norm_unaffine(q,1e-6)
    value=O.attention(q,q,q)
    return O.reshape(value*T.unary_op('layer_norm_unaffine',eps=1e-6),[1,length,32])

@T.fusion(name='dynamic_sequence_primitives')
def forward(x,cosine,sine):return body(x,cosine,sine,-1)

@T.fusion(name='static_sequence_primitives')
def fixed(x,cosine,sine):return body(x,cosine,sine,x.shape[1])

@T.fusion(name='dynamic_sequence_bf16_attention')
def fused(x,cosine,sine):
    q=O.rotary_split_half(O.reshape(x,[1,-1,2,16]),cosine,sine)
    return O.reshape(O.attention(q,q,q,compute_dtype='bfloat16'),[1,-1,32])

@T.fusion(name='invalid_multiple_inferred_axes')
def invalid(x):return O.reshape(x,[1,-1,-1])
