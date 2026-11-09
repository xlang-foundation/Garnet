# SPDX-License-Identifier: Apache-2.0
# garnet-dependency: ../../../../xModel/generation/tensor_ops.py
from xModel.generation import tensor_ops as O
T=O.T
@T.fusion(name='fused_fp16_attention_reference',boundary='required')
def Fused(q,k,v):return O.attention(q,k,v,compute_dtype='float16')
@T.fusion(name='fused_fp16_masked_attention_reference',boundary='required')
def Masked(q,k,v,mask):return O.attention(q,k,v,mask=mask,compute_dtype='float16')
@T.fusion(name='fused_fp16_bf16_output_attention_reference',boundary='required')
def BF16Output(q,k,v):return O.round_bfloat16(O.attention(q,k,v,compute_dtype='float16'))
