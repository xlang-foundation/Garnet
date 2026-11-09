# SPDX-License-Identifier: Apache-2.0
"""Shared BF16 attention graphs; no model-specific operators or weights."""
import garnet
T=garnet.tensor()

@T.fusion(name="bf16_attention_default_scale",boundary="required")
def attention(query,key,value):
    return query*T.unary_op("scaled_dot_product_attention",key=key,value=value,compute_dtype="bfloat16")

@T.fusion(name="bf16_attention_masked_gqa",boundary="required")
def masked_gqa(query,key,value,mask):
    return query*T.unary_op("scaled_dot_product_attention",key=key,value=value,mask=mask,scale=0.25,compute_dtype="bfloat16")

@T.fusion(name="bf16_attention_unscaled",boundary="required")
def unscaled(query,key,value):
    return query*T.unary_op("scaled_dot_product_attention",key=key,value=value,scale=1.0,compute_dtype="bfloat16")

@T.fusion(name="bf16_attention_long_sequence",boundary="required")
def long_sequence(query,key,value):
    return query*T.unary_op("scaled_dot_product_attention",key=key,value=value,compute_dtype="bfloat16")

@T.fusion(name="bf16_attention_invalid_rounding",boundary="required")
def invalid_rounding(query,key,value):
    return query*T.unary_op("scaled_dot_product_attention",key=key,value=value,compute_dtype="bfloat16",score_dtype="bfloat16")

@T.fusion(name="bf16_attention_invalid_dtype",boundary="required")
def invalid_dtype(query,key,value):
    return query*T.unary_op("scaled_dot_product_attention",key=key,value=value,compute_dtype="int32")
