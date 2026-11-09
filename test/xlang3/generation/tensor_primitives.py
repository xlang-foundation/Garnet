# SPDX-License-Identifier: Apache-2.0
"""Weight-free graphs for the reusable operations needed by image/video models."""
import garnet
T = garnet.tensor()

@T.fusion(name="generation_layer_norm", boundary="required")
def norm(x):
    return x * T.unary_op("layer_norm_unaffine", eps=0.000001)

@T.fusion(name="generation_swiglu", boundary="required")
def swiglu(x):
    return x * T.unary_op("swiglu")

@T.fusion(name="generation_modulation", boundary="required")
def modulate(x, scale, shift):
    return x * T.unary_op("modulate", scale=scale, shift=shift)

@T.fusion(name="generation_rotary", boundary="required")
def rotary(x, cosine, sine):
    return x * T.unary_op("rotary_interleaved", cos=cosine, sin=sine)

@T.fusion(name="generation_attention", boundary="required")
def attention(query, key, value):
    return query * T.unary_op("scaled_dot_product_attention", key=key, value=value)

@T.fusion(name="generation_embedding", boundary="required")
def embedding(timestep):
    return timestep * T.unary_op("sinusoidal_embedding", width=8, scale=1000.0)

@T.fusion(name="generation_slice_concat", boundary="required")
def slice_concat(left, right):
    a = left * T.unary_op("tensor_slice", axis=-1, start=1, count=2)
    return (a * T.binary_op("tensor_concat", axis=-1)) * right

@T.fusion(name="generation_invalid_slice", boundary="required")
def invalid_slice(x):
    return x * T.unary_op("tensor_slice", axis=-1, start=2, count=3)

@T.fusion(name="generation_invalid_axis", boundary="required")
def invalid_axis(x):
    return x * T.unary_op("tensor_slice", axis=2, start=0, count=1)

@T.fusion(name="generation_group_norm", boundary="required")
def group_norm(x,weight,bias):
    return x * T.unary_op("group_norm",weight=weight,bias=bias,groups=2,eps=0.000001)

@T.fusion(name="generation_conv2d", boundary="required")
def conv2d(x,weight,bias):
    return x * T.unary_op("conv2d",weight=weight,bias=bias,groups=2)

@T.fusion(name="generation_conv3d", boundary="required")
def conv3d(x,weight,bias):
    return x * T.unary_op("conv3d",weight=weight,bias=bias)

@T.fusion(name="generation_upsample", boundary="required")
def upsample(x):
    return x * T.unary_op("upsample_nearest2d",factor=2)

@T.fusion(name="generation_rotary_split",boundary="required")
def rotary_split(x,cosine,sine):
    return x * T.unary_op("rotary_split_half",cos=cosine,sin=sine)

@T.fusion(name="generation_masked_gqa",boundary="required")
def masked_gqa(query,key,value,mask):
    return query * T.unary_op("scaled_dot_product_attention",key=key,value=value,mask=mask)

@T.fusion(name="generation_round_bfloat16",boundary="required")
def round_bfloat16(x):
    return (x * T.unary_op("cast",dtype="bfloat16")) * T.unary_op("cast",dtype="float32")

@T.fusion(name="generation_scalar_math",boundary="required")
def scalar_math(x):
    return (2.0+x)*0.5-1.0

@T.fusion(name="generation_rank_broadcast",boundary="required")
def rank_broadcast(x,y):
    return x+y

@T.fusion(name="generation_conv2d_asymmetric",boundary="required")
def conv2d_asymmetric(x,weight):
    return x * T.unary_op("conv2d",weight=weight,strides=[2,2],padding=[0,0],padding_after=[1,1])

@T.fusion(name="generation_conv3d_causal",boundary="required")
def conv3d_causal(x,weight):
    return x * T.unary_op("conv3d",weight=weight,padding=[2,0,0],padding_after=[0,0,0])

@T.fusion(name="generation_unscaled_bf16_attention",boundary="required")
def unscaled_bf16_attention(query,key,value,mask):
    return query * T.unary_op("scaled_dot_product_attention",key=key,value=value,mask=mask,
        scale=1.0,score_dtype="bfloat16",probability_dtype="bfloat16")

@T.fusion(name="generation_l2_normalize",boundary="required")
def l2_normalize(x):
    return x*T.unary_op("l2_normalize",eps=1e-12)

@T.fusion(name="generation_reduce_mean",boundary="required")
def reduce_mean(x):
    return x*T.unary_op("reduce_mean",axis=1)

@T.fusion(name="generation_l2_channel_axis",boundary="required")
def l2_channel_axis(x):
    return x*T.unary_op("l2_normalize",axis=1,eps=1e-12)
