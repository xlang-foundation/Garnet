# SPDX-License-Identifier: Apache-2.0
"""Backend-neutral tensor operations shared by text, image, and video models."""
import garnet
T = garnet.tensor()

def linear(value, prefix, bias=False, fp32_accumulation_output=False, cache_weights=True, projection_compute=None):
    """Optional accumulation boundary and backend weight-residency hint."""
    options={'weight_name':prefix+'.weight'}
    if bias:options['bias_name']=prefix+'.bias'
    if fp32_accumulation_output:options['accumulation_output_dtype']='float32'
    if cache_weights is not True:options['cache_weights']=cache_weights
    if projection_compute is not None:options['projection_compute']=projection_compute
    return value*T.unary_op('linear',**options)

def embedding(value, prefix, preserve_dtype=True):
    return value * T.unary_op("embedding",weight_name=prefix+".weight",preserve_dtype=preserve_dtype)

def cast(value, dtype):
    return value * T.unary_op("cast",dtype=dtype)

def round_bfloat16(value):
    return cast(cast(value,"bfloat16"),"float32")

def reshape(value, shape):
    return value * T.unary_op("reshape", shape=shape)

def permute(value, axes):
    return value * T.unary_op("permute", axes=axes)

def slice(value, axis, start, count):
    return value * T.unary_op("tensor_slice", axis=axis, start=start, count=count)

def concat(left, right, axis):
    return (left * T.binary_op("tensor_concat", axis=axis)) * right

def silu(value):
    return value * T.unary_op("silu")

def sqrt(value):
    return value * T.unary_op("sqrt")

def l2_normalize(value,eps=1e-12,axis=-1):
    return value * T.unary_op("l2_normalize",eps=eps,axis=axis)

def reduce_mean(value,axis=-1):
    return value * T.unary_op("reduce_mean",axis=axis)

def reduce_sum(value,axis=-1): return value*T.unary_op("reduce_sum",axis=axis)
def reduce_min(value,axis=-1): return value*T.unary_op("reduce_min",axis=axis)
def reduce_max(value,axis=-1): return value*T.unary_op("reduce_max",axis=axis)
def gather(value,indices,axis=0): return value*T.unary_op("tensor_gather",indices=indices,axis=axis)
def rms_norm_unaffine(value,eps=1e-6,axis=-1): return value*T.unary_op("rms_norm_unaffine",eps=eps,axis=axis)

def pixel_norm(value,eps=1e-6,axis=1,bfloat16=False):
    # Pixel normalization preserves the source dtype at each arithmetic step.
    # Gemma/transformer RMS normalization instead accumulates in float32.
    if not bfloat16:return rms_norm_unaffine(value,eps,axis)
    variance=round_bfloat16(reduce_mean(round_bfloat16(value*value),axis))
    denominator=round_bfloat16(sqrt(round_bfloat16(variance+eps)))
    return round_bfloat16(value/denominator)
def leaky_relu(value,negative_slope=0.01): return value*T.unary_op("leaky_relu",negative_slope=negative_slope)
def tanh(value): return value*T.unary_op("tanh")
def sin(value): return value*T.unary_op("sin")
def exp(value): return value*T.unary_op("exp")
def log(value): return value*T.unary_op("log")
def sigmoid(value): return value*T.unary_op("sigmoid")
def conv1d(value,prefix,stride=1,padding=0,dilation=1,bias=True,transposed=False):
    options={"weight_name":prefix+".weight","stride":stride,"padding":padding,"dilation":dilation}
    if bias: options["bias_name"]=prefix+".bias"
    return value*T.unary_op("conv_transpose1d" if transposed else "conv1d",**options)

def swiglu(value):
    return value * T.unary_op("swiglu")

def layer_norm(value, eps):
    return value * T.unary_op("layer_norm_unaffine", eps=eps)

def rms_norm(value, prefix, eps, round_before_scale=False):
    return value * T.unary_op("rms_norm", weight_name=prefix+".weight", eps=eps,round_before_scale=round_before_scale)

def modulate(value, shift, scale):
    return value * T.unary_op("modulate", shift=shift, scale=scale)

def rotary(value, cosine, sine):
    return value * T.unary_op("rotary_interleaved", cos=cosine, sin=sine)

def rotary_split_half(value, cosine, sine):
    return value * T.unary_op("rotary_split_half",cos=cosine,sin=sine)

def attention(query, key, value, mask=None,scale=None,score_dtype=None,probability_dtype=None,compute_dtype=None):
    """SDPA with FP32 accumulation; optional BF16 operands/output permit fusion.

    Separate score/probability rounding describes explicit intermediate tensors
    and cannot be combined with BF16 compute, which may keep them fused.
    """
    options={"key":key,"value":value}
    if mask is not None: options["mask"]=mask
    if scale is not None: options["scale"]=scale
    if score_dtype is not None: options["score_dtype"]=score_dtype
    if probability_dtype is not None: options["probability_dtype"]=probability_dtype
    if compute_dtype is not None: options["compute_dtype"]=compute_dtype
    return query * T.unary_op("scaled_dot_product_attention",**options)

def gelu_tanh(value):
    return value * T.unary_op("gelu_tanh")

def sinusoidal(timestep, width, scale=1.0, max_period=10000.0):
    return timestep * T.unary_op("sinusoidal_embedding", width=width, scale=scale, max_period=max_period)

def conv2d(value, prefix, strides=(1,1), padding=(0,0), bias=True, padding_after=None):
    options={"weight_name":prefix+".weight","strides":strides,"padding":padding}
    if padding_after is not None: options["padding_after"]=padding_after
    if bias:
        options["bias_name"]=prefix+".bias"
    return value * T.unary_op("conv2d",**options)

def conv3d(value,prefix,strides=(1,1,1),padding=(0,0,0),bias=True,padding_after=None):
    options={"weight_name":prefix+".weight","strides":strides,"padding":padding}
    if padding_after is not None: options["padding_after"]=padding_after
    if bias: options["bias_name"]=prefix+".bias"
    return value * T.unary_op("conv3d",**options)

def group_norm(value, prefix, groups=32, eps=0.000001):
    return value * T.unary_op("group_norm",weight_name=prefix+".weight",bias_name=prefix+".bias",groups=groups,eps=eps)

def upsample_nearest2d(value, factor=2):
    return value * T.unary_op("upsample_nearest2d",factor=factor)
