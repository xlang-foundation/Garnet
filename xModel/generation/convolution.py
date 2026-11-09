# SPDX-License-Identifier: Apache-2.0
"""Convolution with bias accumulated before BF16 activation rounding."""
import tensor
from . import tensor_ops as O

def convolution(value,prefix,weights,spatial,bias=True,transposed=False,compute_dtype=None,**geometry):
    if spatial not in (1,2,3) or (transposed and spatial!=1):raise ValueError('unsupported convolution dimensions')
    if compute_dtype not in (None,'float32','float16'):raise ValueError('unsupported convolution compute dtype')
    bf16=weights[prefix+'.weight'].dtype==tensor.bfloat16 and compute_dtype is None
    # Native bias input 2 accumulates before output rounding. Keep checkpoint
    # weights in BF16 rather than materializing a second FP32 weight copy.
    if bf16:value=O.round_bfloat16(value)
    options={'weight_name':prefix+'.weight','fuse_bias':1}
    if bias:options['bias_name']=prefix+'.bias'
    if compute_dtype is not None:
        options={'weight':O.cast(weights[prefix+'.weight'],compute_dtype),'fuse_bias':1}
        if bias:options['bias']=O.cast(weights[prefix+'.bias'],compute_dtype)
    options.update(geometry)
    result=value*O.T.unary_op('conv_transpose1d' if transposed else 'conv'+str(spatial)+'d',**options)
    return O.round_bfloat16(result) if bf16 else result
