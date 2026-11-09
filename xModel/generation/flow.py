# SPDX-License-Identifier: Apache-2.0
"""Shared native XLang3 flow-matching schedule, noise, and Euler graph."""
import math
import random
import tensor
import garnet
from .tensor_ops import T
GARNET_MODEL_SPEC={"arguments":[{"name":"latents","kind":"tensor"},
    {"name":"velocity","kind":"tensor"},{"name":"delta","kind":"tensor"}]}

def euler_sigmas(steps, mu=None, shift=None):
    if not isinstance(steps,int) or isinstance(steps,bool) or steps<1 or steps>10000:
        raise ValueError("Euler schedule requires 1..10000 steps")
    if mu is not None and shift is not None:
        raise ValueError("choose dynamic mu or static shift")
    factor=math.exp(mu) if mu is not None else (1.0 if shift is None else shift)
    if not math.isfinite(factor) or factor<=0: raise ValueError("flow shift must be positive and finite")
    return [factor/(factor+(1/sigma-1)) for sigma in [1-index/steps for index in range(steps)]]+[0.0]

def gaussian_noise(shape, seed):
    if not shape or any(not isinstance(size,int) or isinstance(size,bool) or size<1 for size in shape):
        raise ValueError("noise shape requires positive integer dimensions")
    if not isinstance(seed,int) or isinstance(seed,bool) or seed<0 or seed>=2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    if hasattr(garnet,'gaussian_noise'):return garnet.gaussian_noise(shape,str(seed))
    generator=random.Random(seed)
    return tensor.tensor([generator.gauss(0,1) for _ in range(math.prod(shape))],shape=shape,dtype=tensor.float32)

@T.fusion(name="flow_euler_step",boundary="required")
def FlowEulerStep(latents, velocity, delta):
    return latents+velocity*delta
