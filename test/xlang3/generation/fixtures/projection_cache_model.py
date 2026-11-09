# SPDX-License-Identifier: Apache-2.0
from xModel.generation import tensor_ops as O
T=O.T
GARNET_MODEL_SPEC={'arguments':[{'name':'x','kind':'tensor'},{'name':'weights','kind':'weights'},{'name':'config','kind':'config'}]}

@T.fusion(name='projection_cache_admit',boundary='required')
def Cached(x,weights,config):return O.linear(x,'projection')

@T.fusion(name='projection_cache_bypass',boundary='required')
def Uncached(x,weights,config):return O.linear(x,'projection',cache_weights=False)

@T.fusion(name='projection_cache_invalid',boundary='required')
def Invalid(x,weights,config):return x*T.unary_op('linear',weight_name='projection.weight',cache_weights='no')
