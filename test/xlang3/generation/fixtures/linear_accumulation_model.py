# SPDX-License-Identifier: Apache-2.0
# garnet-dependency: ../../../../xModel/generation/tensor_ops.py
from xModel.generation import tensor_ops as O
T=O.T
GARNET_MODEL_SPEC={'arguments':[{'name':'x','kind':'tensor'},{'name':'weights','kind':'weights'},{'name':'config','kind':'config'}]}
@T.fusion(name='linear_preserved_accumulation',boundary='required')
def Preserved(x,weights,config):return O.round_bfloat16(O.linear(x,'projection',bias=True,fp32_accumulation_output=True))
@T.fusion(name='linear_legacy_accumulation',boundary='required')
def Legacy(x,weights,config):return O.round_bfloat16(O.linear(x,'projection',bias=True))
@T.fusion(name='linear_float_accumulation',boundary='required')
def FloatOutput(x,weights,config):return O.linear(x,'projection',bias=True,fp32_accumulation_output=True)
