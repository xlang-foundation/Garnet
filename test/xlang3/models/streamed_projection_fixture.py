# SPDX-License-Identifier: Apache-2.0
import garnet
T=garnet.tensor()
GARNET_MODEL_SPEC={'arguments':[{'name':'x','kind':'tensor'},{'name':'weights','kind':'weights'}]}
@T.fusion(atomic=True)
def layer(x,weights,index):
    return x*T.unary_op('linear',weight_name='layer.'+str(index)+'.weight',bias_name='layer.'+str(index)+'.bias')
@T.fusion(boundary='required')
def forward(x,weights):
    for index in range(4):x=layer(x,weights,index)
    return x

@T.fusion(boundary='required')
def projection(x,weights):
    return layer(x,weights,0)

@T.fusion(boundary='required')
def projection_fp32(x,weights):
    return x*T.unary_op('linear',weight_name='layer.0.weight',bias_name='layer.0.bias',compute_dtype='float32')

@T.fusion(boundary='required')
def projection_accumulation_fp32(x,weights):
    return x*T.unary_op('linear',weight_name='layer.0.weight',bias_name='layer.0.bias',accumulation_dtype='float32')
