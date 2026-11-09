# SPDX-License-Identifier: Apache-2.0
import garnet
T=garnet.tensor()
GARNET_MODEL_SPEC={'arguments':[{'name':n,'kind':'tensor'} for n in ('a','b','c')]+[{'name':'weights','kind':'weights'}]}
@T.fusion(name='dynamic_first',boundary='required')
def first(x):return (x*T.unary_op('linear',weight_name='proj.weight'))*T.unary_op('reshape',shape=[1,-1,2,4])
@T.fusion(name='dynamic_second',boundary='required')
def second(x,b,c):
    value=x+b*T.unary_op('reshape',shape=[1,-1,2,4])
    return (value*T.binary_op('tensor_concat',axis=1))*(c*T.unary_op('reshape',shape=[1,1,2,4]))
@T.fusion(name='dynamic_partition_fixture',boundary='required')
def forward(a,b,c,weights):return second(first(a+b),b,c)*T.unary_op('reshape',shape=[1,-1,8])*2.0
