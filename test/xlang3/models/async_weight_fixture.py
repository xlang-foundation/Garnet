# SPDX-License-Identifier: Apache-2.0
import garnet
T=garnet.tensor()
GARNET_MODEL_SPEC={'arguments':[{'name':'x','kind':'tensor'},{'name':'weights','kind':'weights'}]}
@T.fusion(name='async_weight_fixture')
def forward(x,weights):return x*T.unary_op('linear',weight_name='proj.weight')
