# SPDX-License-Identifier: Apache-2.0
import garnet
T=garnet.tensor()
@T.fusion(name='first_device',boundary='required')
def first(x):return x*x
@T.fusion(name='second_device',boundary='required')
def second(x,b):return x+b
@T.fusion(name='device_partition_fixture',boundary='required')
def forward(a,b):return second(first(a+b),b)*b
