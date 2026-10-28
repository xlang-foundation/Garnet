# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0
import garnet
from xModel.qwen35.vl_0_8b.tensor_compat import tensor
T = tensor()
@T.fusion(name='fp8_policy_cache_addition_fixture', boundary='required')
def Model(x):
    doubled = x + x
    scaled = doubled * x
    return scaled - x
