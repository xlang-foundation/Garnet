// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>
namespace Garnet {
// Explicit research geometry only. This selects the existing flat M128 GEMM;
// it does not flatten recurrent state, attention caches, or logical batches.
inline bool ModelOptFP8Batch4PrefillSelected(const char* policy,int batch,int tokens,int64_t inputRows){
    if(policy&&*policy&&std::strcmp(policy,"0")&&std::strcmp(policy,"1"))
        throw std::invalid_argument("GARNET_FP8_BATCH4_PREFILL_CANONICAL must be0 or1");
    if(!policy||std::strcmp(policy,"1")||batch!=4||tokens!=32)return false;
    if(inputRows!=128)throw std::invalid_argument("B4 T32 canonical FP8 requires exactly128 input rows");
    return true;
}
}
