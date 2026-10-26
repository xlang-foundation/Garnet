// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>
namespace Garnet {
inline bool ModelOptFP8Z128GDNZSelected(const char* policy,int batch,int tokens,int64_t inputRows,int outputRows,int columns,bool gdnZ){
    if(policy&&*policy&&std::strcmp(policy,"0")&&std::strcmp(policy,"1"))
        throw std::invalid_argument("GARNET_FP8_GDN_Z128_CANONICAL must be0 or1");
    if(!policy||std::strcmp(policy,"1")||batch!=1||tokens!=128||inputRows!=128||!gdnZ)return false;
    if(outputRows!=6144||columns!=5120)throw std::invalid_argument("Unqualified M128 GDN Z projection");
    return true;
}
}
