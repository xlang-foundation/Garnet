// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>
namespace Garnet {
inline bool ModelOptFP8M128GDNSelected(const char* policy,int batch,int tokens,int64_t inputRows,int outputRows,int columns,bool gdnOutput){
    if(policy&&*policy&&std::strcmp(policy,"0")&&std::strcmp(policy,"1"))
        throw std::invalid_argument("GARNET_FP8_GDN_M128_CANONICAL must be0 or1");
    if(!policy||std::strcmp(policy,"1")||batch!=1||tokens!=128||inputRows!=128||!gdnOutput)return false;
    if(outputRows!=5120||columns!=6144)throw std::invalid_argument("Unqualified M128 GDN output projection");
    return true;
}
}
