// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "modelopt_fp8_batch4_prefill_profile.h"
#include <cstdint>
#include <cstring>
#include <stdexcept>
namespace Garnet {
inline bool ModelOptFP8M128GDNSelected(const char* policy,int batch,int tokens,int64_t inputRows,int outputRows,int columns,bool gdnOutput,const char* batch4Policy=nullptr){
    if(policy&&*policy&&std::strcmp(policy,"0")&&std::strcmp(policy,"1"))
        throw std::invalid_argument("GARNET_FP8_GDN_M128_CANONICAL must be0 or1");
    const bool batch4Prefill=ModelOptFP8Batch4PrefillSelected(batch4Policy,batch,tokens,inputRows);
    if(!policy||std::strcmp(policy,"1")||(!(batch==1&&tokens==128)&&!batch4Prefill)||inputRows!=128||!gdnOutput)return false;
    if(outputRows!=5120||columns!=6144)throw std::invalid_argument("Unqualified M128 GDN output projection");
    return true;
}
}
