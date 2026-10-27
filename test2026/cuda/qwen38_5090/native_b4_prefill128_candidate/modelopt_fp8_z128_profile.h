// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "modelopt_fp8_batch4_prefill_profile.h"
#include <cstdint>
#include <cstring>
#include <stdexcept>
namespace Garnet {
inline bool ModelOptFP8Z128GDNZSelected(const char* policy,int batch,int tokens,int64_t inputRows,int outputRows,int columns,bool gdnZ,const char* batch4Policy=nullptr){
    if(policy&&*policy&&std::strcmp(policy,"0")&&std::strcmp(policy,"1"))
        throw std::invalid_argument("GARNET_FP8_GDN_Z128_CANONICAL must be0 or1");
    const bool batch4Prefill=ModelOptFP8Batch4PrefillSelected(batch4Policy,batch,tokens,inputRows);
    if(!policy||std::strcmp(policy,"1")||(!(batch==1&&tokens==128)&&!batch4Prefill)||inputRows!=128||!gdnZ)return false;
    if(outputRows!=6144||columns!=5120)throw std::invalid_argument("Unqualified M128 GDN Z projection");
    return true;
}
}
