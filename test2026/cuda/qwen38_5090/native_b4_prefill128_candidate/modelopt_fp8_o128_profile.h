// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "modelopt_fp8_batch4_prefill_profile.h"
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
namespace Garnet {
inline bool ModelOptFP8O128TextWeight(const std::string& name){
    for(int layer=3;layer<64;layer+=4){
        const auto prefix="model.language_model.layers."+std::to_string(layer)+".self_attn.";
        if(name==prefix+"o_proj.weight")return true;
    }
    return false;
}
inline bool ModelOptFP8O128Selected(const char* policy,int batch,int tokens,int64_t inputRows,
    int outputRows,int columns,const std::string& weight,const char* batch4Policy=nullptr){
    if(policy&&*policy&&std::strcmp(policy,"0")&&std::strcmp(policy,"1"))
        throw std::invalid_argument("GARNET_FP8_FULL_ATTN_O128_CANONICAL must be0 or1");
    const bool batch4Prefill=ModelOptFP8Batch4PrefillSelected(batch4Policy,batch,tokens,inputRows);
    if(!policy||std::strcmp(policy,"1")||(!(batch==1&&tokens==128)&&!batch4Prefill)||inputRows!=128||!ModelOptFP8O128TextWeight(weight))return false;
    if(outputRows!=5120||columns!=6144)throw std::invalid_argument("Unqualified M128 text O projection");
    return true;
}
}
