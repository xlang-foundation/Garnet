// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
namespace Garnet {
inline bool ModelOptFP8Q128TextWeight(const std::string& name){
    for(int layer=3;layer<64;layer+=4){
        const auto prefix="model.language_model.layers."+std::to_string(layer)+".self_attn.";
        if(name==prefix+"q_proj.weight")return true;
    }
    return false;
}
inline bool ModelOptFP8Q128Selected(const char* policy,int batch,int tokens,int64_t inputRows,
    int outputRows,int columns,const std::string& weight){
    if(policy&&*policy&&std::strcmp(policy,"0")&&std::strcmp(policy,"1"))
        throw std::invalid_argument("GARNET_FP8_FULL_ATTN_Q128_CANONICAL must be0 or1");
    if(!policy||std::strcmp(policy,"1")||batch!=1||tokens!=128||inputRows!=128||!ModelOptFP8Q128TextWeight(weight))return false;
    if(outputRows!=12288||columns!=5120)throw std::invalid_argument("Unqualified M128 text Q projection");
    return true;
}
}
