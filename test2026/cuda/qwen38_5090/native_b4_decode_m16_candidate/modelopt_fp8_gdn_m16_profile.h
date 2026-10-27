// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
namespace Garnet {
inline int ModelOptFP8GDNM16TextProjection(const std::string& name){
    for(int layer=0;layer<64;++layer){
        if(layer%4==3)continue;
        const auto prefix="model.language_model.layers."+std::to_string(layer)+".linear_attn.";
        if(name==prefix+"in_proj_qkv.weight")return 10240;
        if(name==prefix+"in_proj_z.weight")return 6144;
    }
    return 0;
}
inline bool ModelOptFP8GDNM16Selected(const char* policy,int batch,int tokens,int64_t inputRows,
    int outputRows,int columns,const std::string& weight){
    if(policy&&*policy&&std::strcmp(policy,"0")&&std::strcmp(policy,"1"))
        throw std::invalid_argument("GARNET_FP8_GDN_BATCH4_DECODE_CANONICAL must be0 or1");
    if(!policy||std::strcmp(policy,"1")||batch!=4||tokens!=4||inputRows!=16)return false;
    const int expected=ModelOptFP8GDNM16TextProjection(weight);if(!expected)return false;
    if(outputRows!=expected||columns!=5120)throw std::invalid_argument("Unqualified physicalB4T4 GDN FP8 QKV/Z projection");
    return true;
}
}
