// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
namespace Garnet {
inline void RequireQwen38FA3ProjectionWitness(const char* policy,const std::string& precision,
    const std::vector<int>& shape,const std::string& layer,double epsilon,bool external,bool floatInput){
    if(!policy||std::strcmp(policy,"1"))
        throw std::invalid_argument("FA3 projection witness requires explicit GARNET_FP8_FA3_PROJECTION_WITNESS=1");
    if(precision!="modelopt_mixed"||shape.size()!=3||shape[0]!=1||(shape[1]!=32&&shape[1]!=128)||shape[2]!=5120||
        layer!="model.language_model.layers.3"||!std::isfinite(epsilon)||epsilon!=1.e-6||external||!floatInput)
        throw std::invalid_argument("FA3 projection witness requires embedded mixed27B layer3 F32 B1 T32/T128");
}
}
