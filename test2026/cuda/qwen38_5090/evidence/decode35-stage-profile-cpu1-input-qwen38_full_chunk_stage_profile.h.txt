// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
namespace Garnet {
inline void RequireFullChunkStageGeometry(const std::vector<int>& shape,const std::string& layer,
    int heads,int kv,int dim,int rotary,std::array<int,3> sections,double theta,double epsilon,
    bool lowBatch,bool prefill128,int64_t stateWidth,int64_t keyOffset,int64_t valueOffset){
    if(shape.size()!=3||shape[0]!=1||(shape[1]!=32&&shape[1]!=128)||shape[2]!=5120||
        layer!="model.language_model.layers.3"||heads!=24||kv!=4||dim!=256||rotary!=64||
        sections!=std::array<int,3>{11,11,10}||theta!=1.e7||!std::isfinite(theta)||
        epsilon!=1.e-6||!std::isfinite(epsilon)||lowBatch!=(shape[1]==32)||prefill128!=(shape[1]==128)||
        stateWidth!=106823680||keyOffset!=2482176||valueOffset!=4579328)
        throw std::invalid_argument("Full chunk stage witness requires exact layer3 B1 T32/T128 canonical BF16 geometry");
}
inline void RequireFullChunkStageWitness(const char* policy,const std::string& precision,bool externalCompressed,
    bool externalBF16,bool floatInput,bool ordinaryDiagnostic,bool legacyBF16,bool overlay,bool reference,
    bool fused,bool joined){
    if(!policy||std::strcmp(policy,"1")||precision!="modelopt_mixed"||!externalCompressed||externalBF16||
        !floatInput||ordinaryDiagnostic||legacyBF16||overlay||reference||!fused||!joined)
        throw std::invalid_argument("Full chunk stage witness requires explicit exclusive shared mixed joined F32 study policy");
}
}
