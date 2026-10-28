// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace Garnet {
inline constexpr std::array<int,7> Decode35StageWidths{{20480,20480,49152,4096,4096,4096,4096}};
inline constexpr int Decode35StageTotalWidth=106496;

inline void RequireDecode35StagePolicy(const char* policy,const std::string& precision,
    bool externalCompressed,bool externalBF16,bool floatInput,bool ordinaryDiagnostic,
    bool oldChunkWitness,bool overlay,bool reference,bool fused,bool joined){
    if(!policy||std::strcmp(policy,"1")||precision!="modelopt_mixed"||!externalCompressed||externalBF16||
        !floatInput||ordinaryDiagnostic||oldChunkWitness||overlay||reference||!fused||!joined)
        throw std::invalid_argument("Decode35 stage witness requires explicit exclusive shared mixed F32 joined policy");
}

inline void RequireDecode35StageGeometry(const std::vector<int>& shape,const std::string& layer,
    int heads,int kv,int dim,int rotary,std::array<int,3> sections,double theta,double epsilon,
    bool legacyBF16,bool lowBatch,bool prefill128,int64_t stateWidth,int64_t keyOffset,int64_t valueOffset){
    if(shape.size()!=3||(shape[0]!=1&&shape[0]!=4)||shape[1]!=4||shape[2]!=5120||
        layer!="model.language_model.layers.35"||heads!=24||kv!=4||dim!=256||rotary!=64||
        sections!=std::array<int,3>{11,11,10}||theta!=1.e7||!std::isfinite(theta)||
        epsilon!=1.e-6||!std::isfinite(epsilon)||legacyBF16!=(shape[0]==4)||lowBatch!=(shape[0]==1)||prefill128||
        stateWidth!=106823680||keyOffset!=55894016||valueOffset!=57991168)
        throw std::invalid_argument("Decode35 stage witness requires exact layer35 B1/B4 T4 canonical BF16 geometry");
}

inline void TraceDecode35StageWitness(int batch){
    // Called only after policy/geometry and all seven actual tensor shapes pass.
    std::fprintf(stderr,"GARNET_DECODE35_STAGE_WITNESS {\"batch\":%d,\"tokens\":4,\"layer\":35,\"stages\":7,\"row_words\":106496}\n",batch);
}
}
