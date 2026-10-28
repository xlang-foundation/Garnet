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
// Separate default-off research profile. Original normalization and witness
// profiles remain unchanged when this metadata is absent or zero.
inline void RequireRowwiseKDecodeNormPolicy(const char* policy,const std::string& precision,
    bool externalCompressed,bool externalBF16,bool floatInput,bool ordinaryDiagnostic,
    bool oldChunkWitness,bool decode35Witness,bool overlay,bool reference,bool fused,bool joined){
    if(!policy||std::strcmp(policy,"1")||precision!="modelopt_mixed"||!externalCompressed||externalBF16||
        !floatInput||ordinaryDiagnostic||oldChunkWitness||decode35Witness||overlay||reference||!fused||!joined)
        throw std::invalid_argument("Rowwise K decode norm requires explicit exclusive shared mixed F32 joined policy");
}

inline int RowwiseKDecodeNormLayer(const std::string& layer){
    for(int group=0;group<16;++group){
        const int index=group*4+3;
        if(layer=="model.language_model.layers."+std::to_string(index))return index;
    }
    throw std::invalid_argument("Rowwise K decode norm requires one of the sixteen pinned full-attention layers");
}

inline void RequireRowwiseKDecodeNormGeometry(const std::vector<int>& shape,const std::string& layer,
    int heads,int kv,int dim,int rotary,std::array<int,3> sections,double theta,double epsilon,
    bool legacyBF16,bool lowBatch,bool prefill128,int64_t stateWidth,int64_t keyOffset,int64_t valueOffset){
    const int index=RowwiseKDecodeNormLayer(layer);
    const int64_t expectedKey=int64_t((index-3)/4)*6676480+2482176;
    if(shape!=std::vector<int>{4,4,5120}||heads!=24||kv!=4||dim!=256||rotary!=64||
        sections!=std::array<int,3>{11,11,10}||!std::isfinite(theta)||theta!=1.e7||
        !std::isfinite(epsilon)||epsilon!=1.e-6||!legacyBF16||lowBatch||prefill128||
        stateWidth!=106823680||keyOffset!=expectedKey||valueOffset!=expectedKey+2097152)
        throw std::invalid_argument("Rowwise K decode norm requires exact B4 T4 canonical BF16 full-attention geometry");
}

inline void TraceRowwiseKDecodeNorm(const std::string& layer){
    // Counts graph construction subgraphs, not executed CUDA kernel launches.
    std::fprintf(stderr,"GARNET_ROWWISE_K_DECODE_NORM {\"batch\":4,\"tokens\":4,\"layer\":%d,\"rms_subgraphs\":4}\n",RowwiseKDecodeNormLayer(layer));
}
}
