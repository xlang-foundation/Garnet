// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <array>
#include <cstdint>
#include <limits>
#include <initializer_list>
namespace Garnet {
// batch,chunk,heads,kvHeads,dim,capacity,q/k/v/kc/vc/s/a/g/o strides,gated.
// Pointer addresses stay native-width; this proves the element-index arithmetic
// in the bounded chunk kernel. Allocation, dtype and tensor layouts are still
// validated separately by the encoder.
inline bool ChunkAttentionIndex32Fits(const std::array<uint64_t,16>& p) {
    constexpr uint64_t maximum=std::numeric_limits<uint32_t>::max();
    for(auto field:p)if(field>maximum)return false;
    const auto b=p[0],t=p[1],h=p[2],kv=p[3],d=p[4],cap=p[5];
    if(!b||b>64||!t||t>256||!h||!kv||h%kv||!d||d>maximum-128||!cap||cap>4096||p[15]>1)return false;
    const auto rows=b*t;
    if(h>maximum/d||kv>maximum/d||h>maximum/rows)return false;
    const auto qWidth=h*d,kvWidth=kv*d;
    if(kvWidth>maximum/cap)return false;
    if(p[6]<qWidth||p[7]<kvWidth||p[8]<kvWidth||p[9]<kvWidth*cap||p[10]<kvWidth*cap||
       p[11]<t||p[12]<t||p[13]<qWidth||p[14]<qWidth)return false;
    for(auto field:{6,7,8,13,14})if(p[field]>maximum/rows)return false;
    for(auto field:{9,10,11,12})if(p[field]>maximum/b)return false;
    return true;
}
}
