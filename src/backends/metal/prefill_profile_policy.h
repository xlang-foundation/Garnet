// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>
namespace Garnet {
struct DenseOwnerPrefillFamily {size_t rows,chunk,fallbackChunk;};
// Pure planning over prepared profiles. Physical row identities never move.
// The budget counts actual tokens; each selected GPU family separately fits
// the compiled512-token workspace. Short tails choose an up-to shape, rather
// than padding every request to its largest configured chunk.
inline size_t SelectDenseOwnerPrefillChunk(const std::vector<size_t>& rows,
    const std::vector<size_t>& remaining,size_t maximum,size_t budget,
    const std::vector<DenseOwnerPrefillFamily>& families) {
    if(rows.size()!=remaining.size()||rows.size()>4||maximum<2||maximum>512||(maximum&(maximum-1))||budget>512)
        throw std::invalid_argument("Invalid prepared owner-prefill policy bounds");
    for(const auto& family:families)if((family.rows!=1&&family.rows!=2)||family.chunk<2||family.chunk>512||
        (family.chunk&(family.chunk-1))||family.rows>512/family.chunk||family.fallbackChunk!=std::min(family.chunk,size_t(128)))
        throw std::invalid_argument("Invalid prepared owner-prefill family");
    std::vector<size_t> live;size_t longest=0;
    for(size_t i=0;i<rows.size();++i) {
        if(rows[i]>=4||std::find(rows.begin(),rows.begin()+i,rows[i])!=rows.begin()+i)
            throw std::invalid_argument("Duplicate/out-of-range physical prefill owner");
        if(remaining[i]){live.push_back(rows[i]);longest=std::max(longest,remaining[i]);}
    }
    std::sort(live.begin(),live.end());
    const bool compact=live.size()==1||(live.size()==2&&live[1]==live[0]+1);
    size_t wanted=2;
    if(longest<=128)while(wanted<longest&&wanted<maximum)wanted*=2;
    else while(wanted*2<=std::min(longest,maximum))wanted*=2;
    for(size_t chunk=std::min(wanted,maximum);chunk>=2;chunk/=2) {
        bool prepared=false;
        for(const auto& family:families) {
            if(chunk<=128&&family.fallbackChunk==chunk)prepared=true;
            if(compact&&family.rows==live.size()&&family.chunk==chunk&&family.rows*chunk<=512)prepared=true;
        }
        if(!prepared)continue;
        size_t tokens=0;for(size_t count:remaining)tokens+=std::min(count,chunk);
        if(tokens<=budget)return chunk;
    }
    return 0; // Defer without releasing any stable row lease.
}
}
