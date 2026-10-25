// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "qwen_unicode_normalization_data.h"
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace Garnet::Tokenization::Unicode {
inline bool IsScalar(std::uint32_t cp) {
    return cp<=0x10FFFF && !(cp>=0xD800 && cp<=0xDFFF);
}
inline std::vector<std::uint32_t> DecodeUtf8(const std::string& text) {
    std::vector<std::uint32_t> result;
    for (std::size_t pos=0; pos<text.size();) {
        const unsigned char first=static_cast<unsigned char>(text[pos++]);
        std::uint32_t cp=first; unsigned extra=0;
        if (first>=0xC2 && first<=0xDF) { cp=first&0x1F; extra=1; }
        else if (first>=0xE0 && first<=0xEF) { cp=first&0x0F; extra=2; }
        else if (first>=0xF0 && first<=0xF4) { cp=first&0x07; extra=3; }
        else if (first>=0x80) throw std::invalid_argument("invalid tokenizer UTF-8 lead byte");
        for (unsigned k=0; k<extra; ++k) {
            if (pos==text.size()) throw std::invalid_argument("truncated tokenizer UTF-8");
            const unsigned char next=static_cast<unsigned char>(text[pos++]);
            if ((next&0xC0)!=0x80) throw std::invalid_argument("invalid tokenizer UTF-8 continuation");
            cp=(cp<<6)|(next&0x3F);
        }
        if ((extra==1 && cp<0x80) || (extra==2 && cp<0x800) ||
            (extra==3 && cp<0x10000) || !IsScalar(cp))
            throw std::invalid_argument("invalid tokenizer Unicode scalar");
        result.push_back(cp);
    }
    return result;
}
inline void AppendUtf8(std::string& out, std::uint32_t cp) {
    if (cp<=0x7F) out.push_back(static_cast<char>(cp));
    else if (cp<=0x7FF) {
        out.push_back(static_cast<char>(0xC0|(cp>>6)));
        out.push_back(static_cast<char>(0x80|(cp&0x3F)));
    } else if (cp<=0xFFFF) {
        out.push_back(static_cast<char>(0xE0|(cp>>12)));
        out.push_back(static_cast<char>(0x80|((cp>>6)&0x3F)));
        out.push_back(static_cast<char>(0x80|(cp&0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0|(cp>>18)));
        out.push_back(static_cast<char>(0x80|((cp>>12)&0x3F)));
        out.push_back(static_cast<char>(0x80|((cp>>6)&0x3F)));
        out.push_back(static_cast<char>(0x80|(cp&0x3F)));
    }
}
inline std::uint32_t CombiningOrder(std::uint32_t cp) {
    std::size_t lo=0, hi=sizeof(kCombiningRanks)/sizeof(kCombiningRanks[0]);
    while (lo<hi) {
        const std::size_t mid=lo+(hi-lo)/2;
        if (cp<kCombiningRanks[mid].cp) hi=mid;
        else if (cp>kCombiningRanks[mid].cp) lo=mid+1;
        else return kCombiningRanks[mid].rank;
    }
    return 0;
}
inline std::uint32_t Compose(std::uint32_t first, std::uint32_t second) {
    if (first>=0x1100 && first<0x1113 && second>=0x1161 && second<0x1176)
        return 0xAC00+(first-0x1100)*588+(second-0x1161)*28;
    if (first>=0xAC00 && first<=0xD7A3 && (first-0xAC00)%28==0 && second>0x11A7 && second<0x11C3)
        return first+second-0x11A7;
    std::size_t lo=0, hi=sizeof(kCompositions)/sizeof(kCompositions[0]);
    while (lo<hi) {
        const std::size_t mid=lo+(hi-lo)/2; const auto& pair=kCompositions[mid];
        if (first<pair.first || (first==pair.first && second<pair.second)) hi=mid;
        else if (first>pair.first || (first==pair.first && second>pair.second)) lo=mid+1;
        else return pair.result;
    }
    return 0;
}
inline std::vector<std::uint32_t> NormalizeNfc(const std::vector<std::uint32_t>& input) {
    bool ascii=true;
    for (auto cp:input) {
        if (!IsScalar(cp)) throw std::invalid_argument("invalid normalization Unicode scalar");
        if (cp>=0x80) ascii=false;
    }
    if (ascii) return input;
    std::vector<std::uint32_t> ordered;
    auto append=[&](std::uint32_t cp) {
        const auto rank=CombiningOrder(cp); ordered.push_back(cp);
        if (rank==0) return;
        std::size_t end=ordered.size()-1;
        while (end>0 && CombiningOrder(ordered[end-1])>rank) {
            ordered[end]=ordered[end-1]; --end;
        }
        ordered[end]=cp;
    };
    for (auto cp:input) {
        if (cp>=0xAC00 && cp<=0xD7A3) {
            const auto index=cp-0xAC00;
            append(0x1100+index/588); append(0x1161+(index%588)/28);
            if (index%28) append(0x11A7+index%28);
            continue;
        }
        std::size_t lo=0, hi=sizeof(kDecompositions)/sizeof(kDecompositions[0]);
        while (lo<hi) {
            const auto mid=lo+(hi-lo)/2;
            if (cp>kDecompositions[mid].cp) lo=mid+1; else hi=mid;
        }
        if (lo<sizeof(kDecompositions)/sizeof(kDecompositions[0]) && kDecompositions[lo].cp==cp) {
            const auto& item=kDecompositions[lo];
            for (std::uint32_t j=0; j<item.length; ++j) append(kNfdValues[item.offset+j]);
        } else append(cp);
    }
    std::vector<std::uint32_t> result;
    std::size_t starter=0; std::uint32_t previousRank=0;
    for (auto cp:ordered) {
        const auto rank=CombiningOrder(cp);
        const auto combined=result.empty()?0:Compose(result[starter],cp);
        if (combined && (previousRank==0 || previousRank<rank)) result[starter]=combined;
        else {
            if (rank==0) starter=result.size();
            result.push_back(cp); previousRank=rank;
        }
    }
    return result;
}
} // namespace Garnet::Tokenization::Unicode
