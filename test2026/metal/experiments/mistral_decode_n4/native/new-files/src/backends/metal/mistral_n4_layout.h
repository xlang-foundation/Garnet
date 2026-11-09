// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
namespace Garnet {
inline constexpr uint64_t kMistralN4CacheBudget=uint64_t(3)*1024*1024*1024;
inline bool MistralN4Shape(size_t k,size_t n) noexcept {
    return (k==4096&&n==14336)||(k==14336&&n==4096);
}
inline bool MistralN4Profile(size_t k,size_t n,size_t rows) noexcept {
    return rows==4&&MistralN4Shape(k,n);
}
inline size_t MistralN4PayloadBytes(size_t k,size_t n) {
    if(!MistralN4Shape(k,n))throw std::invalid_argument("N4 cache requires a qualified Mistral MLP shape");
    return n*(k/8)*sizeof(uint32_t)+2*n*(k/64)*sizeof(uint16_t);
}
inline bool MistralN4MemoryFits(uint64_t cached,uint64_t bytes,uint64_t used,uint64_t recommended) noexcept {
    constexpr uint64_t reserve=256*1024*1024,alignment=64*1024;
    if(!bytes||cached>kMistralN4CacheBudget||bytes>kMistralN4CacheBudget-cached||bytes>UINT64_MAX-(alignment-1))return false;
    const auto charge=(bytes+alignment-1)&~(alignment-1);
    return recommended>reserve&&used<=recommended-reserve&&charge<=recommended-reserve-used;
}
template<class T> void MistralN4Pack(const T* input,T* output,size_t columns,size_t width) {
    if(!input||!output||input==output||!columns||columns%4||!width||columns>SIZE_MAX/width)
        throw std::invalid_argument("Invalid immutable N4 packing layout");
    for(size_t group=0;group<columns/4;++group)for(size_t word=0;word<width;++word)
        for(size_t c=0;c<4;++c)output[(group*width+word)*4+c]=input[(group*4+c)*width+word];
}
}
