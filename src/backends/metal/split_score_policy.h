// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <initializer_list>
namespace Garnet {
inline bool SplitScoreProfileSupported(size_t batch,size_t heads,size_t kvHeads,size_t dim,size_t capacity,size_t visible) {
    return (batch==1||batch==2||batch==4)&&heads==32&&kvHeads==8&&dim==128&&capacity>=512&&capacity<=4096&&visible>=512&&visible<=capacity;
}
// Element indices are narrowed only after proving each complete row extent
// fits uint32. Buffer offsets remain native Metal byte offsets.
inline bool Profile32ScoreSupported(const uint64_t* p,size_t visible) {
    if(!p||!SplitScoreProfileSupported(p[0],p[1],p[2],p[3],p[4],visible)||
       p[4]!=4096||p[5]!=4096||p[6]!=4194304||p[9]!=131072||!p[7]||!p[8])return false;
    constexpr uint64_t maximum=std::numeric_limits<uint32_t>::max();
    for(size_t field:{size_t(5),size_t(6),size_t(7),size_t(8),size_t(9)})if(p[field]>maximum/p[0])return false;
    return true;
}
}
