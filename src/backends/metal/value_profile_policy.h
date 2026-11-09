// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
namespace Garnet {
inline bool ValueProfile32Supported(const uint64_t* p) {
    if(!p||(p[0]!=1&&p[0]!=2&&p[0]!=4)||p[1]!=32||p[2]!=8||p[3]!=128||p[4]!=4096||
       p[5]!=131072||p[6]!=4194304||p[7]!=4096||!p[8]||p[9]!=4096||p[10]!=0)return false;
    constexpr uint64_t maximum=std::numeric_limits<uint32_t>::max();
    for(size_t field=5;field<=9;++field)if(p[field]>maximum/p[0])return false;
    return true;
}
}
