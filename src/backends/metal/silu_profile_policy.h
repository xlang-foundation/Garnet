// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <initializer_list>
#include <limits>
namespace Garnet {
inline bool SiluProfile32Supported(uint64_t rows,uint64_t columns,uint64_t aStride,uint64_t bStride,uint64_t cStride) {
 if(!rows||rows>512||(columns!=4096&&columns!=14336))return false;
 const uint64_t maximum=std::numeric_limits<uint32_t>::max();
 for(auto stride:{aStride,bStride,cStride})if(stride<columns||stride>maximum/rows)return false;
 return true;
}
}
