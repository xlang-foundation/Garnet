// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <limits>
#include <initializer_list>
namespace Garnet {
inline bool BulkSiluHalfSupported(uint64_t rows,uint64_t columns,uint64_t aStride,uint64_t bStride,uint64_t halfStride){
 if(!rows||rows>8192||(columns!=4096&&columns!=14336))return false;
 for(auto stride:{aStride,bStride,halfStride})if(stride<columns||stride>std::numeric_limits<uint32_t>::max()/rows)return false;
 return true;
}
}
