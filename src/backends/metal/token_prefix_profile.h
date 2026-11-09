// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <algorithm>
#include <cstdint>
#ifndef GARNET_METAL_TOKEN_PREFIX_HOST_MIB
#define GARNET_METAL_TOKEN_PREFIX_HOST_MIB 512
#endif
namespace Garnet {
static_assert(GARNET_METAL_TOKEN_PREFIX_HOST_MIB==512||GARNET_METAL_TOKEN_PREFIX_HOST_MIB==1024||GARNET_METAL_TOKEN_PREFIX_HOST_MIB==3072,"Invalid compiled prefix host budget");
// Full prompt checkpoints are captured separately. Avoid a near-identical
// power-of-two snapshot whose replay saving is under64 tokens.
inline bool TokenPrefixIntermediateUseful(uint64_t tokens,uint64_t total) {
    return tokens>=64&&!(tokens&(tokens-1))&&tokens<=total&&(tokens==total||total-tokens>=64);
}
inline constexpr uint64_t kTokenPrefixHostLimitBytes=uint64_t(GARNET_METAL_TOKEN_PREFIX_HOST_MIB)*1024*1024;
// The larger profile reserves the whole cache plus a prospective replacement
// before configuration. Capture checks its actual additional payload again.
// Device allocated and task footprint overlap, so use their maximum. These are
// task-local snapshot ceilings, not a combined physical-memory upper bound
// or a measurement of system free memory.
inline bool TokenPrefixHostMemoryFits(uint64_t allocated,uint64_t physical,uint64_t recommended,uint64_t bytes,unsigned copies) {
    constexpr uint64_t reserve=256ull*1024*1024;
    if(!bytes||bytes>kTokenPrefixHostLimitBytes||copies<1||copies>2||recommended<=reserve)return false;
    const uint64_t used=std::max(allocated,physical),available=recommended-reserve;
    if(used>available)return false;
    return bytes<=(available-used)/copies;
}
}
