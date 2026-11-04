// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
namespace Garnet {
// Conservative element-index bounds for the current affine tiles. Base device
// pointers remain native-width. This proves only index arithmetic, not device
// allocation/admission or weight validity; those checks remain independent.
inline bool AffineProfileIndex32Fits(size_t maximumRows,size_t inner,size_t outputs,
    size_t inputStride,size_t outputStride) {
    constexpr auto maximum=std::numeric_limits<uint32_t>::max();
    // Tile coordinates can extend beyond logical tails before masking. Leave
    // room for those additions and for the final loop increment.
    if(!maximumRows||maximumRows>512||inner<64||inner%64||inner>maximum-128||
       !outputs||outputs>maximum-128||inputStride<inner||outputStride<outputs)return false;
    // Division form avoids overflowing the host arithmetic being validated.
    return inputStride<=maximum/maximumRows&&outputStride<=maximum/maximumRows&&
        outputs<=maximum/(inner/8);
}
}
