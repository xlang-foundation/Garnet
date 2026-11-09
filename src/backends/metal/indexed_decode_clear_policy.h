// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
namespace Garnet {
// Device, ownership, initialized-input, row uniqueness and byte-span validation
// remain prerequisites in EncodeAffineInt4IndexedDecode. This pure predicate
// is only the additional envelope for the separately prepared zero writer.
inline bool IndexedDecodeClearSupported(size_t physicalRows,size_t outputRows,
    size_t selectedRows,size_t columns,size_t outputStride) {
    return physicalRows==4&&outputRows==4&&selectedRows>0&&selectedRows<4&&
        columns>0&&outputStride==columns&&
        outputStride<=std::numeric_limits<uint32_t>::max()/4;
}
}
