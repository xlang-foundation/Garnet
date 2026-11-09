// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
namespace Garnet {
// Keep the64-row control for small counts and odd64-row tile counts. The
//128-row launch must not add a complete extra padded64-row tile of work.
inline bool UseAffineReuse128(size_t rows,size_t columns,size_t minimumOutputs) {
    return rows>=128&&rows<=512&&columns>=minimumOutputs&&((rows+63)/64)%2==0;
}
}
