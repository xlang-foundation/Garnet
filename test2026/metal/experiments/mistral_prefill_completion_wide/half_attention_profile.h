// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
namespace Garnet {
// The private Q arena is always512 token rows; wider owners do not enlarge it.
inline constexpr size_t kMetalHalfAttentionMaximumChunk=
#ifdef GARNET_METAL_CHUNK_HALF_WIDE
    512;
#else
    128;
#endif
}
