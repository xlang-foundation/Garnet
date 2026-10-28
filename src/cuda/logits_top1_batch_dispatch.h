// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

namespace Garnet {
// Configure before the first sampler call or memory_json query. Only "1"
// enables the experimental route; the process then keeps that choice fixed.
inline bool ParseParallelBatchTop1(const char* value) {
    return value && value[0] == '1' && value[1] == '\0';
}
inline bool SelectParallelBatchTop1(bool enabled, int rows, int vocab) {
    return enabled && rows >= 1 && rows <= 128 && vocab >= 8192;
}
struct BatchTop1DispatchSnapshot {
    bool enabled;
    std::uint64_t legacyHostCalls;
    std::uint64_t parallelHostCalls;
};
// Host dispatch attempts, including capture construction. These are neither
// completed GPU operations nor graph replay/kernel counts.
BatchTop1DispatchSnapshot GetBatchTop1DispatchSnapshot();
}
