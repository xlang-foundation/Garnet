// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "qwen35_attention_core.cuh"
#include <cstdint>
namespace Garnet {
// A distinct unintegrated component. Never widen the old T4/T32 admission.
inline bool BF16KVPrefill128AttentionProfile(const Qwen35AttentionCoreConfig& c) {
    return c.batch==1 && c.tokens==128 && c.heads==24 && c.kv==4 && c.capacity==4096 && c.dim==256;
}
size_t Qwen38BF16KVPrefill128AttentionWorkspace(const Qwen35AttentionCoreConfig&) noexcept;
// Q/K/V/gate/canonical/slots/active; canonical packed BF16 KV is readonly.
// Caller validates active-prefix/contiguous integer slots/range. Strategy0 uses
// original score math, strategy1 tiles16positions with the same FMA/reduce order.
int RunQwen38BF16KVPrefill128Attention(const Qwen35AttentionCoreConfig&,const void* const*,
    void* output,void* workspace,int64_t keyOffset,int64_t valueOffset,int strategy,cudaStream_t) noexcept;
}
