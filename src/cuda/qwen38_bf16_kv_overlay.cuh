// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "qwen35_attention_core.cuh"
#include <cstdint>
namespace Garnet {
// Unwired study: Q/K have already undergone normalization and RoPE. Inputs are
// Q,K,V,gate,canonical FP32 carrier,slots,active. Both current-chunk K/V and past
// BF16 K/V enter attention at BF16 precision; query/gate/reductions remain FP32.
// Caller validates active-prefix tokens [0,count), integral slots=slot0+t and
// unique slot ownership. Inactive slots may be4096. Other masks are unsupported.
// This readonly operator never commits cache entries. One long-context strategy
// is used for all prefixes to avoid crossing a reduction-strategy boundary.
size_t Qwen38BF16KVOverlayWorkspace(const Qwen35AttentionCoreConfig&) noexcept;
int RunQwen38BF16KVOverlay(const Qwen35AttentionCoreConfig&,const void* const*,
    void* output,void* workspace,int64_t keyOffset,int64_t valueOffset,cudaStream_t) noexcept;
}
