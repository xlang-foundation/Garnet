// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "qwen38_bf16_kv_overlay.cuh"
namespace Garnet {
// Unwired B4T32 component study. Same FP32 arithmetic/BF16 KV contract as
// the legacy operator; sixteen score positions per warp. Actual native byte
// comparison and timing required. No serving factory or existing ABI changes.
size_t Qwen38BF16KVPrefill32TiledScoresWorkspace(const Qwen35AttentionCoreConfig&) noexcept;
int RunQwen38BF16KVPrefill32TiledScores(const Qwen35AttentionCoreConfig&,const void* const*,
    void* output,void* workspace,int64_t keyOffset,int64_t valueOffset,cudaStream_t) noexcept;
}
