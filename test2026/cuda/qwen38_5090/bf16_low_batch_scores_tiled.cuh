// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "qwen38_bf16_kv_low_batch_overlay.cuh"
namespace Garnet {
// Unwired B1 research candidate. Full GPU parity/FP64 replay and timing required.
size_t Qwen38BF16KVTiledScoresWorkspace(const Qwen35AttentionCoreConfig&) noexcept;
int RunQwen38BF16KVTiledScores(const Qwen35AttentionCoreConfig&,const void* const*,
    void* output,void* workspace,int64_t keyOffset,int64_t valueOffset,cudaStream_t) noexcept;
}
