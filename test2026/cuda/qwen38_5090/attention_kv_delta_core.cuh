// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "qwen35_attention_core.cuh"
namespace Garnet {
// Standalone experiment only: exactly B4/T4/H24/KV4/C4096/D256 FP32.
// Overlay reads old caches plus current chunk K/V without changing canonical KV.
// Inputs/controls follow the existing validated core contract; output cannot alias.
int RunQwen35AttentionKVOverlay(const Qwen35AttentionCoreConfig&,const void* const*,void*,cudaStream_t) noexcept;
// Separate transaction-final write: counts are validated integer accepted active
// prefix lengths per row, 0..number of active tokens. Only accepted slots change.
// Caller guarantees unique valid active slots and stream order after all readers.
// This explicit mutable operation does not alter the existing plugin no-alias API.
int CommitQwen35AttentionKVPrefix(const Qwen35AttentionCoreConfig&,const void* const*,const float*,void*,void*,cudaStream_t) noexcept;
}
