// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "qwen35_attention_core.cuh"
#include <cstdint>
namespace Garnet {
// Separate unintegrated numerical profile; the existing B4/B8 admission stays
// unchanged. This does not admit a complete decoder, plugin or serving worker.
inline bool BF16KVLowBatchCandidateProfile(const Qwen35AttentionCoreConfig& c) {
    return c.batch==1 && (c.tokens==4 || c.tokens==32) && c.heads==24 && c.kv==4 && c.capacity==4096 && c.dim==256;
}
size_t Qwen38BF16KVLowBatchCandidateWorkspace(const Qwen35AttentionCoreConfig&) noexcept;
int RunQwen38BF16KVLowBatchCandidate(const Qwen35AttentionCoreConfig&,const void* const*,
    void* output,void* workspace,int64_t keyOffset,int64_t valueOffset,cudaStream_t) noexcept;
}
