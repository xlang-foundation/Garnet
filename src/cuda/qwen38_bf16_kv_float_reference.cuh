// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "qwen35_attention_core.cuh"
namespace Garnet {
// Unwired correctness reference, never a serving optimization. B4,T4/T32 only.
// Uses the existing dense FP32 full-cache attention core. Current K/V are
// explicitly rounded BF16->FP32 before the core writes its separate full caches.
// The caller must initialize past caches to zero or exact BF16-roundtrip FP32
// values and preserve that invariant using these rounded outputs thereafter.
// Inputs/outputs follow RunQwen35AttentionCore; caches are ordinary contiguous
// [B,4,4096,256] FP32 arrays, with no packed carrier offsets or compact commit.
// Controls must be valid finite active prefixes and unique integral cache slots.
// All output/workspace spans are distinct from inputs and one another. A failed
// enqueue leaves outputs unqualified; callers discard them. Inputs stay readonly.
size_t Qwen38BF16KVFloatReferenceWorkspace(const Qwen35AttentionCoreConfig&) noexcept;
int RunQwen38BF16KVFloatReference(const Qwen35AttentionCoreConfig&,
    const void* const* inputs,void* const* outputs,void* workspace,cudaStream_t) noexcept;
}
