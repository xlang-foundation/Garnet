// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "qwen35_tree_metadata.cuh"
#include "qwen35_attention_core.cuh"
namespace Garnet {
struct Qwen35DeltaConfig;
// Metadata is readonly and must be produced by RunQwen35TreeMetadata with the
// exact same profile. Native graph lowering must bind producer/consumer config;
// an arbitrary INT32 buffer is not an admitted descriptor.
size_t Qwen35TreeDeltaWorkspace(const Qwen35DeltaConfig&,Qwen35TreeMetadataConfig) noexcept;
// Same11 FP32 inputs as the serial delta kernel. Outputs node features/status,
// never a canonical state. Workspace holds convolved token inputs only.
int RunQwen35TreeDelta(const Qwen35DeltaConfig&,Qwen35TreeMetadataConfig,const void* const*,
    const Qwen35TreeMetadata*,float* output,float* status,void* workspace,cudaStream_t) noexcept;
size_t Qwen35TreeAttentionWorkspace(const Qwen35AttentionCoreConfig&) noexcept;
// Inputs Q,K,V,gate,oldK,oldV,active. Metadata is profile{B,T,capacity}.
// Canonical K/V stays readonly; status[B] gates every eventual commit.
int RunQwen35TreeAttention(const Qwen35AttentionCoreConfig&,const void* const*,const Qwen35TreeMetadata*,
    float* output,float* status,void* workspace,cudaStream_t) noexcept;
}
