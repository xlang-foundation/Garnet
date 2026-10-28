// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "attention_kv_delta_core.cuh"
namespace Garnet {
// Standalone B4/T4 research component; old cache and controls are readonly.
// strategy0: cached compact;1: cached legacy geometry;2: per-query hybrid,
// using compact at visible prefix <=512 and legacy geometry above512.
// strategy1/2 require existing core workspace; explicit commit remains separate.
int RunQwen35AttentionKVOverlayCached(const Qwen35AttentionCoreConfig&,const void* const*,void*,void*,int,cudaStream_t) noexcept;
}
