// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>
namespace Garnet {
// Pinned 27B B4/T4 canonical and compact-output geometry. FP32 row widths.
constexpr int64_t Qwen35KVCanonicalWidth=173932544;
constexpr int64_t Qwen35KVTargetWidth=43030528;
constexpr int64_t Qwen35KVReplayWidth=39714816;
constexpr int64_t Qwen35KVLinearWidth=827392;
struct Qwen35KVDeltaLinearLayout { int64_t canonical,target,replay,width; };
struct Qwen35KVDeltaFullLayout { int64_t canonical,target; };
struct Qwen35KVDeltaSpans { int start[4],count[4]; };
// Caller supplies validated complete layout metadata and contiguous active prefixes.
// Reads target chunk K/V plus either target or prefix-replayed linear state.
int RunQwen35KVDeltaCommit(const float* target,const float* linear,int64_t linearWidth,
    float* canonical,const Qwen35KVDeltaLinearLayout*,const Qwen35KVDeltaFullLayout*,
    Qwen35KVDeltaSpans,cudaStream_t) noexcept;
}
