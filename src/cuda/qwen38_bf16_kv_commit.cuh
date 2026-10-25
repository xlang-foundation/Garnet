// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>
namespace Garnet {
// Separate, currently unwired study operator. The old FP32/B4 helper is unchanged.
constexpr int64_t Qwen38BF16KVCanonicalWidth=106823680;
constexpr int64_t Qwen38BF16KVReplayWidth=39714816;
struct Qwen38BF16KVSpans { int start[8],count[8]; };
// FP32 carrier matrices; only full-attention cache intervals contain BF16 bits.
// T4 may use accepted-prefix replay state. T32 is final-prefill state only.
// On a launch failure after copying begins, completion is unknown: discard the
// owning request state. This function does not promise transactional rollback.
int RunQwen38BF16KVCommit(const float* target,const float* linear,int64_t linearWidth,
    float* canonical,int batch,int chunk,Qwen38BF16KVSpans,cudaStream_t) noexcept;
}
