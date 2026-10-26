// SPDX-License-Identifier: Apache-2.0
#pragma once
// Separate final-prefill operator; original T4/T32 helpers remain unchanged.
#include "qwen38_bf16_kv_commit.cuh"
namespace Garnet {
constexpr int64_t Qwen38BF16Prefill128TargetWidth=44564480;
inline bool BF16Prefill128CommitProfile(int batch,int chunk) noexcept {
    return batch==1&&chunk==128;
}
// FP32 target delta to canonical packed BF16 KV/FP32 linear state. No replay
// source or partial linear-state recomputation is accepted by this interface.
// A failed launch after copying begins poisons the request; no rollback promise.
int RunQwen38BF16Prefill128Commit(const float* target,float* canonical,int batch,
    int chunk,Qwen38BF16KVSpans,cudaStream_t) noexcept;
}
