// SPDX-License-Identifier: Apache-2.0
#pragma once
// Isolated B1 operator. No legacy B4/B8 guard or model admission changes.
#include "qwen38_bf16_kv_commit.cuh"
namespace Garnet {
inline bool BF16LowBatchCommitCandidateProfile(int batch,int chunk) noexcept {
    return batch==1&&(chunk==4||chunk==32);
}
int RunQwen38BF16LowBatchCommitCandidate(const float* target,const float* linear,
    int64_t linearWidth,float* canonical,int batch,int chunk,Qwen38BF16KVSpans,cudaStream_t) noexcept;
}
