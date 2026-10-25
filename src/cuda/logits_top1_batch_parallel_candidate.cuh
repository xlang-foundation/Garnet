// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Unintegrated candidate: no production caller includes this header yet.
// Preserve the legacy batch sampler's -FLT_MAX/index0 initial candidate,
// ignored NaNs/-infinity, accepted +infinity, lowest-index ties and signed zero.
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace Garnet::ParallelBatchTop1Candidate {
constexpr float initialValue = -3.4028234663852886e38f;
constexpr int threads = 256;
constexpr int chunkSize = 1024;

__device__ __forceinline__ float read(float value) { return value; }
__device__ __forceinline__ float read(__nv_bfloat16 value) { return __bfloat162float(value); }

template<class Scalar>
__global__ void chunks(const Scalar* logits, unsigned long long* packedBest, int vocab) {
    __shared__ float values[threads];
    __shared__ int indices[threads];
    const int tid = threadIdx.x;
    const int first = blockIdx.x * chunkSize;
    const int end = first + min(chunkSize, vocab - first);
    const Scalar* row = logits + static_cast<size_t>(blockIdx.y) * vocab;
    float best = initialValue;
    int index = 0;
    for (size_t i = static_cast<size_t>(first) + tid; i < static_cast<size_t>(end); i += threads) {
        const float value = read(row[i]);
        if (value > best || (value == best && i < index)) {
            best = value;
            index = static_cast<int>(i);
        }
    }
    values[tid] = best;
    indices[tid] = index;
    __syncthreads();
    for (int stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) {
            const float other = values[tid + stride];
            const int otherIndex = indices[tid + stride];
            if (other > values[tid] || (other == values[tid] && otherIndex < indices[tid])) {
                values[tid] = other;
                indices[tid] = otherIndex;
            }
        }
        __syncthreads();
    }
    if (!tid && values[0] > initialValue) {
        const float value = values[0] == 0.0f ? 0.0f : values[0];
        const unsigned int raw = __float_as_uint(value);
        const unsigned int ordered = raw & 0x80000000u ? ~raw : raw ^ 0x80000000u;
        const unsigned long long packed = (static_cast<unsigned long long>(ordered) << 32)
            | (0xffffffffu - static_cast<unsigned int>(indices[0]));
        atomicMax(packedBest + blockIdx.y, packed);
    }
}

template<class Scalar>
__global__ void finish(const Scalar* logits, long long* ids, float* scores, int rows, int vocab) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const auto packed = reinterpret_cast<const unsigned long long*>(ids)[row];
    const int index = packed ? int(0xffffffffu - static_cast<unsigned int>(packed)) : 0;
    ids[row] = index;
    if (scores) scores[row] = packed ? read(logits[static_cast<size_t>(row) * vocab + index]) : initialValue;
}

// Uses only caller-owned ID storage as transient packed reduction storage.
// It must be disjoint from the readonly logits and optional score allocation.
// The candidate is deliberately bounded to current tensor-IO row admission.
template<class Scalar>
cudaError_t run(const Scalar* logits, long long* ids, float* scores, int rows, int vocab, cudaStream_t stream) {
    if (!logits || !ids || rows < 1 || rows > 128 || vocab < 1) return cudaErrorInvalidValue;
    auto status = cudaMemsetAsync(ids, 0, sizeof(long long) * rows, stream);
    if (status != cudaSuccess) return status;
    const unsigned int blocks = (static_cast<unsigned int>(vocab) - 1) / chunkSize + 1;
    chunks<<<dim3(blocks, rows), threads, 0, stream>>>(logits, reinterpret_cast<unsigned long long*>(ids), vocab);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    finish<<<1, 128, 0, stream>>>(logits, ids, scores, rows, vocab);
    return cudaGetLastError();
}
} // namespace Garnet::ParallelBatchTop1Candidate
