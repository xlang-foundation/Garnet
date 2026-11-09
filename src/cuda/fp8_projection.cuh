// SPDX-License-Identifier: Apache-2.0
#pragma once
#if CUDART_VERSION >= 12080
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <algorithm>

namespace Garnet {
namespace {
constexpr int ProjectionPackThreads = 256;

__global__ void ProjectionAmax(const __nv_bfloat16* source, int count, unsigned* maximum) {
    float local = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        local = fmaxf(local, fabsf(__bfloat162float(source[i])));
    }
    constexpr int warpSize = 32;
    __shared__ float warpPeaks[ProjectionPackThreads / warpSize];
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        local = fmaxf(local, __shfl_down_sync(0xffffffff, local, offset));
    if (threadIdx.x % warpSize == 0) warpPeaks[threadIdx.x / warpSize] = local;
    __syncthreads();
    if (threadIdx.x < warpSize) {
        local = threadIdx.x < ProjectionPackThreads / warpSize ? warpPeaks[threadIdx.x] : 0;
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            local = fmaxf(local, __shfl_down_sync(0xffffffff, local, offset));
        if (threadIdx.x == 0) atomicMax(maximum, __float_as_uint(local));
    }
}

__global__ void ProjectionScale(const unsigned* maximum, float* scale) {
    *scale = fmaxf(__uint_as_float(*maximum) / 448.0f, 1.0e-20f);
}

__global__ void ProjectionPack(const __nv_bfloat16* source, unsigned char* packed, int count, const float* scale) {
    const float inverseScale = 1.0f / *scale;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        packed[i] = __nv_cvt_float_to_fp8(__bfloat162float(source[i]) * inverseScale, __NV_SATFINITE, __NV_E4M3);
    }
}

struct FP8ProjectionDescriptors {
    cublasLtMatmulDesc_t operation{};
    cublasLtMatrixLayout_t weight{}, input{}, output{};
    cublasLtMatmulPreference_t preference{};

    ~FP8ProjectionDescriptors() {
        if (preference) cublasLtMatmulPreferenceDestroy(preference);
        if (output) cublasLtMatrixLayoutDestroy(output);
        if (input) cublasLtMatrixLayoutDestroy(input);
        if (weight) cublasLtMatrixLayoutDestroy(weight);
        if (operation) cublasLtMatmulDescDestroy(operation);
    }
};

struct FP8ProjectionScratch {
    cudaStream_t stream;
    void* data{};
    ~FP8ProjectionScratch() { if (data) cudaFreeAsync(data, stream); }
};

bool PackProjectionOperand(const void* source, unsigned char* packed, int count,
        unsigned* maximum, float* scale, cudaStream_t stream) {
    if (cudaMemsetAsync(maximum, 0, sizeof(*maximum), stream) != cudaSuccess) return false;
    const int blocks = std::min(4096, (count + ProjectionPackThreads - 1) / ProjectionPackThreads);
    const auto* bf16 = static_cast<const __nv_bfloat16*>(source);
    ProjectionAmax<<<blocks, ProjectionPackThreads, 0, stream>>>(bf16, count, maximum);
    ProjectionScale<<<1, 1, 0, stream>>>(maximum, scale);
    ProjectionPack<<<blocks, ProjectionPackThreads, 0, stream>>>(bf16, packed, count, scale);
    return cudaGetLastError() == cudaSuccess;
}

// Pack checkpoint BF16 operands on the execution stream. Keep the graph's
// BF16 output boundary and accumulate the E4M3 product in FP32.
bool RunFP8Projection(cublasLtHandle_t handle, const void* input, const void* weight, void* output,
        int rows, int inner, int columns, cudaStream_t stream, void* workspace, size_t workspaceBytes) {
    const int inputCount = rows * inner;
    const int weightCount = columns * inner;
    FP8ProjectionScratch storage{stream};
    constexpr size_t scalarsBytes = 3 * sizeof(float);
    if (cudaMallocAsync(&storage.data, size_t(inputCount) + weightCount + scalarsBytes, stream) != cudaSuccess) return false;
    auto* packedInput = static_cast<unsigned char*>(storage.data);
    auto* packedWeight = packedInput + inputCount;
    auto* maximum = reinterpret_cast<unsigned*>(packedWeight + weightCount);
    auto* inputScale = reinterpret_cast<float*>(maximum + 1);
    auto* weightScale = inputScale + 1;

    FP8ProjectionDescriptors d;
    const cublasOperation_t transpose = CUBLAS_OP_T;
    if (cublasLtMatmulDescCreate(&d.operation, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulDescSetAttribute(d.operation, CUBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulDescSetAttribute(d.operation, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &weightScale, sizeof(weightScale)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulDescSetAttribute(d.operation, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &inputScale, sizeof(inputScale)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&d.weight, CUDA_R_8F_E4M3, inner, columns, inner) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&d.input, CUDA_R_8F_E4M3, inner, rows, inner) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&d.output, CUDA_R_16BF, columns, rows, columns) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulPreferenceCreate(&d.preference) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulPreferenceSetAttribute(d.preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspaceBytes, sizeof(workspaceBytes)) != CUBLAS_STATUS_SUCCESS) return false;

    cublasLtMatmulHeuristicResult_t candidates[8]{};
    int count = 0;
    if (cublasLtMatmulAlgoGetHeuristic(handle, d.operation, d.weight, d.input, d.output, d.output,
            d.preference, 8, candidates, &count) != CUBLAS_STATUS_SUCCESS || count == 0) return false;
    if (!PackProjectionOperand(input, packedInput, inputCount, maximum, inputScale, stream) ||
        !PackProjectionOperand(weight, packedWeight, weightCount, maximum, weightScale, stream)) return false;
    const float alpha = 1, beta = 0;
    for (int i = 0; i < count; ++i) {
        if (candidates[i].state != CUBLAS_STATUS_SUCCESS) continue;
        const auto status = cublasLtMatmul(handle, d.operation, &alpha, packedWeight, d.weight, packedInput, d.input,
            &beta, output, d.output, output, d.output, &candidates[i].algo, workspace, workspaceBytes, stream);
        if (status == CUBLAS_STATUS_SUCCESS) return true;
        if (status != CUBLAS_STATUS_NOT_SUPPORTED) return false;
    }
    return false;
}
}
}
#endif
