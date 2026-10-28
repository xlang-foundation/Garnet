// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cublasLt.h>
#include <cstddef>
#include <memory>
namespace Garnet {
struct ModelOptFP8QKM16Context;
struct ModelOptFP8QKM16Info {
    int device=-1,algorithm=-1,rows=0;
    size_t requiredWorkspace=0;
    cublasLtMatmulAlgo_t opaque{};
};
constexpr size_t ModelOptFP8QKM16BlasWorkspace=4ULL<<20;
std::shared_ptr<ModelOptFP8QKM16Context> CreateModelOptFP8QKM16Context(int rows);
bool GetModelOptFP8QKM16Info(const std::shared_ptr<ModelOptFP8QKM16Context>&,ModelOptFP8QKM16Info&) noexcept;
cublasStatus_t RunModelOptFP8QKM16Matmul(const std::shared_ptr<ModelOptFP8QKM16Context>&,
    const void* weights,const void* input,void* output,int batch,int rows,int columns,float globalScale,
    void* workspace,size_t workspaceBytes,cudaStream_t stream) noexcept;
}
