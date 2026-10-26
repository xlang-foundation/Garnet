// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cublasLt.h>
#include <cstddef>
#include <memory>
namespace Garnet {
struct ModelOptFP8Z128Context;
struct ModelOptFP8Z128Info {
    int device=-1,algorithm=-1;
    size_t requiredWorkspace=0;
    cublasLtMatmulAlgo_t opaque{};
};
constexpr size_t ModelOptFP8Z128BlasWorkspace=4ULL<<20;
std::shared_ptr<ModelOptFP8Z128Context> CreateModelOptFP8Z128Context();
bool GetModelOptFP8Z128Info(const std::shared_ptr<ModelOptFP8Z128Context>&,ModelOptFP8Z128Info&) noexcept;
cublasStatus_t RunModelOptFP8Z128Matmul(const std::shared_ptr<ModelOptFP8Z128Context>&,
    const void* weights,const void* input,void* output,int batch,int rows,int columns,float globalScale,
    void* workspace,size_t workspaceBytes,cudaStream_t stream) noexcept;
}
