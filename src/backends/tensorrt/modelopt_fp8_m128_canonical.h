// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cublasLt.h>
#include <cstddef>
#include <memory>
namespace Garnet {
struct ModelOptFP8M128Context;
struct ModelOptFP8M128Info {
    int device=-1,algorithm=-1;
    size_t requiredWorkspace=0;
    cublasLtMatmulAlgo_t opaque{};
};
constexpr size_t ModelOptFP8M128BlasWorkspace=8ULL<<20;
std::shared_ptr<ModelOptFP8M128Context> CreateModelOptFP8M128Context();
bool GetModelOptFP8M128Info(const std::shared_ptr<ModelOptFP8M128Context>&,ModelOptFP8M128Info&) noexcept;
cublasStatus_t RunModelOptFP8M128Matmul(const std::shared_ptr<ModelOptFP8M128Context>&,
    const void* weights,const void* input,void* output,int batch,int rows,int columns,float globalScale,
    void* workspace,size_t workspaceBytes,cudaStream_t stream) noexcept;
}
