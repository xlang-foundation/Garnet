// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cublasLt.h>
#include <cstddef>
#include <memory>
namespace Garnet {
struct ModelOptFP8VM16Context;
struct ModelOptFP8VM16Info {
    int device=-1,algorithm=-1,rows=0;
    size_t requiredWorkspace=0;
    cublasLtMatmulAlgo_t opaque{};
};
constexpr size_t ModelOptFP8VM16BlasWorkspace=4ULL<<20;
std::shared_ptr<ModelOptFP8VM16Context> CreateModelOptFP8VM16Context(int rows);
bool GetModelOptFP8VM16Info(const std::shared_ptr<ModelOptFP8VM16Context>&,ModelOptFP8VM16Info&) noexcept;
cublasStatus_t RunModelOptFP8VM16Matmul(const std::shared_ptr<ModelOptFP8VM16Context>&,
    const void* weights,const void* input,void* output,int batch,int rows,int columns,float globalScale,
    void* workspace,size_t workspaceBytes,cudaStream_t stream) noexcept;
}
