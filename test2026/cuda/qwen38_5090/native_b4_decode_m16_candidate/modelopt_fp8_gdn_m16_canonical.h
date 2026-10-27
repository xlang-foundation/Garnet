// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cublasLt.h>
#include <cstddef>
#include <memory>
namespace Garnet {
struct ModelOptFP8GDNM16Context;
struct ModelOptFP8GDNM16Info {
    int device=-1,algorithm=-1,rows=0;
    size_t requiredWorkspace=0;
    cublasLtMatmulAlgo_t opaque{};
};
constexpr size_t ModelOptFP8GDNM16BlasWorkspace=4ULL<<20;
std::shared_ptr<ModelOptFP8GDNM16Context> CreateModelOptFP8GDNM16Context(int rows);
bool GetModelOptFP8GDNM16Info(const std::shared_ptr<ModelOptFP8GDNM16Context>&,ModelOptFP8GDNM16Info&) noexcept;
cublasStatus_t RunModelOptFP8GDNM16Matmul(const std::shared_ptr<ModelOptFP8GDNM16Context>&,
    const void* weights,const void* input,void* output,int batch,int rows,int columns,float globalScale,
    void* workspace,size_t workspaceBytes,cudaStream_t stream) noexcept;
}
