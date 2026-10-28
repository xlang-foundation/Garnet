// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cublasLt.h>
#include <cstddef>
#include <memory>
namespace Garnet {
struct ModelOptFP8KV128Context;
struct ModelOptFP8KV128Info {
    int device=-1,algorithm=-1;
    size_t requiredWorkspace=0;
    cublasLtMatmulAlgo_t opaque{};
};
constexpr size_t ModelOptFP8KV128BlasWorkspace=4ULL<<20;
std::shared_ptr<ModelOptFP8KV128Context> CreateModelOptFP8KV128Context();
bool GetModelOptFP8KV128Info(const std::shared_ptr<ModelOptFP8KV128Context>&,ModelOptFP8KV128Info&) noexcept;
cublasStatus_t RunModelOptFP8KV128Matmul(const std::shared_ptr<ModelOptFP8KV128Context>&,
    const void* weights,const void* input,void* output,int batch,int rows,int columns,float globalScale,
    void* workspace,size_t workspaceBytes,cudaStream_t stream) noexcept;
}
