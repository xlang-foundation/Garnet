// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cublasLt.h>
#include <cstddef>
#include <memory>
namespace Garnet {
struct ModelOptFP8KM4X4Context;
struct ModelOptFP8KM4X4Info {
    int device=-1,algorithm=-1,rows=0;
    int inputRowsPerMatmul=4,matmulsPerEnqueue=4;
    size_t requiredWorkspace=0;
    cublasLtMatmulAlgo_t opaque{};
};
constexpr size_t ModelOptFP8KM4X4BlasWorkspace=4ULL<<20;
std::shared_ptr<ModelOptFP8KM4X4Context> CreateModelOptFP8KM4X4Context(int rows);
bool GetModelOptFP8KM4X4Info(const std::shared_ptr<ModelOptFP8KM4X4Context>&,ModelOptFP8KM4X4Info&) noexcept;
cublasStatus_t RunModelOptFP8KM4X4Matmul(const std::shared_ptr<ModelOptFP8KM4X4Context>&,
    const void* weights,const void* input,void* output,int batch,int rows,int columns,float globalScale,
    void* workspace,size_t workspaceBytes,cudaStream_t stream) noexcept;
}
