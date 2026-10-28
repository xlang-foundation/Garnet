// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>
namespace Garnet {
constexpr int Qwen35TreeMaxNodes=64,Qwen35TreeMaxPath=8;
struct Qwen35TreeMetadataConfig {int batch,nodes,capacity;};
// Plain INT32 TensorRT storage. Invalid rows have valid==0 and error!=0.
// No canonical state/cache pointer is accepted by the metadata builder.
struct Qwen35TreeMetadata {
    int32_t valid,error,leaf_count,prefix;
    int32_t parent[Qwen35TreeMaxNodes],depth[Qwen35TreeMaxNodes];
    int32_t leaf[Qwen35TreeMaxNodes],owner[Qwen35TreeMaxNodes];
    int32_t active_length[Qwen35TreeMaxNodes];
    int32_t active_ancestor[Qwen35TreeMaxNodes][Qwen35TreeMaxPath];
};
static_assert(sizeof(Qwen35TreeMetadata)==836*sizeof(int32_t),"Tree metadata ABI must be plain INT32");
bool ValidQwen35TreeMetadataConfig(Qwen35TreeMetadataConfig) noexcept;
// Inputs: exact-integer FP32 parents[B,N], binary active[B,N], prefix[B].
// Host shape/alias failures return nonzero without output advance. GPU control
// failures set per-row status; every consumer must require valid==1.
int RunQwen35TreeMetadata(Qwen35TreeMetadataConfig,const float* parents,const float* active,
    const float* prefix,Qwen35TreeMetadata* output,cudaStream_t) noexcept;
}
