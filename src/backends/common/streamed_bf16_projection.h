// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "safetensors_index.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
namespace Garnet {
// Checkpoint-backed dense GEMM, independent of any model architecture.
struct StreamedBF16ProjectionSpec {
    std::string file;
    uint64_t offset=0,bytes=0,fileBytes=0;
    int64_t modified=0;
    int columns=0,inner=0;
    bool registeredSource=false;
    bool floatOutput=false; // Opt-in FP32 result; retain BF16 operands and weights.
    bool fp8Compute=false; // Pack BF16 operands for E4M3 compute and return BF16.
    bool cacheWeights=true; // Graph residency hint; does not change arithmetic.
    static StreamedBF16ProjectionSpec FromMetadata(const SafeTensorMetadata&);
    static StreamedBF16ProjectionSpec Parse(const std::string&);
    std::string Serialize() const;
    void Validate() const;
};
class StreamedBF16Projection {
    struct State;
    std::unique_ptr<State> state_;
public:
    explicit StreamedBF16Projection(StreamedBF16ProjectionSpec);
    ~StreamedBF16Projection();
    StreamedBF16Projection(const StreamedBF16Projection&)=delete;
    StreamedBF16Projection& operator=(const StreamedBF16Projection&)=delete;
    void Run(const void* input,void* output,int rows,cudaStream_t stream,void* workspace,size_t workspaceBytes);
};
std::string StreamedBF16ProjectionStats();
void ClearStreamedProjectionDeviceCache(const std::vector<std::string>& checkpointFiles);
}
