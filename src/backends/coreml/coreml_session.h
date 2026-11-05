// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
namespace Garnet {
// CPU transport buffers. Core ML owns compiled execution and persistent state.
struct CoreMLTensor {
    std::string name,dtype; // float16, float32, int32
    std::vector<int64_t> shape;
    std::vector<uint8_t> bytes;
};
class CoreMLSession {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    CoreMLSession();
    ~CoreMLSession();
    CoreMLSession(const CoreMLSession&)=delete;
    CoreMLSession& operator=(const CoreMLSession&)=delete;
    // Compiled .mlmodelc only. Profile supplies compute_units (cpu/cpu_ane),
    // maximum_io_bytes, maximum_state_bytes and inputs/outputs mappings from
    // feature name to {dtype,maximum_shape}. Actual shapes may be smaller.
    void Open(const std::string& compiledPath,const std::string& profileJSON);
    std::vector<CoreMLTensor> Predict(const std::vector<CoreMLTensor>& inputs);
    std::vector<CoreMLTensor> SnapshotState();
    void RestoreState(const std::vector<CoreMLTensor>& state);
    void Reset();
    void Close();
    std::string ReportJSON() const;
};
}
