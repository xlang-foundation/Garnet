// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
namespace Garnet {
// A negative device preserves the caller's device for legacy profiles.
class CUDADeviceScope {
    int previous_=-1;
public:
    explicit CUDADeviceScope(int device) {
        auto status=cudaGetDevice(&previous_);
        if(status==cudaSuccess && device>=0 && device!=previous_)status=cudaSetDevice(device);
        if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
    }
    ~CUDADeviceScope(){if(previous_>=0)cudaSetDevice(previous_);}
    CUDADeviceScope(const CUDADeviceScope&)=delete;
    CUDADeviceScope& operator=(const CUDADeviceScope&)=delete;
};
}
