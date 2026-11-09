// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>

namespace Garnet {
// Allocation checkpoints, not isolated process peaks. Timing-only mode avoids
// cudaMemGetInfo, whose driver work can perturb concurrent engine preparation.
inline bool TRTMemoryProfileEnabled() {
    const char* value=std::getenv("GARNET_TRT_LOG_ENGINE_MEMORY");
    return value && std::strcmp(value,"1")==0;
}
inline void LogTRTMemoryCheckpoint(const char* stage,const std::string& engine,
    size_t contextUpperBound=0,size_t contextCount=0) {
    const bool memory=TRTMemoryProfileEnabled();
    const char* timing=std::getenv("GARNET_TRT_LOG_ENGINE_TIMING");
    if(!memory&&!(timing&&std::strcmp(timing,"1")==0))return;
    const auto timestamp=std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    int device=-1;size_t free=0,total=0;
    if(cudaGetDevice(&device)!=cudaSuccess)return;
    if(memory&&cudaMemGetInfo(&free,&total)!=cudaSuccess)return;
    static std::mutex outputMutex;
    std::lock_guard<std::mutex> guard(outputMutex);
    std::cout<<(memory?"[TRTMemoryCheckpoint]":"[TRTPhaseCheckpoint]")
        <<" stage="<<stage<<" device="<<device<<" monotonic_us="<<timestamp;
    if(memory)std::cout<<" device_used_bytes="<<(total-free)<<" device_total_bytes="<<total;
    std::cout
        <<" context_device_memory_upper_bound_bytes="<<contextUpperBound
        <<" context_count="<<contextCount<<" engine="<<engine<<std::endl;
}
}
