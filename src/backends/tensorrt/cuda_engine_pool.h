// SPDX-License-Identifier: Apache-2.0
#pragma once
#if !defined(GARNET_CPU_ONLY) && !defined(GARNET_METAL)
#include "xlang3/xlang3.h"
#include <cstdint>
#include <memory>
#include <string>
namespace Garnet {
// Explicit native residency for compiled engines. It retains immutable refitted
// weights; released model leases retire idle execution contexts independently.
class CUDAEnginePool {
public:
    struct State;
private:
    std::shared_ptr<State> state_;
public:
    CUDAEnginePool();
    ~CUDAEnginePool();
    void Configure(int64_t maximumWeightBytes);
    void Enter();
    void Leave();
    void Close();
    std::string StatsJson() const;
    static bool Active();
    static void Retain(const std::string& path,const std::shared_ptr<void>& execution);
    BEGIN_PACKAGE(CUDAEnginePool)
        APISET().AddFunc<1>("configure",&CUDAEnginePool::Configure);
        APISET().AddFunc<0>("enter",&CUDAEnginePool::Enter);
        APISET().AddFunc<0>("leave",&CUDAEnginePool::Leave);
        APISET().AddFunc<0>("close",&CUDAEnginePool::Close);
        APISET().AddFunc<0>("stats_json",&CUDAEnginePool::StatsJson);
    END_PACKAGE
};
}
#endif
