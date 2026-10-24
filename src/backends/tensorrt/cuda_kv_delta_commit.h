// SPDX-License-Identifier: Apache-2.0
#pragma once
#if !defined(GARNET_CPU_ONLY) && !defined(GARNET_METAL)
#include "xlang3/xlang3.h"
#include "../../cuda/qwen35_kv_delta_commit.cuh"
#include <mutex>
#include <string>
namespace Garnet {
class CUDAKVDeltaCommit {
    X::Tensor linear_,full_;bool ready_=false;mutable std::mutex mutex_;
    void CloseUnlocked();
public:
    ~CUDAKVDeltaCommit();
    bool Build(const std::string& specification);
    void Commit(X::Value target,X::Value linear,X::Value canonical,X::Value spans);
    std::string StatsJson() const;
    void Close();
    BEGIN_PACKAGE(CUDAKVDeltaCommit)
        APISET().AddFunc<1>("build",&CUDAKVDeltaCommit::Build);
        APISET().AddFunc<4>("commit",&CUDAKVDeltaCommit::Commit);
        APISET().AddFunc<0>("stats_json",&CUDAKVDeltaCommit::StatsJson);
        APISET().AddFunc<0>("close",&CUDAKVDeltaCommit::Close);
    END_PACKAGE
};
}
#endif
