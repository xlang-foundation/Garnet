// SPDX-License-Identifier: Apache-2.0
#pragma once
#if !defined(GARNET_CPU_ONLY) && !defined(GARNET_METAL)
#include "xlang3/xlang3.h"
#include "../../cuda/qwen35_packed_commit.cuh"
#include <mutex>
#include <string>
namespace Garnet {
class CUDAPackedPrefix {
    Qwen35PackedCommitConfig config_{};
    X::Tensor linear_,full_,parameters_,workspace_;
    bool ready_=false;
    mutable std::mutex mutex_;
    void CloseUnlocked();
    void CheckCompactStatus(const X::Tensor& source) const;
    bool BuildPhaseOutput(const std::string& checkpoint,const std::string& specification);
public:
    ~CUDAPackedPrefix();
    bool Build(const std::string& checkpoint,const std::string& specification);
    void Commit(X::Value packed,X::Value canonical,X::Value spans);
    void CommitAll(X::Value packed,X::Value canonical,X::Value slots,X::Value active);
    std::string StatsJson() const;
    std::string PhaseOutputLayout() const{return "qwen35-phase-outputs-v1";}
    void Close();
    BEGIN_PACKAGE(CUDAPackedPrefix)
        APISET().AddFunc<2>("build",&CUDAPackedPrefix::Build);
        APISET().AddFunc<3>("commit",&CUDAPackedPrefix::Commit);
        APISET().AddFunc<4>("commit_all",&CUDAPackedPrefix::CommitAll);
        APISET().AddFunc<0>("stats_json",&CUDAPackedPrefix::StatsJson);
        APISET().AddFunc<0>("phase_output_layout",&CUDAPackedPrefix::PhaseOutputLayout);
        APISET().AddFunc<0>("close",&CUDAPackedPrefix::Close);
    END_PACKAGE
};
}
#endif
