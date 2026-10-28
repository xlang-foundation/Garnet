// SPDX-License-Identifier: Apache-2.0
#pragma once
#if !defined(GARNET_CPU_ONLY) && !defined(GARNET_METAL)
#include "xlang3/xlang3.h"
#include "../../cuda/qwen35_tree_commit.cuh"
#include <mutex>
#include <string>
namespace Garnet {
class CUDATreeCommit {
    Qwen35TreeCommitConfig config_{};
    X::Tensor linear_,full_,parameters_,workspace_,status_,selection_;
    bool ready_=false;mutable std::mutex mutex_;
    void CloseUnlocked();
public:
    ~CUDATreeCommit();
    bool Build(const std::string& checkpoint,const std::string& specification);
    //0committed;1GPU packet/control rejection BEFORE any canonical write.
    //Exceptions after dispatch mean completion unknown: caller must poison.
    int Commit(X::Value packed,X::Value canonical,X::Value parents,X::Value active,X::Value prefix,X::Value selection);
    std::string StatsJson() const;
    void Close();
    BEGIN_PACKAGE(CUDATreeCommit)
        APISET().AddFunc<2>("build",&CUDATreeCommit::Build);
        APISET().AddFunc<6>("commit",&CUDATreeCommit::Commit);
        APISET().AddFunc<0>("stats_json",&CUDATreeCommit::StatsJson);
        APISET().AddFunc<0>("close",&CUDATreeCommit::Close);
    END_PACKAGE
};
}
#endif
