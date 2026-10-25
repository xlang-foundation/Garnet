// SPDX-License-Identifier: Apache-2.0
#pragma once
#if !defined(GARNET_CPU_ONLY) && !defined(GARNET_METAL)
#include "xlang3/xlang3.h"
#include "qwen38_bf16_kv_layout_contract.h"
#include <mutex>
#include <string>
namespace Garnet {
class CUDABF16KVDeltaCommit {
    BF16KVLayoutSpec layout_{};bool ready_=false,poisoned_=false;uint64_t commits_=0;mutable std::mutex mutex_;
public:
    bool Build(const std::string& specification);
    void Commit(X::Value target,X::Value linear,X::Value canonical,X::Value spans);
    std::string StatsJson() const;
    void Close();
    BEGIN_PACKAGE(CUDABF16KVDeltaCommit)
        APISET().AddFunc<1>("build",&CUDABF16KVDeltaCommit::Build);
        APISET().AddFunc<4>("commit",&CUDABF16KVDeltaCommit::Commit);
        APISET().AddFunc<0>("stats_json",&CUDABF16KVDeltaCommit::StatsJson);
        APISET().AddFunc<0>("close",&CUDABF16KVDeltaCommit::Close);
    END_PACKAGE
};
}
#endif
