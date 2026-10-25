// SPDX-License-Identifier: Apache-2.0
#pragma once
#if !defined(GARNET_CPU_ONLY) && !defined(GARNET_METAL)
#include "xlang3/xlang3.h"
#include "bf16_low_batch_layout_contract.h"
#include <mutex>
#include <string>
namespace Garnet {
class CUDABF16LowBatchKVDeltaCommit {
    BF16LowBatchKVLayoutSpec layout_{};bool ready_=false,poisoned_=false;uint64_t commits_=0;mutable std::mutex mutex_;
public:
    bool Build(const std::string& specification);
    void Commit(X::Value target,X::Value linear,X::Value canonical,X::Value spans);
    std::string StatsJson() const;
    void Close();
    BEGIN_PACKAGE(CUDABF16LowBatchKVDeltaCommit)
        APISET().AddFunc<1>("build",&CUDABF16LowBatchKVDeltaCommit::Build);
        APISET().AddFunc<4>("commit",&CUDABF16LowBatchKVDeltaCommit::Commit);
        APISET().AddFunc<0>("stats_json",&CUDABF16LowBatchKVDeltaCommit::StatsJson);
        APISET().AddFunc<0>("close",&CUDABF16LowBatchKVDeltaCommit::Close);
    END_PACKAGE
};
}
#endif
