// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "decoder_session.h"
#include "../../runtime/scheduler/continuous_batch_scheduler.h"
#include <map>
#include <memory>

namespace Garnet {
// Native policy/row-state bridge. The inference runtime supplies embeddings
// and calls session.run_leased with each returned plan's epochs and masks.
class MetalContinuousScheduler {
    class RowAllocator;
    std::unique_ptr<RowAllocator> allocator_;
    std::unique_ptr<ContinuousBatchScheduler> scheduler_;
    X::Value sessionOwner_;
    std::map<uint64_t,RequestStage> inflight_;
#ifdef GARNET_METAL_TOKEN_PREFIX_CACHE
    std::map<uint64_t,int> trackedExpected_;
#endif
    size_t rows_=0,capacity_=0;
    bool failed_=false;
    std::mutex mutex_;
    void Ready() const;
    bool CloseUnlocked();
public:
    MetalContinuousScheduler();
    ~MetalContinuousScheduler();
    bool Initialize(X::Value session,int64_t maxPending);
    bool SetPrefillChunk(int64_t tokens);
    bool CompletePrefillChunk(int64_t request,int64_t tokens);
    int64_t Submit(int64_t prompt,int64_t output,int64_t priority);
#ifdef GARNET_METAL_TOKEN_PREFIX_CACHE
    int64_t SubmitTokens(X::Value tokens,int64_t output,int64_t priority);
#endif
    std::string NextJson();
    int64_t CompletePrefill(int64_t request,int64_t token,bool finished);
    int64_t CompleteDecode(int64_t request,int64_t token,bool finished);
#ifdef GARNET_METAL_MTP_BRIDGE
    std::string GenerateMTP(X::Value stops);
#endif
    bool Cancel(int64_t request);
    std::string StatusJson();
    bool Close();
    BEGIN_PACKAGE(MetalContinuousScheduler)
        APISET().AddFunc<2>("initialize",&MetalContinuousScheduler::Initialize);
        APISET().AddFunc<1>("set_prefill_chunk",&MetalContinuousScheduler::SetPrefillChunk);
        APISET().AddFunc<2>("complete_prefill_chunk",&MetalContinuousScheduler::CompletePrefillChunk);
        APISET().AddFunc<3>("submit",&MetalContinuousScheduler::Submit);
#ifdef GARNET_METAL_TOKEN_PREFIX_CACHE
        APISET().AddFunc<3>("submit_tokens",&MetalContinuousScheduler::SubmitTokens);
#endif
        APISET().AddFunc<0>("next_json",&MetalContinuousScheduler::NextJson);
        APISET().AddFunc<3>("complete_prefill",&MetalContinuousScheduler::CompletePrefill);
        APISET().AddFunc<3>("complete_decode",&MetalContinuousScheduler::CompleteDecode);
#ifdef GARNET_METAL_MTP_BRIDGE
        APISET().AddFunc<1>("mtp_generate_json",&MetalContinuousScheduler::GenerateMTP);
#endif
        APISET().AddFunc<1>("cancel",&MetalContinuousScheduler::Cancel);
        APISET().AddFunc<0>("status_json",&MetalContinuousScheduler::StatusJson);
        APISET().AddFunc<0>("close",&MetalContinuousScheduler::Close);
    END_PACKAGE
};
}
