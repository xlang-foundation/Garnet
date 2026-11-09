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
#ifdef GARNET_METAL_WIDE_OWNER_PREFILL
    bool adaptiveProfiles_=false;
    size_t maximumPrefillChunk_=128;
#endif
#if defined(GARNET_METAL_DIAGNOSTIC_LAYER_BULK) && defined(GARNET_METAL_TOKEN_PREFIX_CACHE)
    std::vector<ScheduledRequest> groupedRequests_;
    int groupedChunks_=0,maximumGroupedChunks_=0;
    uint64_t groupedTicket_=0;
    bool groupedExecuted_=false;
    bool GroupedRequest(uint64_t id) const;
    void EraseGroupedRequest(uint64_t id);
#endif
    bool failed_=false;
    std::mutex mutex_;
    void Ready() const;
    bool CloseUnlocked();
public:
    MetalContinuousScheduler();
    ~MetalContinuousScheduler();
    bool Initialize(X::Value session,int64_t maxPending);
    bool SetPrefillChunk(int64_t tokens);
    bool SetPrefillCompletionPriority(bool enabled);
#ifdef GARNET_METAL_WIDE_OWNER_PREFILL
    bool SetPrefillCompletionOwners(int64_t owners);
#endif
#if defined(GARNET_METAL_DIAGNOSTIC_LAYER_BULK) && defined(GARNET_METAL_TOKEN_PREFIX_CACHE)
    bool SetBulkPrefill(int64_t chunks);
    X::Value RunBulkPrefill(X::Value hiddenChunks,int64_t ticket);
#endif
    bool CompletePrefillChunk(int64_t request,int64_t tokens);
    int64_t Submit(int64_t prompt,int64_t output,int64_t priority);
#ifdef GARNET_METAL_TOKEN_PREFIX_CACHE
    int64_t SubmitTokens(X::Value tokens,int64_t output,int64_t priority);
#endif
    std::string NextJson();
    int64_t CompletePrefill(int64_t request,int64_t token,bool finished);
    int64_t CompleteDecode(int64_t request,int64_t token,bool finished);
#ifdef GARNET_METAL_LOOKAHEAD_RUNTIME
    std::string GenerateLookahead(X::Value stops);
#endif
#ifdef GARNET_METAL_DENSE_VERIFY_SCHEDULER
    std::string VerifyDecode(X::Value hidden,X::Value positions,X::Value slots,X::Value active,X::Value tokens,X::Value stops);
#endif
#ifdef GARNET_METAL_MTP_BRIDGE
    std::string GenerateMTP(X::Value stops);
#endif
    bool Cancel(int64_t request);
    std::string StatusJson();
    bool Close();
    BEGIN_PACKAGE(MetalContinuousScheduler)
        APISET().AddFunc<2>("initialize",&MetalContinuousScheduler::Initialize);
#if defined(GARNET_METAL_DIAGNOSTIC_LAYER_BULK) && defined(GARNET_METAL_TOKEN_PREFIX_CACHE)
        APISET().AddFunc<1>("set_bulk_prefill",&MetalContinuousScheduler::SetBulkPrefill);
        APISET().AddFunc<2>("run_bulk_prefill",&MetalContinuousScheduler::RunBulkPrefill);
#endif
        APISET().AddFunc<1>("set_prefill_chunk",&MetalContinuousScheduler::SetPrefillChunk);
        APISET().AddFunc<1>("set_prefill_completion_priority",&MetalContinuousScheduler::SetPrefillCompletionPriority);
#ifdef GARNET_METAL_WIDE_OWNER_PREFILL
        APISET().AddFunc<1>("set_prefill_completion_owners",&MetalContinuousScheduler::SetPrefillCompletionOwners);
#endif
        APISET().AddFunc<2>("complete_prefill_chunk",&MetalContinuousScheduler::CompletePrefillChunk);
        APISET().AddFunc<3>("submit",&MetalContinuousScheduler::Submit);
#ifdef GARNET_METAL_TOKEN_PREFIX_CACHE
        APISET().AddFunc<3>("submit_tokens",&MetalContinuousScheduler::SubmitTokens);
#endif
        APISET().AddFunc<0>("next_json",&MetalContinuousScheduler::NextJson);
        APISET().AddFunc<3>("complete_prefill",&MetalContinuousScheduler::CompletePrefill);
        APISET().AddFunc<3>("complete_decode",&MetalContinuousScheduler::CompleteDecode);
#ifdef GARNET_METAL_LOOKAHEAD_RUNTIME
        APISET().AddFunc<1>("lookahead_generate_json",&MetalContinuousScheduler::GenerateLookahead);
#endif
#ifdef GARNET_METAL_DENSE_VERIFY_SCHEDULER
        APISET().AddFunc<6>("verify_decode_json",&MetalContinuousScheduler::VerifyDecode);
#endif
#ifdef GARNET_METAL_MTP_BRIDGE
        APISET().AddFunc<1>("mtp_generate_json",&MetalContinuousScheduler::GenerateMTP);
#endif
        APISET().AddFunc<1>("cancel",&MetalContinuousScheduler::Cancel);
        APISET().AddFunc<0>("status_json",&MetalContinuousScheduler::StatusJson);
        APISET().AddFunc<0>("close",&MetalContinuousScheduler::Close);
    END_PACKAGE
};
}
