// SPDX-License-Identifier: Apache-2.0
#include "continuous_scheduler.h"
#include "../../runtime/scheduler/packed_token_plan.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <limits>
#include <climits>
namespace Garnet {
#ifdef GARNET_METAL_DENSE_PREFILL_MAX_TOKENS
static constexpr size_t kSchedulerPrefillChunkLimit=GARNET_METAL_DENSE_PREFILL_MAX_TOKENS;
#elif defined(GARNET_METAL_QWEN38_LARGE_PREFILL)
static constexpr size_t kSchedulerPrefillChunkLimit=256;
#else
static constexpr size_t kSchedulerPrefillChunkLimit=64;
#endif
class MetalContinuousScheduler::RowAllocator final:public KVSequenceAllocator {
    struct Sequence {int64_t row=0,epoch=0;int length=0;};
    MetalDecoderSession* session_;
    size_t rows_,capacity_;
    std::map<uint64_t,Sequence> sequences_;
    void Validate(const Sequence& sequence) const {
        if(!session_->RequestRowMatches(size_t(sequence.row),uint64_t(sequence.epoch)))
            throw std::logic_error("Native scheduler lost its Metal request row lease");
    }
public:
    RowAllocator(MetalDecoderSession* session,size_t rows,size_t capacity):session_(session),rows_(rows),capacity_(capacity){}
    bool Reserve(uint64_t id,int capacity) override {
        if(!id||id>uint64_t(INT64_MAX)||capacity<0||size_t(capacity)>capacity_||
           sequences_.size()>=rows_||sequences_.count(id))return false;
        // Allocate host metadata before clearing/claiming a GPU request row.
        auto entry=sequences_.emplace(id,Sequence{}).first;
        try {
            auto lease=session_->ClaimRequestRow(int64_t(id));
            entry->second.row=lease.Get(uint64_t(0)).ToInt64();entry->second.epoch=lease.Get(uint64_t(1)).ToInt64();
        }catch(...){sequences_.erase(entry);throw;}
        return true;
    }
    bool EnsureCapacity(uint64_t id,int capacity) override {
        auto found=sequences_.find(id);
        if(found==sequences_.end()||capacity<0||size_t(capacity)>capacity_)return false;
        Validate(found->second);return true;
    }
    bool CommitLength(uint64_t id,int length) override {
        if(!EnsureCapacity(id,length))return false;
        sequences_.at(id).length=length;return true;
    }
    bool Release(uint64_t id) override {
        auto found=sequences_.find(id);if(found==sequences_.end())return false;
        if(!session_->ReleaseRequestRow(found->second.row,found->second.epoch))
            throw std::logic_error("Native scheduler cannot release a stale request row");
        sequences_.erase(found);return true;
    }
    int LogicalLength(uint64_t id) const override {
        auto found=sequences_.find(id);if(found==sequences_.end())return -1;
        Validate(found->second);return found->second.length;
    }
    std::vector<int> PageTable(uint64_t) const override{return {};}
    KVSequencePlacement Placement(uint64_t id) const override {
        auto found=sequences_.find(id);if(found==sequences_.end())return {};
        Validate(found->second);return {int(found->second.row),uint64_t(found->second.epoch)};
    }
    bool Clear() noexcept {
        bool result=true;
        for(const auto& entry:sequences_) {
            try {result=session_->ReleaseRequestRow(entry.second.row,entry.second.epoch)&&result;}
            catch(...){result=false;}
        }
        sequences_.clear();return result;
    }
};
MetalContinuousScheduler::MetalContinuousScheduler()=default;
MetalContinuousScheduler::~MetalContinuousScheduler(){try{Close();}catch(...){}}
void MetalContinuousScheduler::Ready() const {
    if(!scheduler_||!allocator_||failed_)throw std::logic_error("Native Metal scheduler is closed or failed");
}
bool MetalContinuousScheduler::Initialize(X::Value value,int64_t pending) {
    std::lock_guard<std::mutex> lock(mutex_);
    if(scheduler_)throw std::logic_error("Close the native Metal scheduler before reinitializing it");
    if(value.runtime()!=Host()->runtime)throw std::invalid_argument("Native scheduler session runtime mismatch");
    auto session=value.NativeData<MetalDecoderSession>();
    if(!session)throw std::invalid_argument("Native scheduler requires a Metal decoder session");
    const auto layout=session->RequestRowLayout();
    if(!layout.first||layout.first>64||layout.second>size_t(INT_MAX)/layout.first||
       pending<int64_t(layout.first)||pending>4096)
        throw std::invalid_argument("Native scheduler row/context/pending bounds exceeded");
#ifdef GARNET_METAL_DENSE_PREFILL_MAX_TOKENS
    if(layout.first>4||layout.second>4096||layout.first>kSchedulerPrefillChunkLimit)
        throw std::invalid_argument("Native dense scheduler exceeds compiled row/context bounds");
#endif
    ContinuousBatchSchedulerConfig config;
    config.decodeBuckets={int(layout.first)};config.maxPrefillBatch=int(layout.first);
    config.maxPrefillTokensPerTick=int(layout.first*layout.second);config.maxActiveRequests=int(pending);
#ifdef GARNET_METAL_DENSE_PREFILL_MAX_TOKENS
    config.maxPrefillTokensPerTick=int(kSchedulerPrefillChunkLimit);
    config.prefillChunkTokens=int(std::min(layout.second,kSchedulerPrefillChunkLimit/layout.first));
#endif
    auto policy=std::make_unique<ContinuousBatchScheduler>(config);
    auto allocator=std::make_unique<RowAllocator>(session,layout.first,layout.second);
    sessionOwner_=value;rows_=layout.first;capacity_=layout.second;
    scheduler_=std::move(policy);allocator_=std::move(allocator);failed_=false;return true;
}
bool MetalContinuousScheduler::SetPrefillChunk(int64_t tokens) {
    std::lock_guard<std::mutex> lock(mutex_);Ready();
    if(tokens<0||uint64_t(tokens)>kSchedulerPrefillChunkLimit||uint64_t(tokens)>capacity_||!inflight_.empty())return false;
#ifdef GARNET_METAL_DENSE_PREFILL_MAX_TOKENS
    // The compiled graph has all physical rows, even when only one is active.
    // A per-request chunk must fit their shared token-row workspace. Zero would
    // reenable unbounded whole-prompt scheduling in the portable policy.
    if(!tokens||uint64_t(tokens)>kSchedulerPrefillChunkLimit/rows_)return false;
#endif
    return scheduler_->SetPrefillChunkTokens(int(tokens));
}
bool MetalContinuousScheduler::CompletePrefillChunk(int64_t id,int64_t tokens) {
    std::lock_guard<std::mutex> lock(mutex_);Ready();
    auto found=inflight_.find(uint64_t(id));
    if(id<=0||tokens<1||uint64_t(tokens)>kSchedulerPrefillChunkLimit||found==inflight_.end()||found->second!=RequestStage::Prefill)return false;
    try {
        if(!scheduler_->CompletePrefillChunk(uint64_t(id),int(tokens),*allocator_))return false;
        inflight_.erase(found);return true;
    }catch(...){failed_=true;throw;}
}
int64_t MetalContinuousScheduler::Submit(int64_t prompt,int64_t output,int64_t priority) {
    std::lock_guard<std::mutex> lock(mutex_);Ready();
    if(prompt<1||output<1||uint64_t(prompt)>capacity_||uint64_t(output)>capacity_-size_t(prompt)||
       priority<INT_MIN||priority>INT_MAX)return 0;
    GenerationRequest request;request.promptTokens=int(prompt);request.maxNewTokens=int(output);request.priority=int(priority);
    const auto id=scheduler_->Submit(request);
    if(id>uint64_t(INT64_MAX)){failed_=true;throw std::overflow_error("Native scheduler request identities exhausted");}
    return int64_t(id);
}
std::string MetalContinuousScheduler::NextJson() {
    std::lock_guard<std::mutex> lock(mutex_);Ready();
    if(!inflight_.empty())throw std::logic_error("Complete or cancel the prior native scheduler tick before scheduling again");
    try {
        const auto tick=scheduler_->Schedule(*allocator_);nlohmann::json batches=nlohmann::json::array();
        const auto packed=BuildPackedTokenPlan(tick,*allocator_,int(rows_*capacity_),int(capacity_));
        nlohmann::json segments=nlohmann::json::array();
        for(const auto& segment:packed.segments) {
            const auto& request=segment.request;
            segments.push_back({{"id",request.id},{"row",request.requestRow},{"epoch",request.rowEpoch},
                {"stage",segment.stage==RequestStage::Decode?"decode":"prefill"},
                {"packed_offset",segment.packedOffset},{"token_count",segment.tokenCount},
                {"slot",request.slotPosition},{"absolute_position",segment.absolutePosition}});
        }
        for(const auto& batch:tick.batches) {
            std::vector<int> mask(rows_,0);std::vector<uint64_t> epochs(rows_,0);
            nlohmann::json requests=nlohmann::json::array();
            for(const auto& request:batch.requests) {
                if(request.requestRow<0||size_t(request.requestRow)>=rows_||!request.rowEpoch||mask[size_t(request.requestRow)])
                    throw std::logic_error("Native scheduler supplied an invalid physical row plan");
                const auto row=size_t(request.requestRow);mask[row]=1;epochs[row]=request.rowEpoch;
                requests.push_back({{"id",request.id},{"row",row},{"epoch",request.rowEpoch},{"token_id",request.tokenId},
                    {"slot",request.slotPosition},{"context_length",request.contextLength},{"prompt_tokens",request.promptTokens},
                    {"generated_tokens",request.generatedTokens},{"position_delta",request.positionDelta},
                    {"prefill_offset",request.prefillOffset},{"prefill_tokens",request.prefillTokens}});
                if(!inflight_.emplace(request.id,batch.stage).second)throw std::logic_error("Duplicate request in native scheduler tick");
            }
            batches.push_back({{"stage",batch.stage==RequestStage::Prefill?"prefill":"decode"},
                {"bucket_size",rows_},{"active_rows",mask},{"epochs",epochs},{"requests",requests}});
        }
        return nlohmann::json{{"batches",batches},
            {"packed",{{"token_count",packed.tokenCount},{"segments",segments},{"execution","metadata_only"}}},
            {"scope","Native policy and Metal row state; dispatch via run_leased"}}.dump();
    }catch(...){failed_=true;throw;}
}
int64_t MetalContinuousScheduler::CompletePrefill(int64_t id,int64_t token,bool finished) {
    std::lock_guard<std::mutex> lock(mutex_);Ready();
    auto found=inflight_.find(uint64_t(id));
    if(id<=0||token<0||token>INT_MAX||found==inflight_.end()||found->second!=RequestStage::Prefill)return -1;
    try {
        if(!scheduler_->CompletePrefill(uint64_t(id),int(token),*allocator_,finished))return -1;
        const auto stage=scheduler_->Stage(uint64_t(id));inflight_.erase(found);
        if(stage==RequestStage::Finished)scheduler_->ForgetFinished(uint64_t(id));
        return int64_t(stage);
    }catch(...){failed_=true;throw;}
}
int64_t MetalContinuousScheduler::CompleteDecode(int64_t id,int64_t token,bool finished) {
    std::lock_guard<std::mutex> lock(mutex_);Ready();
    auto found=inflight_.find(uint64_t(id));
    if(id<=0||token<0||token>INT_MAX||found==inflight_.end()||found->second!=RequestStage::Decode)return -1;
    try {
        if(!scheduler_->CompleteDecode(uint64_t(id),int(token),finished,*allocator_))return -1;
        const auto stage=scheduler_->Stage(uint64_t(id));inflight_.erase(found);
        if(stage==RequestStage::Finished)scheduler_->ForgetFinished(uint64_t(id));
        return int64_t(stage);
    }catch(...){failed_=true;throw;}
}
#ifdef GARNET_METAL_MTP_BRIDGE
std::string MetalContinuousScheduler::GenerateMTP(X::Value stops) {
    std::lock_guard<std::mutex> lock(mutex_);Ready();
    auto session=sessionOwner_.NativeData<MetalDecoderSession>();
    auto known=X::Value::List(Host()),remaining=X::Value::List(Host()),epochs=X::Value::List(Host());
    struct Item {uint64_t id;int row,token,budget;uint64_t epoch;std::vector<int> emitted;bool eos=false;};
    std::vector<Item> items;std::vector<int> tokens(rows_,0),budgets(rows_,0);std::vector<uint64_t> leases(rows_,0);
    for(const auto& entry:inflight_)if(entry.second==RequestStage::Decode) {
        int token=0,budget=0;
        if(!scheduler_->DecodeProgress(entry.first,token,budget))throw std::logic_error("MTP scheduler decode progress invalid");
        auto placement=allocator_->Placement(entry.first);
        if(placement.requestRow<0||size_t(placement.requestRow)>=rows_||!placement.rowEpoch||leases[size_t(placement.requestRow)])
            throw std::logic_error("MTP scheduler row placement invalid");
        const auto row=size_t(placement.requestRow);tokens[row]=token;budgets[row]=budget;leases[row]=placement.rowEpoch;
        items.push_back({entry.first,placement.requestRow,token,budget,placement.rowEpoch,{}});
    }
    if(items.empty())throw std::invalid_argument("Schedule a native MTP decode batch first");
    for(size_t row=0;row<rows_;++row){known.Append(X::Value(int64_t(tokens[row])));remaining.Append(X::Value(int64_t(budgets[row])));epochs.Append(X::Value(int64_t(leases[row])));}
    try {
        auto result=session->MTPGenerateScheduled(known,remaining,stops,epochs);
        const auto statistics=nlohmann::json::parse(session->MTPGenerationStatsJson());
        // Validate all native row results before publishing any scheduler progress.
        for(auto& item:items) {
            auto row=result.Get(uint64_t(item.row)),ids=row.Get(uint64_t(0));
            const auto count=row.Get(uint64_t(1)).ToInt64();
            if(count<1||count>4||count>item.budget||uint64_t(count)!=ids.Size())throw std::logic_error("Native MTP scheduler result budget mismatch");
            for(uint64_t i=0;i<ids.Size();++i){auto token=ids.Get(i);if(!token.IsInt64()||token.ToInt64()<0||token.ToInt64()>INT_MAX)throw std::logic_error("Native MTP scheduler result token invalid");item.emitted.push_back(int(token.ToInt64()));}
            auto eos=row.Get(uint64_t(2));if(eos.raw().tag!=X3_TAG_BOOL)throw std::logic_error("Native MTP scheduler EOS status invalid");item.eos=eos.ToInt64()!=0;
        }
        nlohmann::json completed=nlohmann::json::array();
        for(const auto& item:items) {
            if(!scheduler_->CompleteDecodeTokens(item.id,item.emitted,item.eos,*allocator_))throw std::logic_error("Native MTP scheduler commit failed");
            const auto stage=scheduler_->Stage(item.id);inflight_.erase(item.id);
            completed.push_back({{"id",item.id},{"row",item.row},{"epoch",item.epoch},{"known",item.token},
                {"remaining_before",item.budget},{"emitted",item.emitted},{"consumed",item.emitted.size()},
                {"eos",item.eos},{"finished",stage==RequestStage::Finished}});
            if(stage==RequestStage::Finished)scheduler_->ForgetFinished(item.id);
        }
        return nlohmann::json{{"requests",completed},{"generation",statistics},
            {"scope","Native scheduled MTP generation and exact per-request progress; no serving/speed admission"}}.dump();
    }catch(...){failed_=true;throw;}
}
#endif
bool MetalContinuousScheduler::Cancel(int64_t id) {
    std::lock_guard<std::mutex> lock(mutex_);Ready();if(id<=0)return false;
    try {
        if(!scheduler_->Cancel(uint64_t(id),*allocator_))return false;
        inflight_.erase(uint64_t(id));scheduler_->ForgetFinished(uint64_t(id));return true;
    }catch(...){failed_=true;throw;}
}
std::string MetalContinuousScheduler::StatusJson() {
    std::lock_guard<std::mutex> lock(mutex_);Ready();
    return nlohmann::json{{"active",scheduler_->ActiveRequestCount()},{"queued",scheduler_->QueuedRequestCount()},
        {"tracked",scheduler_->TrackedRequestCount()},{"inflight",inflight_.size()},{"rows",rows_},{"capacity",capacity_}}.dump();
}
bool MetalContinuousScheduler::CloseUnlocked() {
    bool result=!allocator_||allocator_->Clear();
    inflight_.clear();scheduler_.reset();allocator_.reset();sessionOwner_=X::Value();
    rows_=capacity_=0;failed_=false;return result;
}
bool MetalContinuousScheduler::Close(){std::lock_guard<std::mutex> lock(mutex_);return CloseUnlocked();}
}
