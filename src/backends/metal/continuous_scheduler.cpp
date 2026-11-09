// SPDX-License-Identifier: Apache-2.0
#include "continuous_scheduler.h"
#include "lowering_context.h"
#include <atomic>
#include "token_prefix_profile.h"
#include "../../runtime/scheduler/packed_token_plan.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <limits>
#include <climits>
#include <cmath>
namespace Garnet {
#ifdef GARNET_METAL_DENSE_PREFILL_MAX_TOKENS
static constexpr size_t kSchedulerPrefillChunkLimit=GARNET_METAL_DENSE_PREFILL_MAX_TOKENS;
#elif defined(GARNET_METAL_QWEN38_LARGE_PREFILL)
static constexpr size_t kSchedulerPrefillChunkLimit=256;
#else
static constexpr size_t kSchedulerPrefillChunkLimit=64;
#endif
class MetalContinuousScheduler::RowAllocator final:public KVSequenceAllocator {
    struct Sequence {int64_t row=0,epoch=0;int length=0,cached=0;
#ifdef GARNET_METAL_TOKEN_PREFIX_CACHE
        X::Value prompt;
#endif
    };
    MetalDecoderSession* session_;
    size_t rows_,capacity_;
    std::map<uint64_t,Sequence> sequences_;
#ifdef GARNET_METAL_TOKEN_PREFIX_CACHE
    std::map<uint64_t,X::Value> pendingTokens_;
#endif
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
#ifdef GARNET_METAL_TOKEN_PREFIX_CACHE
            auto pending=pendingTokens_.find(id);
            if(pending!=pendingTokens_.end()) {
                entry->second.prompt=pending->second;
                entry->second.cached=int(session_->RestoreTokenPrefix(entry->second.row,entry->second.epoch,pending->second));
                entry->second.length=entry->second.cached;pendingTokens_.erase(pending);
            }
#endif
        }catch(...){
            if(entry->second.epoch>0)try{session_->ReleaseRequestRow(entry->second.row,entry->second.epoch);}catch(...){}
            sequences_.erase(entry);throw;
        }
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
#ifdef GARNET_METAL_TOKEN_PREFIX_CACHE
        pendingTokens_.erase(id);
#endif
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
    int RestoredPrefixTokens(uint64_t id) const override {
        auto found=sequences_.find(id);if(found==sequences_.end())return 0;
        Validate(found->second);return found->second.cached;
    }
    bool PrefixPredictionReady(uint64_t id) const override {return RestoredPrefixTokens(id)>0;}
#ifdef GARNET_METAL_TOKEN_PREFIX_CACHE
    void Track(uint64_t id,X::Value tokens) {
        if(!pendingTokens_.emplace(id,std::move(tokens)).second)throw std::logic_error("Duplicate tracked request identity");
    }
    int64_t PromptToken(uint64_t id,int position) const {
        auto found=sequences_.find(id);if(found==sequences_.end()||!found->second.prompt.IsList()||position<0||uint64_t(position)>=found->second.prompt.Size())throw std::invalid_argument("Grouped prefill token is outside its immutable native request");
        return found->second.prompt.Get(uint64_t(position)).ToInt64();
    }
    bool Tracked(uint64_t id) const {
        auto found=sequences_.find(id);return found!=sequences_.end()&&found->second.prompt.IsList();
    }
    bool ProgressValid(uint64_t id,int expected) const {
        if(!Tracked(id))return true;
        const auto& sequence=sequences_.at(id);
        return session_->TrackedTokenCount(sequence.row,sequence.epoch)==expected;
    }
    void Capture(uint64_t id,int expected,bool intermediate=false) {
        if(!Tracked(id))return;
        const auto& sequence=sequences_.at(id);
#ifndef GARNET_METAL_PREFIX_SHARED_BLOCKS
        if(intermediate&&(expected<64||(expected&(expected-1))))return;
#endif
#if GARNET_METAL_TOKEN_PREFIX_HOST_MIB > 512
#ifndef GARNET_METAL_PREFIX_SHARED_BLOCKS
        if(intermediate&&!TokenPrefixIntermediateUseful(size_t(expected),size_t(sequence.prompt.Size())))return;
#endif
        // Protect the original prompt before a decoded snapshot can evict it,
        // then leave it recent for a returning request. Other live rows retain
        // their own device KV; this only adjusts the bounded host-cache LRU.
        if(!intermediate)session_->TouchTokenPrefix(sequence.row,sequence.epoch,sequence.prompt);
#endif
        // Exact head-only hits retain their checkpoint. Budget rejection is a
        // miss policy, not a generation failure.
        if(expected!=sequence.cached)session_->CacheTokenPrefix(sequence.row,sequence.epoch);
#if GARNET_METAL_TOKEN_PREFIX_HOST_MIB > 512
        if(!intermediate)session_->TouchTokenPrefix(sequence.row,sequence.epoch,sequence.prompt);
#endif
    }
#endif
    bool Clear() noexcept {
        bool result=true;
        for(const auto& entry:sequences_) {
            try {result=session_->ReleaseRequestRow(entry.second.row,entry.second.epoch)&&result;}
            catch(...){result=false;}
        }
        sequences_.clear();
#ifdef GARNET_METAL_TOKEN_PREFIX_CACHE
        pendingTokens_.clear();
#endif
        return result;
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
#ifdef GARNET_METAL_WIDE_OWNER_PREFILL
    const bool adaptive=session->MaximumPreparedPrefillChunk()>128;
    if(adaptive)config.prefillChunkSelector=[this,session](const BatchPlan& plan,int budget) {
        std::vector<size_t> rows,remaining;std::vector<uint64_t> epochs;
        for(const auto& request:plan.requests) {
            if(request.requestRow<0||request.prefillOffset<0||request.prefillOffset>request.promptTokens)
                throw std::logic_error("Invalid prepared prefill request geometry");
            rows.push_back(size_t(request.requestRow));epochs.push_back(request.rowEpoch);
            remaining.push_back(size_t(request.promptTokens-request.prefillOffset));
        }
        return int(session->SelectPreparedPrefillChunk(rows,epochs,remaining,maximumPrefillChunk_,size_t(budget)));
    };
#endif
    auto policy=std::make_unique<ContinuousBatchScheduler>(config);
    auto allocator=std::make_unique<RowAllocator>(session,layout.first,layout.second);
    sessionOwner_=value;rows_=layout.first;capacity_=layout.second;
#ifdef GARNET_METAL_WIDE_OWNER_PREFILL
    adaptiveProfiles_=adaptive;maximumPrefillChunk_=std::min(size_t(128),capacity_);
#endif
    scheduler_=std::move(policy);allocator_=std::move(allocator);failed_=false;return true;
}
#if defined(GARNET_METAL_DIAGNOSTIC_LAYER_BULK) && defined(GARNET_METAL_TOKEN_PREFIX_CACHE)
namespace {
std::atomic<uint64_t> nextBulkTicket{1};
uint64_t BulkTicket(){auto n=nextBulkTicket.load();for(;;){if(n>=uint64_t(INT64_MAX))throw std::overflow_error("Native grouped plan tickets exhausted");if(nextBulkTicket.compare_exchange_weak(n,n+1))return n;}}
}
bool MetalContinuousScheduler::GroupedRequest(uint64_t id) const {return std::any_of(groupedRequests_.begin(),groupedRequests_.end(),[&](const auto& r){return r.id==id;});}
void MetalContinuousScheduler::EraseGroupedRequest(uint64_t id) {groupedRequests_.erase(std::remove_if(groupedRequests_.begin(),groupedRequests_.end(),[&](const auto& r){return r.id==id;}),groupedRequests_.end());if(groupedRequests_.empty()){groupedChunks_=0;groupedTicket_=0;groupedExecuted_=false;}}
bool MetalContinuousScheduler::SetBulkPrefill(int64_t chunks) {
    std::lock_guard<std::mutex> lock(mutex_);Ready();
    auto session=sessionOwner_.NativeData<MetalDecoderSession>();
    if(rows_!=4||chunks<2||chunks>16||chunks>session->DiagnosticBulkChunkBound()||!inflight_.empty()||scheduler_->TrackedRequestCount())return false;
    if(!scheduler_->SetPrefillGrouping(128,int(chunks),int(rows_*128*size_t(chunks))))return false;
    maximumGroupedChunks_=int(chunks);return true;
}
X::Value MetalContinuousScheduler::RunBulkPrefill(X::Value hiddenChunks,int64_t ticket) {
    std::lock_guard<std::mutex> lock(mutex_);Ready();
    if(ticket<=0||uint64_t(ticket)!=groupedTicket_||groupedExecuted_||groupedChunks_<2||groupedRequests_.size()!=rows_||
       hiddenChunks.runtime()!=Host()->runtime||!hiddenChunks.IsList()||hiddenChunks.Size()!=uint64_t(groupedChunks_))throw std::invalid_argument("Run only the current native grouped prefill ticket once");
    for(uint64_t c=0;c<hiddenChunks.Size();++c){auto value=hiddenChunks.Get(c);if(value.runtime()!=Host()->runtime)throw std::invalid_argument("Grouped hidden runtime differs");X::Tensor t(value);auto lease=t.Acquire();auto info=t.Info();std::vector<int64_t> shape{4,128,4096};if(info.dtype!=X3_TENSOR_FLOAT32||info.symbolic||info.device_type||!info.data||info.rank!=3||!std::equal(shape.begin(),shape.end(),info.shape)||info.byte_size!=4*128*4096*4||!IsContiguousTensor(t))throw std::invalid_argument("Grouped prefill hidden profile differs");}
    std::vector<const ScheduledRequest*> byRow(rows_,nullptr);auto epochs=X::Value::List(Host());
    for(const auto& request:groupedRequests_){auto found=inflight_.find(request.id);auto placement=allocator_->Placement(request.id);
        if(request.requestRow<0||size_t(request.requestRow)>=rows_||byRow[size_t(request.requestRow)]||found==inflight_.end()||found->second!=RequestStage::Prefill||!allocator_->Tracked(request.id)||placement.requestRow!=request.requestRow||placement.rowEpoch!=request.rowEpoch||allocator_->LogicalLength(request.id)!=request.prefillOffset||request.prefillTokens!=groupedChunks_*128||trackedExpected_.at(request.id)!=request.contextLength)throw std::logic_error("Current grouped native request/lease/progress differs");byRow[size_t(request.requestRow)]=&request;
    }
    for(size_t row=0;row<rows_;++row){if(!byRow[row])throw std::logic_error("Grouped native physical row missing");epochs.Append(X::Value(int64_t(byRow[row]->rowEpoch)));}
    auto controls=X::Value::List(Host()),tokens=X::Value::List(Host());
    for(int c=0;c<groupedChunks_;++c){std::vector<float> positions,slots,active(512,1.f);positions.reserve(1536);slots.reserve(512);auto ids=X::Value::List(Host());
        for(size_t row=0;row<rows_;++row)for(int t=0;t<128;++t){int position=byRow[row]->prefillOffset+c*128+t;slots.push_back(float(position));for(int axis=0;axis<3;++axis)positions.push_back(float(position));ids.Append(X::Value(allocator_->PromptToken(byRow[row]->id,position)));}
        auto bank=X::Value::List(Host());bank.Append(X::Tensor::Create(Host(),X3_TENSOR_FLOAT32,{4,128,3},positions.data(),positions.size()*4));bank.Append(X::Tensor::Create(Host(),X3_TENSOR_FLOAT32,{4,128},slots.data(),slots.size()*4));bank.Append(X::Tensor::Create(Host(),X3_TENSOR_FLOAT32,{4,128},active.data(),active.size()*4));controls.Append(bank);tokens.Append(ids);
    }
    try {auto result=sessionOwner_.NativeData<MetalDecoderSession>()->RunDiagnosticLeasedBulk(hiddenChunks,controls,epochs,tokens,true);groupedExecuted_=true;return result;}
    catch(...){failed_=true;throw;}
}
#endif
bool MetalContinuousScheduler::SetPrefillCompletionPriority(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);Ready();
    if(!inflight_.empty())return false;
    return scheduler_->SetPrefillCompletionPriority(enabled);
}
#ifdef GARNET_METAL_WIDE_OWNER_PREFILL
bool MetalContinuousScheduler::SetPrefillCompletionOwners(int64_t owners) {
    std::lock_guard<std::mutex> lock(mutex_);Ready();
    if(!inflight_.empty()||!adaptiveProfiles_||rows_!=4||(owners!=1&&owners!=2&&owners!=4))return false;
    return scheduler_->SetPrefillCompletionOwners(int(owners));
}
#endif
bool MetalContinuousScheduler::SetPrefillChunk(int64_t tokens) {
    std::lock_guard<std::mutex> lock(mutex_);Ready();
    if(tokens<0||uint64_t(tokens)>kSchedulerPrefillChunkLimit||uint64_t(tokens)>capacity_||!inflight_.empty())return false;
#ifdef GARNET_METAL_DENSE_PREFILL_MAX_TOKENS
#ifdef GARNET_METAL_WIDE_OWNER_PREFILL
    if(adaptiveProfiles_) {
        auto session=sessionOwner_.NativeData<MetalDecoderSession>();
        if(tokens<2||(uint64_t(tokens)&(uint64_t(tokens)-1))||uint64_t(tokens)>session->MaximumPreparedPrefillChunk())return false;
        if(!scheduler_->SetPrefillChunkTokens(int(std::min(tokens,int64_t(128)))))return false;
        maximumPrefillChunk_=size_t(tokens);return true;
    }
#endif
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
    bool tokenBound=tokens>0&&uint64_t(tokens)<=kSchedulerPrefillChunkLimit;
#if defined(GARNET_METAL_DIAGNOSTIC_LAYER_BULK) && defined(GARNET_METAL_TOKEN_PREFIX_CACHE)
    if(GroupedRequest(uint64_t(id)))tokenBound=groupedExecuted_&&tokens==groupedChunks_*128;
#endif
    if(id<=0||!tokenBound||found==inflight_.end()||found->second!=RequestStage::Prefill)return false;
    try {
#ifdef GARNET_METAL_TOKEN_PREFIX_CACHE
        const auto expected=trackedExpected_.find(uint64_t(id));
        if(expected!=trackedExpected_.end()&&!allocator_->ProgressValid(uint64_t(id),expected->second))return false;
#endif
        if(!scheduler_->CompletePrefillChunk(uint64_t(id),int(tokens),*allocator_))return false;
#ifdef GARNET_METAL_TOKEN_PREFIX_CACHE
        if(expected!=trackedExpected_.end()){allocator_->Capture(uint64_t(id),expected->second,true);trackedExpected_.erase(expected);}
#endif
        inflight_.erase(found);
#if defined(GARNET_METAL_DIAGNOSTIC_LAYER_BULK) && defined(GARNET_METAL_TOKEN_PREFIX_CACHE)
        EraseGroupedRequest(uint64_t(id));
#endif
        return true;
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
#ifdef GARNET_METAL_TOKEN_PREFIX_CACHE
int64_t MetalContinuousScheduler::SubmitTokens(X::Value tokens,int64_t output,int64_t priority) {
    std::lock_guard<std::mutex> lock(mutex_);Ready();
    if(tokens.runtime()!=Host()->runtime||!tokens.IsList()||!tokens.Size()||tokens.Size()>capacity_||
       output<1||uint64_t(output)>capacity_-tokens.Size()||priority<INT_MIN||priority>INT_MAX)
        throw std::invalid_argument("Tracked scheduler request exceeds its token/context bounds");
    auto immutable=X::Value::List(Host());
    for(uint64_t i=0;i<tokens.Size();++i) {
        auto token=tokens.Get(i);
        if(!token.IsInt64()||token.ToInt64()<0||token.ToInt64()>INT_MAX)
            throw std::invalid_argument("Tracked scheduler requires integer token IDs");
        immutable.Append(X::Value(token.ToInt64()));
    }
    auto session=sessionOwner_.NativeData<MetalDecoderSession>();
    session->TokenPrefixStatsJson(); // Configuration must precede queue admission.
    GenerationRequest request;request.promptTokens=int(tokens.Size());request.maxNewTokens=int(output);request.priority=int(priority);
    const auto id=scheduler_->Submit(request);if(!id)return 0;
    try{allocator_->Track(id,std::move(immutable));}
    catch(...){scheduler_->Cancel(id,*allocator_);scheduler_->ForgetFinished(id);throw;}
    if(id>uint64_t(INT64_MAX)){failed_=true;throw std::overflow_error("Native scheduler request identities exhausted");}
    return int64_t(id);
}
#endif
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
                    {"prefill_offset",request.prefillOffset},{"prefill_tokens",request.prefillTokens},
                    {"cached_tokens",request.cachedTokens}});
                if(!inflight_.emplace(request.id,batch.stage).second)throw std::logic_error("Duplicate request in native scheduler tick");
#ifdef GARNET_METAL_TOKEN_PREFIX_CACHE
                if(allocator_->Tracked(request.id))trackedExpected_[request.id]=request.contextLength;
#endif
            }
            nlohmann::json body={{"stage",batch.stage==RequestStage::Prefill?"prefill":"decode"},
                {"bucket_size",rows_},{"active_rows",mask},{"epochs",epochs},{"requests",requests}};
            if(batch.prefillChunkTokens)body["prefill_chunk"]=batch.prefillChunkTokens;
#if defined(GARNET_METAL_DIAGNOSTIC_LAYER_BULK) && defined(GARNET_METAL_TOKEN_PREFIX_CACHE)
            if(batch.prefillGroupChunks>1){
                if(batch.stage!=RequestStage::Prefill||maximumGroupedChunks_<batch.prefillGroupChunks||batch.requests.size()!=rows_||!groupedRequests_.empty())throw std::logic_error("Invalid native grouped scheduler plan");
                for(const auto& r:batch.requests)if(!allocator_->Tracked(r.id))throw std::logic_error("Grouped scheduler requests must use native tracked token admission");
                groupedRequests_=batch.requests;groupedChunks_=batch.prefillGroupChunks;groupedExecuted_=false;groupedTicket_=BulkTicket();
                body["prefill_group_chunks"]=groupedChunks_;body["bulk_ticket"]=groupedTicket_;body["execution"]="native_leased_bulk";
            }
#endif
            batches.push_back(std::move(body));
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
#if defined(GARNET_METAL_DIAGNOSTIC_LAYER_BULK) && defined(GARNET_METAL_TOKEN_PREFIX_CACHE)
    if(GroupedRequest(uint64_t(id))&&!groupedExecuted_)return -1;
#endif
    try {
#ifdef GARNET_METAL_TOKEN_PREFIX_CACHE
        const auto expected=trackedExpected_.find(uint64_t(id));
        if(expected!=trackedExpected_.end()) {
            if(!allocator_->ProgressValid(uint64_t(id),expected->second))return -1;
            allocator_->Capture(uint64_t(id),expected->second);
        }
#endif
        if(!scheduler_->CompletePrefill(uint64_t(id),int(token),*allocator_,finished))return -1;
#ifdef GARNET_METAL_TOKEN_PREFIX_CACHE
        trackedExpected_.erase(uint64_t(id));
#endif
        const auto stage=scheduler_->Stage(uint64_t(id));inflight_.erase(found);
#if defined(GARNET_METAL_DIAGNOSTIC_LAYER_BULK) && defined(GARNET_METAL_TOKEN_PREFIX_CACHE)
        EraseGroupedRequest(uint64_t(id));
#endif
        if(stage==RequestStage::Finished)scheduler_->ForgetFinished(uint64_t(id));
        return int64_t(stage);
    }catch(...){failed_=true;throw;}
}
int64_t MetalContinuousScheduler::CompleteDecode(int64_t id,int64_t token,bool finished) {
    std::lock_guard<std::mutex> lock(mutex_);Ready();
    auto found=inflight_.find(uint64_t(id));
    if(id<=0||token<0||token>INT_MAX||found==inflight_.end()||found->second!=RequestStage::Decode)return -1;
    try {
#ifdef GARNET_METAL_TOKEN_PREFIX_CACHE
        const auto expected=trackedExpected_.find(uint64_t(id));
        if(expected!=trackedExpected_.end()) {
            if(!allocator_->ProgressValid(uint64_t(id),expected->second))return -1;
            int known=0,remaining=0;
            if(!scheduler_->DecodeProgress(uint64_t(id),known,remaining))return -1;
            if(finished||remaining==1)allocator_->Capture(uint64_t(id),expected->second);
        }
#endif
        if(!scheduler_->CompleteDecode(uint64_t(id),int(token),finished,*allocator_))return -1;
#ifdef GARNET_METAL_TOKEN_PREFIX_CACHE
        trackedExpected_.erase(uint64_t(id));
#endif
        const auto stage=scheduler_->Stage(uint64_t(id));inflight_.erase(found);
#if defined(GARNET_METAL_DIAGNOSTIC_LAYER_BULK) && defined(GARNET_METAL_TOKEN_PREFIX_CACHE)
        EraseGroupedRequest(uint64_t(id));
#endif
        if(stage==RequestStage::Finished)scheduler_->ForgetFinished(uint64_t(id));
        return int64_t(stage);
    }catch(...){failed_=true;throw;}
}

#ifdef GARNET_METAL_LOOKAHEAD_RUNTIME
std::string MetalContinuousScheduler::GenerateLookahead(X::Value stops) {
    std::lock_guard<std::mutex> lock(mutex_);Ready();
    auto session=sessionOwner_.NativeData<MetalDecoderSession>();
    struct Item {uint64_t id;size_t row;uint64_t epoch;int known,budget,slot;std::vector<int> emitted;bool eos=false;};
    std::vector<Item> items;
    std::vector<int64_t> roots(rows_,0),budgets(rows_,0),leases(rows_,0),positions(rows_,0);
    for(const auto& entry:inflight_)if(entry.second==RequestStage::Decode) {
        int known=0,budget=0;
        if(!scheduler_->DecodeProgress(entry.first,known,budget))throw std::logic_error("Lookahead scheduled decode progress invalid");
        const auto placement=allocator_->Placement(entry.first);
        const auto expected=trackedExpected_.find(entry.first);
        const int slot=allocator_->LogicalLength(entry.first);
        if(!allocator_->Tracked(entry.first)||expected==trackedExpected_.end()||expected->second!=slot+1||
           slot<0||placement.requestRow<0||size_t(placement.requestRow)>=rows_||!placement.rowEpoch||
           placement.rowEpoch>uint64_t(INT64_MAX)||leases[size_t(placement.requestRow)])
            throw std::logic_error("Lookahead requires unique native tracked row placement");
        const size_t row=size_t(placement.requestRow);
        roots[row]=known;budgets[row]=budget;leases[row]=int64_t(placement.rowEpoch);positions[row]=slot;
        items.push_back({entry.first,row,placement.rowEpoch,known,budget,slot,{}});
        items.back().emitted.reserve(4);
    }
    if(items.empty())throw std::invalid_argument("Schedule a native lookahead decode batch first");
    auto list=[&](const std::vector<int64_t>& values){auto out=X::Value::List(Host());for(auto v:values)out.Append(X::Value(v));return out;};
    auto known=list(roots),slots=list(positions),remaining=list(budgets),epochs=list(leases);
    try {
        const auto result=session->LookaheadGenerate(known,slots,remaining,stops,epochs);
        const auto statistics=nlohmann::json::parse(session->LookaheadStatsJson());
        if(!result.IsList()||result.Size()!=rows_)throw std::logic_error("Lookahead result batch differs");
        // Validate every native accepted span before publishing scheduler progress
        // or releasing a terminal physical row for the next request.
        for(auto& item:items) {
            auto row=result.Get(uint64_t(item.row));
            if(!row.IsList()||row.Size()!=3)throw std::logic_error("Lookahead result tuple differs");
            auto ids=row.Get(uint64_t(0)),countValue=row.Get(uint64_t(1)),eos=row.Get(uint64_t(2));
            if(!countValue.IsInt64())throw std::logic_error("Lookahead consumed count is not integral");
            const auto count=countValue.ToInt64();
            if(!ids.IsList()||count<1||count>4||count>item.budget||uint64_t(count)!=ids.Size()||eos.raw().tag!=X3_TAG_BOOL)
                throw std::logic_error("Lookahead accepted span exceeds native budget");
            for(uint64_t i=0;i<ids.Size();++i){auto v=ids.Get(i);if(!v.IsInt64()||v.ToInt64()<0||v.ToInt64()>INT_MAX)throw std::logic_error("Lookahead emitted invalid token");item.emitted.push_back(int(v.ToInt64()));}
            item.eos=eos.ToInt64()!=0;
            if(!allocator_->ProgressValid(item.id,item.slot+int(count)))throw std::logic_error("Lookahead committed ledger differs from accepted span");
        }
        nlohmann::json completed=nlohmann::json::array();
        for(const auto& item:items) {
            if(item.eos||item.emitted.size()==size_t(item.budget))allocator_->Capture(item.id,item.slot+int(item.emitted.size()));
            if(!scheduler_->CompleteDecodeTokens(item.id,item.emitted,item.eos,*allocator_))throw std::logic_error("Lookahead scheduler span commit failed");
            const auto stage=scheduler_->Stage(item.id);inflight_.erase(item.id);trackedExpected_.erase(item.id);
            completed.push_back({{"id",item.id},{"row",item.row},{"epoch",item.epoch},{"known",item.known},{"slot_before",item.slot},{"remaining_before",item.budget},{"emitted",item.emitted},{"consumed",item.emitted.size()},{"eos",item.eos},{"finished",stage==RequestStage::Finished}});
            if(stage==RequestStage::Finished)scheduler_->ForgetFinished(item.id);
        }
        return nlohmann::json{{"requests",completed},{"lookahead",statistics},{"scope","Native leased lookahead DAG and scheduler span commit; no speed admission"}}.dump();
    }catch(...){failed_=true;throw;}
}
#endif

#ifdef GARNET_METAL_DENSE_VERIFY_SCHEDULER
std::string MetalContinuousScheduler::VerifyDecode(X::Value hidden,X::Value positions,X::Value slots,X::Value active,X::Value tokens,X::Value stops) {
    std::lock_guard<std::mutex> lock(mutex_);Ready();
    auto session=sessionOwner_.NativeData<MetalDecoderSession>();
    const auto geometry=session->VerificationGeometry();const size_t chunk=geometry[1];
    if(geometry[0]!=rows_||!chunk||chunk>4||
#ifdef GARNET_METAL_WIDE_DENSE_VERIFICATION
       (rows_*chunk>4&&!(rows_==4&&chunk==4))
#else
       rows_*chunk>4
#endif
       )
        throw std::invalid_argument("Scheduled verification exceeds prepared physical token rows");
    // Own immutable control snapshots before validating the native schedule.
    auto control=[&](X::Value source,const std::vector<int64_t>& shape,std::vector<float>& values){
        if(source.runtime()!=Host()->runtime)throw std::invalid_argument("Scheduled control runtime differs");
        X::Tensor tensor(source);auto lease=tensor.Acquire();const auto info=tensor.Info();size_t count=1;for(auto dim:shape)count*=size_t(dim);
        if(info.symbolic||info.device_type||info.dtype!=X3_TENSOR_FLOAT32||info.rank!=shape.size()||!std::equal(shape.begin(),shape.end(),info.shape)||!info.data||info.byte_size<count*4||!IsContiguousTensor(tensor))
            throw std::invalid_argument("Scheduled verification control shape/storage differs");
        const auto data=static_cast<const float*>(info.data);values.assign(data,data+count);
        return X::Tensor::Create(Host(),X3_TENSOR_FLOAT32,shape,values.data(),count*4);
    };
    std::vector<float> positionValues,slotValues,flags;
    positions=control(positions,{int64_t(rows_),int64_t(chunk),3},positionValues);
    slots=control(slots,{int64_t(rows_),int64_t(chunk)},slotValues);
    active=control(active,{int64_t(rows_),int64_t(chunk)},flags);
    if(tokens.runtime()!=Host()->runtime||!tokens.IsList()||tokens.Size()!=rows_*chunk)
        throw std::invalid_argument("Scheduled verification token count/runtime differs");
    auto immutableTokens=X::Value::List(Host());std::vector<int64_t> tokenValues;
    for(uint64_t i=0;i<tokens.Size();++i){auto value=tokens.Get(i);if(!value.IsInt64()||value.ToInt64()<0||value.ToInt64()>INT_MAX)throw std::invalid_argument("Scheduled verification tokens must be bounded integers");tokenValues.push_back(value.ToInt64());immutableTokens.Append(X::Value(value.ToInt64()));}
    struct Item{uint64_t id;size_t row;uint64_t epoch;int known,budget,slot,depth;std::vector<int> emitted;bool eos=false;};
    std::vector<Item> items;std::vector<int> budgets(rows_,0);std::vector<uint64_t> leases(rows_,0);
    for(const auto& entry:inflight_)if(entry.second==RequestStage::Decode){
        int known=0,budget=0;if(!scheduler_->DecodeProgress(entry.first,known,budget))throw std::logic_error("Scheduled verification decode progress invalid");
        const auto placement=allocator_->Placement(entry.first);const auto expected=trackedExpected_.find(entry.first);const int slot=allocator_->LogicalLength(entry.first);
        if(!allocator_->Tracked(entry.first)||expected==trackedExpected_.end()||expected->second!=slot+1||slot<0||placement.requestRow<0||size_t(placement.requestRow)>=rows_||!placement.rowEpoch||leases[size_t(placement.requestRow)])
            throw std::logic_error("Scheduled verification requires native tracked row placement");
        const auto row=size_t(placement.requestRow);bool ended=false;int depth=0;
        for(size_t t=0;t<chunk;++t){const auto i=row*chunk+t;const float flag=flags[i];if(flag!=0.0f&&flag!=1.0f)throw std::invalid_argument("Scheduled verification masks must be binary");if(!flag){ended=true;continue;}
            if(ended||slotValues[i]!=float(slot+int(t))||slot+int(t)>=int(capacity_))throw std::invalid_argument("Scheduled verification mask/slot differs from native request");
            for(size_t axis=0;axis<3;++axis)if(positionValues[i*3+axis]!=float(slot+int(t)))throw std::invalid_argument("Scheduled text verification position differs from zero-based native slot");++depth;
        }
        if(depth<1||depth>budget||tokenValues[row*chunk]!=known)throw std::invalid_argument("Scheduled verification known token or depth exceeds native output budget");
        budgets[row]=budget;leases[row]=placement.rowEpoch;items.push_back({entry.first,row,placement.rowEpoch,known,budget,slot,depth,{}});
    }
    if(items.empty())throw std::invalid_argument("Schedule a native dense decode batch first");
    for(size_t row=0;row<rows_;++row)if(!leases[row])for(size_t t=0;t<chunk;++t)if(flags[row*chunk+t]!=0.0f)throw std::invalid_argument("Unscheduled verification row is active");
    auto remaining=X::Value::List(Host()),epochs=X::Value::List(Host());for(size_t row=0;row<rows_;++row){remaining.Append(X::Value(int64_t(budgets[row])));epochs.Append(X::Value(int64_t(leases[row])));}
    try {
        auto result=session->VerifyGreedy(hidden,positions,slots,active,immutableTokens,stops,remaining,epochs);
        const auto statistics=nlohmann::json::parse(session->VerificationStatsJson());
        // Prove every accepted target span and token-ledger length before any
        // scheduler state is published or a terminal request lease released.
        for(auto& item:items){auto row=result.Get(uint64_t(item.row)),ids=row.Get(uint64_t(0));const auto count=row.Get(uint64_t(1)).ToInt64();auto eos=row.Get(uint64_t(2));
            if(!ids.IsList()||count<1||count>item.depth||count>item.budget||uint64_t(count)!=ids.Size()||eos.raw().tag!=X3_TAG_BOOL)throw std::logic_error("Scheduled target result exceeds accepted budget");
            for(uint64_t i=0;i<ids.Size();++i){auto value=ids.Get(i);if(!value.IsInt64()||value.ToInt64()<0||value.ToInt64()>INT_MAX)throw std::logic_error("Scheduled target result token invalid");item.emitted.push_back(int(value.ToInt64()));}item.eos=eos.ToInt64()!=0;
            if(!allocator_->ProgressValid(item.id,item.slot+int(count)))throw std::logic_error("Scheduled verification accepted token ledger differs");
        }
        nlohmann::json completed=nlohmann::json::array();
        for(const auto& item:items){
            if(item.eos||item.emitted.size()==size_t(item.budget))allocator_->Capture(item.id,item.slot+int(item.emitted.size()));
            if(!scheduler_->CompleteDecodeTokens(item.id,item.emitted,item.eos,*allocator_))throw std::logic_error("Scheduled dense target span commit failed");
            const auto stage=scheduler_->Stage(item.id);inflight_.erase(item.id);trackedExpected_.erase(item.id);
            completed.push_back({{"id",item.id},{"row",item.row},{"epoch",item.epoch},{"known",item.known},{"slot_before",item.slot},{"remaining_before",item.budget},{"submitted",item.depth},{"emitted",item.emitted},{"consumed",item.emitted.size()},{"eos",item.eos},{"finished",stage==RequestStage::Finished}});
            if(stage==RequestStage::Finished)scheduler_->ForgetFinished(item.id);
        }
        return nlohmann::json{{"requests",completed},{"verification",statistics},{"scope","Native leased dense target verification and exact scheduler span commit; no serving speed admission"}}.dump();
    }catch(...){failed_=true;throw;}
}
#endif

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
#if defined(GARNET_METAL_DIAGNOSTIC_LAYER_BULK) && defined(GARNET_METAL_TOKEN_PREFIX_CACHE)
        const bool abandon=GroupedRequest(uint64_t(id))&&!groupedExecuted_;
#endif
        if(!scheduler_->Cancel(uint64_t(id),*allocator_))return false;
#ifdef GARNET_METAL_TOKEN_PREFIX_CACHE
        trackedExpected_.erase(uint64_t(id));
#endif
        inflight_.erase(uint64_t(id));scheduler_->ForgetFinished(uint64_t(id));
#if defined(GARNET_METAL_DIAGNOSTIC_LAYER_BULK) && defined(GARNET_METAL_TOKEN_PREFIX_CACHE)
        if(abandon){for(const auto& r:groupedRequests_)if(r.id!=uint64_t(id)){if(!scheduler_->AbandonPrefillChunk(r.id))throw std::logic_error("Cannot abandon an unexecuted grouped survivor");inflight_.erase(r.id);trackedExpected_.erase(r.id);}groupedRequests_.clear();groupedChunks_=0;groupedTicket_=0;groupedExecuted_=false;}
        else EraseGroupedRequest(uint64_t(id));
#endif
        return true;
    }catch(...){failed_=true;throw;}
}
std::string MetalContinuousScheduler::StatusJson() {
    std::lock_guard<std::mutex> lock(mutex_);Ready();
    auto status=nlohmann::json{{"active",scheduler_->ActiveRequestCount()},{"queued",scheduler_->QueuedRequestCount()},
        {"tracked",scheduler_->TrackedRequestCount()},{"inflight",inflight_.size()},{"rows",rows_},{"capacity",capacity_},{"prefill_completion_priority",scheduler_->PrefillCompletionPriority()}};
#ifdef GARNET_METAL_WIDE_OWNER_PREFILL
    status["prefill_completion_owners"]=scheduler_->PrefillCompletionOwners();
#endif
    return status.dump();
}
bool MetalContinuousScheduler::CloseUnlocked() {
    bool result=!allocator_||allocator_->Clear();
    inflight_.clear();scheduler_.reset();allocator_.reset();sessionOwner_=X::Value();
#if defined(GARNET_METAL_DIAGNOSTIC_LAYER_BULK) && defined(GARNET_METAL_TOKEN_PREFIX_CACHE)
    groupedRequests_.clear();groupedChunks_=maximumGroupedChunks_=0;groupedTicket_=0;groupedExecuted_=false;
#endif
#ifdef GARNET_METAL_TOKEN_PREFIX_CACHE
    trackedExpected_.clear();
#endif
    rows_=capacity_=0;failed_=false;return result;
}
bool MetalContinuousScheduler::Close(){std::lock_guard<std::mutex> lock(mutex_);return CloseUnlocked();}
}
