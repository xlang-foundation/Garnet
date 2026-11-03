// SPDX-License-Identifier: Apache-2.0
#include "continuous_scheduler.h"
#include "nlohmann/json.hpp"
#include <limits>
#include <climits>
namespace Garnet {
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
    ContinuousBatchSchedulerConfig config;
    config.decodeBuckets={int(layout.first)};config.maxPrefillBatch=int(layout.first);
    config.maxPrefillTokensPerTick=int(layout.first*layout.second);config.maxActiveRequests=int(pending);
    auto policy=std::make_unique<ContinuousBatchScheduler>(config);
    auto allocator=std::make_unique<RowAllocator>(session,layout.first,layout.second);
    sessionOwner_=value;rows_=layout.first;capacity_=layout.second;
    scheduler_=std::move(policy);allocator_=std::move(allocator);failed_=false;return true;
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
        for(const auto& batch:tick.batches) {
            std::vector<int> mask(rows_,0);std::vector<uint64_t> epochs(rows_,0);
            nlohmann::json requests=nlohmann::json::array();
            for(const auto& request:batch.requests) {
                if(request.requestRow<0||size_t(request.requestRow)>=rows_||!request.rowEpoch||mask[size_t(request.requestRow)])
                    throw std::logic_error("Native scheduler supplied an invalid physical row plan");
                const auto row=size_t(request.requestRow);mask[row]=1;epochs[row]=request.rowEpoch;
                requests.push_back({{"id",request.id},{"row",row},{"epoch",request.rowEpoch},{"token_id",request.tokenId},
                    {"slot",request.slotPosition},{"context_length",request.contextLength},{"prompt_tokens",request.promptTokens},
                    {"generated_tokens",request.generatedTokens},{"position_delta",request.positionDelta}});
                if(!inflight_.emplace(request.id,batch.stage).second)throw std::logic_error("Duplicate request in native scheduler tick");
            }
            batches.push_back({{"stage",batch.stage==RequestStage::Prefill?"prefill":"decode"},
                {"bucket_size",rows_},{"active_rows",mask},{"epochs",epochs},{"requests",requests}});
        }
        return nlohmann::json{{"batches",batches},{"scope","Native policy and Metal row state; dispatch via run_leased"}}.dump();
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
