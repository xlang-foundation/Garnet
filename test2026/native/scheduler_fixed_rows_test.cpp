// SPDX-License-Identifier: Apache-2.0
#include "../../src/runtime/scheduler/continuous_batch_scheduler.h"
#include "../../src/runtime/scheduler/request_row_leases.h"
#include "../../src/runtime/scheduler/packed_token_plan.h"
#include "../../src/runtime/kv/paged_kv_pool.h"
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
using namespace Garnet;
static_assert(std::is_base_of<KVSequenceAllocator,PagedKVPool>::value,"CUDA pool must retain allocator compatibility");
void Check(bool value){if(!value)throw std::runtime_error("Portable scheduler row invariant failed");}
// Test-only metadata allocator. Actual Metal admission must bind its claims to
// the native session's state-clearing callbacks; this fixture allocates no GPU KV.
class FixedRows final:public KVSequenceAllocator {
    struct State{RequestRowLease lease;int length=0;};
    RequestRowLeases rows_{4};
    std::unordered_map<uint64_t,State> states_;
public:
    bool Reserve(uint64_t id,int capacity) override {
        if(!id||capacity<0||capacity>64||states_.size()==4||states_.count(id))return false;
        states_.emplace(id,State{rows_.Claim(id),0});return true;
    }
    bool EnsureCapacity(uint64_t id,int capacity) override{return states_.count(id)&&capacity>=0&&capacity<=64;}
    bool CommitLength(uint64_t id,int length) override {
        if(!EnsureCapacity(id,length))return false;states_.at(id).length=length;return true;
    }
    bool Release(uint64_t id) override {
        auto found=states_.find(id);if(found==states_.end())return false;
        Check(rows_.Release(found->second.lease.row,found->second.lease.epoch));states_.erase(found);return true;
    }
    int LogicalLength(uint64_t id) const override {auto it=states_.find(id);return it==states_.end()?-1:it->second.length;}
    std::vector<int> PageTable(uint64_t) const override{return {};}
    KVSequencePlacement Placement(uint64_t id) const override {
        auto it=states_.find(id);return it==states_.end()?KVSequencePlacement{}:
            KVSequencePlacement{int(it->second.lease.row),it->second.lease.epoch};
    }
};
int main() {
    FixedRows storage;
    ContinuousBatchSchedulerConfig config;config.decodeBuckets={1,2,4};config.maxPrefillBatch=4;
    ContinuousBatchScheduler scheduler(config);
    std::vector<uint64_t> ids;
    for(int i=0;i<5;++i){GenerationRequest request;request.promptTokens=16+i;request.maxNewTokens=8;request.priority=i;ids.push_back(scheduler.Submit(request));Check(ids.back()!=0);}
    auto tick=scheduler.Schedule(storage);Check(tick.batches.size()==1&&tick.batches[0].requests.size()==4);
    std::vector<KVSequencePlacement> before;
    for(int i=0;i<4;++i){before.push_back(storage.Placement(ids[i]));Check(before.back().requestRow==i&&before.back().rowEpoch>0);Check(scheduler.CompletePrefill(ids[i],100+i,storage));}
    tick=scheduler.Schedule(storage);Check(tick.batches.size()==1&&tick.batches[0].requests.size()==4);
    Check(tick.batches[0].requests.front().id==ids[3]); // Priority changes plan order, never physical state rows.
    for(const auto& request:tick.batches[0].requests){auto p=storage.Placement(request.id);Check(request.requestRow==p.requestRow&&request.rowEpoch==p.rowEpoch);}
    Check(scheduler.CompleteDecode(ids[0],201,true,storage));
    tick=scheduler.Schedule(storage);Check(tick.batches.size()==2);
    Check(tick.batches[0].stage==RequestStage::Decode&&tick.batches[0].requests.size()==3);
    Check(tick.batches[1].stage==RequestStage::Prefill&&tick.batches[1].requests.size()==1);
    auto e=storage.Placement(ids[4]);Check(e.requestRow==0&&e.rowEpoch!=before[0].rowEpoch);
    for(int i=1;i<4;++i){auto p=storage.Placement(ids[i]);Check(p.requestRow==before[i].requestRow&&p.rowEpoch==before[i].rowEpoch);}
    Check(scheduler.CompletePrefill(ids[4],205,storage));
    tick=scheduler.Schedule(storage);Check(tick.batches[0].requests.size()==4);
    Check(scheduler.Cancel(ids[1],storage)&&!scheduler.CompleteDecode(ids[1],0,false,storage));
    Check(!scheduler.CompleteDecode(ids[0],0,false,storage));
    Check(!scheduler.CompleteDecode(ids[2],-1,false,storage));
    tick=scheduler.Schedule(storage);
    for(const auto& item:tick.batches[0].requests)if(item.id==ids[2])Check(item.generatedTokens==1);
    FixedRows singleStorage;ContinuousBatchScheduler single(config);
    GenerationRequest request;request.promptTokens=2;request.maxNewTokens=1;
    auto one=single.Submit(request);Check(one!=0);
    Check(single.Schedule(singleStorage).batches.size()==1);
    Check(single.CompletePrefill(one,7,singleStorage));
    Check(single.Stage(one)==RequestStage::Finished&&singleStorage.LogicalLength(one)==-1);
    Check(single.Schedule(singleStorage).batches.empty()&&!single.CompleteDecode(one,8,false,singleStorage));
    request.maxNewTokens=8;auto eos=single.Submit(request);
    Check(single.Schedule(singleStorage).batches.size()==1);
    Check(!single.CompletePrefill(eos,-1,singleStorage));
    Check(single.Stage(eos)==RequestStage::Prefill);
    Check(single.CompletePrefill(eos,9,singleStorage,true));
    Check(single.Stage(eos)==RequestStage::Finished&&single.Schedule(singleStorage).batches.empty());
    Check(!scheduler.ForgetFinished(ids[2]));
    Check(single.TrackedRequestCount()==2&&single.ForgetFinished(one)&&single.ForgetFinished(eos));
    Check(!single.ForgetFinished(eos)&&single.TrackedRequestCount()==0);
    FixedRows chunkStorage;
    ContinuousBatchSchedulerConfig chunkConfig;chunkConfig.decodeBuckets={1,2};chunkConfig.maxPrefillBatch=2;
    chunkConfig.maxPrefillTokensPerTick=8;
    ContinuousBatchScheduler chunked(chunkConfig);
    Check(!chunked.SetPrefillChunkTokens(-1)&&!chunked.SetPrefillChunkTokens(9)&&chunked.SetPrefillChunkTokens(4));
    GenerationRequest shortPrompt;shortPrompt.promptTokens=3;shortPrompt.maxNewTokens=10;
    GenerationRequest longPrompt;longPrompt.promptTokens=11;longPrompt.maxNewTokens=3;
    auto shortId=chunked.Submit(shortPrompt),longId=chunked.Submit(longPrompt);
    Check(!chunked.SetPrefillChunkTokens(0));
    tick=chunked.Schedule(chunkStorage);Check(tick.batches.size()==1&&tick.batches[0].scheduledTokens==7);
    Check(tick.batches[0].requests[0].prefillTokens==3&&tick.batches[0].requests[1].prefillTokens==4);
    const auto longLease=chunkStorage.Placement(longId);
    Check(!chunked.CompletePrefill(longId,9,chunkStorage)); // A partial prompt cannot generate yet.
    Check(!chunked.CompletePrefillChunk(longId,3,chunkStorage));
    Check(chunked.CompletePrefill(shortId,7,chunkStorage));
    Check(chunked.CompletePrefillChunk(longId,4,chunkStorage)&&chunkStorage.LogicalLength(longId)==4);
    Check(!chunked.CompletePrefillChunk(longId,4,chunkStorage)); // Duplicate completion.
    tick=chunked.Schedule(chunkStorage);Check(tick.batches.size()==2);
    Check(tick.batches[0].stage==RequestStage::Decode&&tick.batches[0].requests[0].id==shortId);
    const auto middle=tick.batches[1].requests[0];
    Check(middle.id==longId&&middle.prefillOffset==4&&middle.prefillTokens==4&&middle.slotPosition==4&&middle.contextLength==8);
    Check(middle.requestRow==longLease.requestRow&&middle.rowEpoch==longLease.rowEpoch);
    // Real scheduler output packs decode + ragged prompt work without padding.
    const auto packed=BuildPackedTokenPlan(tick,chunkStorage,5,64);
    Check(packed.tokenCount==5&&packed.segments.size()==2);
    Check(packed.segments[0].stage==RequestStage::Decode&&packed.segments[0].tokenCount==1);
    Check(packed.segments[1].packedOffset==1&&packed.segments[1].tokenCount==4&&
        packed.segments[1].absolutePosition==4);
    auto RejectPacked=[&](const SchedulerTick& candidate,int budget=5) {
        bool rejected=false;
        try{BuildPackedTokenPlan(candidate,chunkStorage,budget,64);}
        catch(const std::invalid_argument&){rejected=true;}
        Check(rejected);
    };
    RejectPacked(tick,4); // Total workspace budget, not per-request budget.
    auto corrupt=tick;corrupt.batches[1].requests[0].rowEpoch++;RejectPacked(corrupt);
    corrupt=tick;corrupt.batches[1].requests[0].prefillOffset++;RejectPacked(corrupt);
    corrupt=tick;auto& shifted=corrupt.batches[1].requests[0];
    shifted.prefillOffset++;shifted.slotPosition++;shifted.contextLength++;RejectPacked(corrupt);
    corrupt=tick;corrupt.batches.push_back(tick.batches[0]);RejectPacked(corrupt);
    corrupt=tick;corrupt.batches[1].requests[0].prefillTokens=0;RejectPacked(corrupt);
    corrupt=tick;corrupt.batches[1].stage=RequestStage::Cancelled;RejectPacked(corrupt);
    Check(chunked.CompleteDecode(shortId,8,false,chunkStorage));
    Check(chunked.CompletePrefillChunk(longId,4,chunkStorage));
    tick=chunked.Schedule(chunkStorage);Check(tick.batches.size()==2);
    const auto tail=tick.batches[1].requests[0];Check(tail.prefillOffset==8&&tail.prefillTokens==3&&tail.contextLength==11);
    Check(!chunked.CompletePrefillChunk(longId,3,chunkStorage)); // Final chunk requires a first token.
    Check(chunked.CompleteDecode(shortId,8,false,chunkStorage)&&chunked.CompletePrefill(longId,9,chunkStorage));
    Check(chunkStorage.LogicalLength(longId)==11);
    tick=chunked.Schedule(chunkStorage);Check(tick.batches.size()==1&&tick.batches[0].requests.size()==2);
    Check(chunked.Cancel(shortId,chunkStorage)&&chunked.Cancel(longId,chunkStorage));
    Check(chunked.ForgetFinished(shortId)&&chunked.ForgetFinished(longId));
    longId=chunked.Submit(longPrompt);tick=chunked.Schedule(chunkStorage);
    Check(chunked.CompletePrefillChunk(longId,4,chunkStorage));
    Check(chunked.Cancel(longId,chunkStorage)&&chunkStorage.LogicalLength(longId)==-1);
    Check(!chunked.CompletePrefillChunk(longId,4,chunkStorage)&&!chunked.CompletePrefill(longId,1,chunkStorage));
    Check(chunked.ForgetFinished(longId)&&chunked.SetPrefillChunkTokens(0));
    std::cout<<"portable_continuous_scheduler PASS stable rows, queued refill, cancellation, one-token budget, prefill EOS, invalid-token rejection and chunked prefill fairness/tail/lease lifecycle\n";
}
