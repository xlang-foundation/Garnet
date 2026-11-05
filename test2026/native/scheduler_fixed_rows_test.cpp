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
    struct State{RequestRowLease lease;int length=0,cached=0;bool ready=false;};
    RequestRowLeases rows_{4};
    std::unordered_map<uint64_t,State> states_;
    std::unordered_map<uint64_t,std::pair<int,bool>> restored_;
    int capacity_;
public:
    explicit FixedRows(int capacity=64):capacity_(capacity){}
    void SeedPrefix(uint64_t id,int tokens,bool ready=true){restored_[id]={tokens,ready};}
    bool Reserve(uint64_t id,int capacity) override {
        if(!id||capacity<0||capacity>capacity_||states_.size()==4||states_.count(id))return false;
        auto restored=restored_.count(id)?restored_.at(id):std::make_pair(0,false);
        states_.emplace(id,State{rows_.Claim(id),restored.first,restored.first,restored.second});return true;
    }
    bool EnsureCapacity(uint64_t id,int capacity) override{return states_.count(id)&&capacity>=0&&capacity<=capacity_;}
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
    int RestoredPrefixTokens(uint64_t id) const override {auto it=states_.find(id);return it==states_.end()?0:it->second.cached;}
    bool PrefixPredictionReady(uint64_t id) const override {auto it=states_.find(id);return it!=states_.end()&&it->second.ready;}
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
    // Actual planning bounds of the dense Mistral profile. This is metadata
    // scheduling, not model inference: five distinct jobs share four rows and
    // a512-token global workspace; the fifth must reuse the first row's epoch.
    FixedRows denseStorage(4096);
    ContinuousBatchSchedulerConfig denseConfig;denseConfig.decodeBuckets={4};denseConfig.maxPrefillBatch=4;
    denseConfig.maxPrefillTokensPerTick=512;denseConfig.prefillChunkTokens=128;
    ContinuousBatchScheduler dense(denseConfig);
    const int prompts[]={513,514,1025,2048,512},outputs[]={1,128,129,130,2};
    std::vector<uint64_t> denseIds;std::unordered_map<uint64_t,KVSequencePlacement> leases;
    for(size_t i=0;i<5;++i){GenerationRequest job;job.promptTokens=prompts[i];job.maxNewTokens=outputs[i];denseIds.push_back(dense.Submit(job));Check(denseIds.back()!=0);}
    bool refilled=false,mixedTick=false;size_t finished=0,ticks=0;
    while(finished<5) {
        Check(++ticks<1000);const auto current=dense.Schedule(denseStorage);Check(!current.batches.empty());
        const auto plan=BuildPackedTokenPlan(current,denseStorage,512,4096);Check(plan.tokenCount>0&&plan.tokenCount<=512);
        mixedTick=mixedTick||current.batches.size()==2;
        size_t prefillTokens=0;
        for(const auto& batch:current.batches)for(const auto& job:batch.requests) {
            const auto placement=denseStorage.Placement(job.id);auto found=leases.find(job.id);
            if(found==leases.end()) {
                leases.emplace(job.id,placement);
                if(job.id==denseIds[4]) {
                    Check(placement.requestRow==leases.at(denseIds[0]).requestRow&&placement.rowEpoch!=leases.at(denseIds[0]).rowEpoch);
                    refilled=true;
                }
            } else Check(placement.requestRow==found->second.requestRow&&placement.rowEpoch==found->second.rowEpoch);
            if(batch.stage==RequestStage::Prefill) {
                Check(job.prefillTokens>0&&job.prefillTokens<=128);prefillTokens+=size_t(job.prefillTokens);
                if(job.prefillOffset+job.prefillTokens==job.promptTokens)Check(dense.CompletePrefill(job.id,42,denseStorage));
                else Check(dense.CompletePrefillChunk(job.id,job.prefillTokens,denseStorage));
            } else {Check(batch.stage==RequestStage::Decode);Check(dense.CompleteDecode(job.id,42,false,denseStorage));}
        }
        Check(prefillTokens<=512);finished=0;
        for(auto id:denseIds)if(dense.Stage(id)==RequestStage::Finished){++finished;Check(denseStorage.LogicalLength(id)==-1);}
    }
    Check(refilled&&mixedTick&&dense.Schedule(denseStorage).batches.empty());
    for(auto id:denseIds)Check(dense.ForgetFinished(id));
    Check(dense.TrackedRequestCount()==0);
    FixedRows cachedStorage(4096);
    ContinuousBatchSchedulerConfig cachedConfig;cachedConfig.decodeBuckets={4};cachedConfig.maxPrefillBatch=4;
    cachedConfig.maxPrefillTokensPerTick=256;cachedConfig.prefillChunkTokens=64;
    ContinuousBatchScheduler cached(cachedConfig);
    std::vector<uint64_t> cachedIds;
    for(int prompt:{128,130,65,64}) {
        GenerationRequest job;job.promptTokens=prompt;job.maxNewTokens=4;
        cachedIds.push_back(cached.Submit(job));
    }
    cachedStorage.SeedPrefix(cachedIds[0],128);cachedStorage.SeedPrefix(cachedIds[1],64);
    cachedStorage.SeedPrefix(cachedIds[3],32);
    tick=cached.Schedule(cachedStorage);
    Check(tick.batches.size()==1&&tick.batches[0].scheduledTokens==160);
    const auto exactHead=tick.batches[0].requests[0];
    Check(exactHead.prefillTokens==0&&exactHead.prefillOffset==128&&exactHead.cachedTokens==128);
    const auto cachedPacked=BuildPackedTokenPlan(tick,cachedStorage,256,4096);
    Check(cachedPacked.tokenCount==160&&cachedPacked.segments.size()==4&&cachedPacked.segments[0].tokenCount==0);
    Check(cached.Schedule(cachedStorage).batches.empty()); // Includes the zero-token in-flight head.
    Check(!cached.CompletePrefillChunk(cachedIds[0],0,cachedStorage));
    Check(cached.CompletePrefill(cachedIds[0],42,cachedStorage));
    Check(!cached.CompletePrefill(cachedIds[0],42,cachedStorage));
    Check(cached.CompletePrefillChunk(cachedIds[1],64,cachedStorage));
    Check(cached.CompletePrefillChunk(cachedIds[2],64,cachedStorage));
    Check(cached.CompletePrefill(cachedIds[3],42,cachedStorage));
    tick=cached.Schedule(cachedStorage);
    Check(tick.batches.size()==2&&tick.batches[1].scheduledTokens==3);
    Check(BuildPackedTokenPlan(tick,cachedStorage,5,4096).tokenCount==5);
    Check(cached.CompleteDecode(cachedIds[0],43,true,cachedStorage));
    Check(cached.CompleteDecode(cachedIds[3],43,false,cachedStorage));
    Check(cached.CompletePrefill(cachedIds[1],43,cachedStorage));
    Check(cached.CompletePrefill(cachedIds[2],43,cachedStorage));
    GenerationRequest returned;returned.promptTokens=128;returned.maxNewTokens=1;
    auto returnedId=cached.Submit(returned);cachedStorage.SeedPrefix(returnedId,128);
    tick=cached.Schedule(cachedStorage);
    Check(tick.batches.size()==2&&tick.batches[1].scheduledTokens==0);
    const auto refill=tick.batches[1].requests[0];
    Check(refill.requestRow==exactHead.requestRow&&refill.rowEpoch!=exactHead.rowEpoch);
    Check(BuildPackedTokenPlan(tick,cachedStorage,3,4096).tokenCount==3);
    auto forged=tick;forged.batches[1].requests[0].cachedTokens=127;bool rejected=false;
    try{BuildPackedTokenPlan(forged,cachedStorage,3,4096);}catch(const std::invalid_argument&){rejected=true;}Check(rejected);
    Check(cached.CompletePrefill(returnedId,44,cachedStorage)&&cached.Stage(returnedId)==RequestStage::Finished);
    for(auto id:cachedIds)if(cached.Stage(id)!=RequestStage::Finished)Check(cached.Cancel(id,cachedStorage));
    // A full cache boundary without final hidden state cannot bypass prefill.
    for(auto invalid:std::vector<std::pair<int,bool>>{{128,false},{129,true},{-1,true}}) {
        FixedRows invalidStorage(4096);ContinuousBatchScheduler invalidPolicy(cachedConfig);
        auto id=invalidPolicy.Submit(returned);invalidStorage.SeedPrefix(id,invalid.first,invalid.second);
        rejected=false;
        try{invalidPolicy.Schedule(invalidStorage);}catch(const std::logic_error&){rejected=true;}
        Check(rejected&&invalidStorage.LogicalLength(id)==-1);
    }
    std::cout<<"portable_continuous_scheduler PASS stable rows, queued refill, cancellation, one-token budget, prefill EOS, invalid-token rejection, chunked prefill lifecycle and4-row/512-token global scheduling\n";
}
