// SPDX-License-Identifier: Apache-2.0
#include "../../src/backends/metal/prefill_profile_policy.h"
#include "../../src/runtime/scheduler/continuous_batch_scheduler.h"
#include "../../src/runtime/scheduler/request_row_leases.h"
#include "../../src/runtime/scheduler/packed_token_plan.h"
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
using namespace Garnet;
namespace {
size_t checks=0;
void Need(bool value,const char* reason) {
    ++checks;if(!value)throw std::runtime_error(reason);
}
template<class F> void Reject(F f,const char* reason) {
    bool rejected=false;try{f();}catch(const std::invalid_argument&){rejected=true;}
    Need(rejected,reason);
}
std::vector<DenseOwnerPrefillFamily> Families() {
    std::vector<DenseOwnerPrefillFamily> result;
    for(size_t rows:{size_t(1),size_t(2)})for(size_t t=2;rows*t<=512;t*=2)
        result.push_back({rows,t,std::min(t,size_t(128))});
    return result;
}
struct Storage final:KVSequenceAllocator {
    struct Item{RequestRowLease lease;int length=0,cached=0;};
    RequestRowLeases rows{4};std::map<uint64_t,Item> state;std::map<uint64_t,int> cached;
    bool Reserve(uint64_t id,int count) override {
        if(state.size()==4||state.count(id)||count<0||count>4096)return false;
        int prefix=cached.count(id)?cached[id]:0;
        state.emplace(id,Item{rows.Claim(id),prefix,prefix});return true;
    }
    bool EnsureCapacity(uint64_t id,int count) override{return state.count(id)&&count>=0&&count<=4096;}
    bool CommitLength(uint64_t id,int count) override {
        if(!EnsureCapacity(id,count))return false;state.at(id).length=count;return true;
    }
    bool Release(uint64_t id) override {
        auto it=state.find(id);if(it==state.end())return false;
        Need(rows.Release(it->second.lease.row,it->second.lease.epoch),"Release lost row epoch");
        state.erase(it);return true;
    }
    int LogicalLength(uint64_t id) const override{auto it=state.find(id);return it==state.end()?-1:it->second.length;}
    std::vector<int> PageTable(uint64_t) const override{return {};}
    KVSequencePlacement Placement(uint64_t id) const override {
        auto it=state.find(id);return it==state.end()?KVSequencePlacement{}:
            KVSequencePlacement{int(it->second.lease.row),it->second.lease.epoch};
    }
    int RestoredPrefixTokens(uint64_t id) const override{return state.count(id)?state.at(id).cached:0;}
    bool PrefixPredictionReady(uint64_t id) const override{return RestoredPrefixTokens(id)>0;}
};
ContinuousBatchSchedulerConfig Config(bool clip=true,int budget=512) {
    ContinuousBatchSchedulerConfig c;c.decodeBuckets={4};c.maxPrefillBatch=4;
    c.maxPrefillTokensPerTick=budget;c.maxActiveRequests=16;c.prefillChunkTokens=std::min(128,budget);
    c.completePrefillFirst=true;c.clipPrefillToBudget=clip;
    c.prefillChunkSelector=[clip](const BatchPlan& p,int available){
        std::vector<size_t> rows,left;
        for(const auto& r:p.requests){rows.push_back(size_t(r.requestRow));left.push_back(size_t(r.promptTokens-r.prefillOffset));}
        return int(SelectDenseOwnerPrefillChunk(rows,left,512,size_t(available),Families(),clip));
    };return c;
}
uint64_t Submit(ContinuousBatchScheduler& s,int prompt=2048,int output=128) {
    GenerationRequest request;request.promptTokens=prompt;request.maxNewTokens=output;
    auto id=s.Submit(request);Need(id!=0,"Submit rejected valid fixture");return id;
}
const BatchPlan& Prefill(const SchedulerTick& tick) {
    for(const auto& batch:tick.batches)if(batch.stage==RequestStage::Prefill)return batch;
    throw std::runtime_error("Expected prefill plan");
}
void Budget(const SchedulerTick& tick,const Storage& storage,int maximum=512) {
    const auto packed=BuildPackedTokenPlan(tick,storage,maximum,4096);
    Need(packed.tokenCount<=maximum,"Packed actual-token budget exceeded");
}
std::vector<uint64_t> Seed(ContinuousBatchScheduler& scheduler,Storage& storage,int count) {
    std::vector<uint64_t> result;
    for(int i=0;i<count;++i) {
        auto id=Submit(scheduler,2);auto tick=scheduler.Schedule(storage);Budget(tick,storage);
        Need(Prefill(tick).requests.at(0).id==id,"Seed ordering");
        while(Prefill(tick).requests.at(0).contextLength<2) {
            Need(scheduler.CompletePrefillChunk(id,Prefill(tick).requests.at(0).prefillTokens,storage),"Seed partial completion");
            tick=scheduler.Schedule(storage);Budget(tick,storage);
        }
        Need(scheduler.CompletePrefill(id,100+i,storage),"Seed completion");result.push_back(id);
    }
    return result;
}
void Ready(const SchedulerTick& tick,const std::vector<uint64_t>& ready) {
    if(ready.empty())return;
    Need(tick.batches.front().stage==RequestStage::Decode,"Decode must precede prefill");
    auto& batch=tick.batches.front();Need(batch.requests.size()==ready.size(),"Ready decoder omitted");
    for(size_t i=0;i<ready.size();++i)Need(batch.requests[i].id==ready[i],"Decode request identity changed");
}
}
int main(){try {
    const auto families=Families();Need(families.size()==17,"Prepared families changed");
    // Independent expected up-to shape: ceil(min(remaining,budget,maximum)),
    // including values much larger than size_t additions could safely hold.
    for(size_t row=0;row<4;++row)for(size_t maximum:{2ul,128ul,256ul,512ul})
    for(size_t left:{0ul,1ul,2ul,3ul,127ul,128ul,129ul,255ul,256ul,257ul,300ul,509ul,510ul,511ul,512ul,513ul,2048ul,std::numeric_limits<size_t>::max()})
    for(size_t budget:{0ul,1ul,2ul,127ul,128ul,129ul,255ul,256ul,257ul,509ul,510ul,511ul,512ul}) {
        size_t expected=2,bound=std::min({left,budget,maximum});
        while(expected<bound)expected*=2;if(left&&!budget)expected=0;
        const size_t selected=SelectDenseOwnerPrefillChunk({row},{left},maximum,budget,families,true);
        Need(selected==expected,"Clipped up-to profile selection");
        Need(!selected||std::min({left,selected,budget})<=budget,"Selector actual budget");
    }
    Need(SelectDenseOwnerPrefillChunk({0},{2048},512,511,families)==256,"Default policy changed");
    Need(SelectDenseOwnerPrefillChunk({0},{263},512,512,families)==256,"Default tail policy changed");
    auto narrow=families;narrow.erase(std::remove_if(narrow.begin(),narrow.end(),[](auto f){return f.chunk>128;}),narrow.end());
    Need(SelectDenseOwnerPrefillChunk({3},{2048},512,509,narrow,true)==128,"Unprepared wide family selected");
    for(int mutation=0;mutation<8;++mutation)Reject([&]{
        if(mutation==0)SelectDenseOwnerPrefillChunk({}, {},512,512,families,true);
        if(mutation==1)SelectDenseOwnerPrefillChunk({0,1},{100,100},512,512,families,true);
        if(mutation==2)SelectDenseOwnerPrefillChunk({0},{},512,512,families,true);
        if(mutation==3)SelectDenseOwnerPrefillChunk({4},{100},512,512,families,true);
        if(mutation==4)SelectDenseOwnerPrefillChunk({0},{100},513,512,families,true);
        if(mutation==5)SelectDenseOwnerPrefillChunk({0},{100},512,513,families,true);
        if(mutation==6){auto bad=families;bad.push_back({2,512,128});SelectDenseOwnerPrefillChunk({0},{100},512,512,bad,true);}
        if(mutation==7)SelectDenseOwnerPrefillChunk({0,0},{0,0},512,0,families,true);
    },"Invalid profile bounds accepted");
    for(int count=0;count<=3;++count) {
        Storage storage;ContinuousBatchScheduler scheduler(Config());auto ready=Seed(scheduler,storage,count);
        auto owner=Submit(scheduler),waiting=Submit(scheduler);auto tick=scheduler.Schedule(storage);
        Budget(tick,storage);Ready(tick,ready);auto first=Prefill(tick).requests.at(0);auto lease=storage.Placement(owner);
        Need(Prefill(tick).requests.size()==1&&first.id==owner,"Completion owner changed");
        Need(Prefill(tick).prefillChunkTokens==512&&first.prefillTokens==512-count,"Residual budget was not filled");
        Need(scheduler.QueuedRequestCount()==1,"Fifth/pending request admitted ahead of owner");
        Need(!scheduler.CompletePrefillChunk(owner,first.prefillTokens-1,storage),"Wrong acknowledgment accepted");
        Need(scheduler.AbandonPrefillChunk(owner),"Abandon failed");
        tick=scheduler.Schedule(storage);Budget(tick,storage);Ready(tick,ready);
        auto retry=Prefill(tick).requests.at(0);
        Need(retry.id==owner&&retry.prefillOffset==0&&retry.prefillTokens==first.prefillTokens&&retry.rowEpoch==lease.rowEpoch,"Retry changed range or lease");
        Need(scheduler.CompletePrefillChunk(owner,retry.prefillTokens,storage),"Partial commit rejected");
        tick=scheduler.Schedule(storage);Budget(tick,storage);auto second=Prefill(tick).requests.at(0);
        Need(second.id==owner&&second.prefillOffset==first.prefillTokens&&second.rowEpoch==lease.rowEpoch,"Continuation moved logical offset or owner");
        Need(scheduler.Cancel(owner,storage),"Owner cancel failed");tick=scheduler.Schedule(storage);Budget(tick,storage);
        auto fresh=Prefill(tick).requests.at(0);
        Need(fresh.id==waiting&&fresh.requestRow==lease.requestRow&&fresh.rowEpoch!=lease.rowEpoch,"Row reuse did not advance epoch");
        auto stale=tick;for(auto& b:stale.batches)if(b.stage==RequestStage::Prefill)b.requests[0].rowEpoch=lease.rowEpoch;
        Reject([&]{BuildPackedTokenPlan(stale,storage,512,4096);},"Stale lease accepted");
        Need(!scheduler.CompletePrefillChunk(owner,first.prefillTokens,storage),"Cancelled owner accepted commit");
        Need(scheduler.Cancel(waiting,storage),"Replacement cancel failed");
        for(auto id:ready)Need(scheduler.Cancel(id,storage),"Survivor cancel failed");
        Need(storage.state.empty(),"Cancellation leaked leases");
    }
    for(int tail:{1,300,509,510,511}) {
        Storage storage;ContinuousBatchScheduler scheduler(Config());auto ready=Seed(scheduler,storage,3);
        auto id=Submit(scheduler,tail,1);auto tick=scheduler.Schedule(storage);Budget(tick,storage);Ready(tick,ready);
        auto item=Prefill(tick).requests.at(0);int actual=std::min(tail,509);
        Need(item.prefillTokens==actual&&Prefill(tick).prefillChunkTokens==(tail==1?2:512),"Tail extent or count");
        if(tail>509) {
            Need(scheduler.CompletePrefillChunk(id,actual,storage),"Clipped tail commit");
            tick=scheduler.Schedule(storage);Budget(tick,storage);item=Prefill(tick).requests.at(0);
            Need(item.prefillOffset==509&&item.prefillTokens==tail-509&&Prefill(tick).prefillChunkTokens==2,"Tail continuation");
        }
        Need(scheduler.CompletePrefill(id,200,storage),"Final tail/head completion");
        for(auto survivor:ready)Need(scheduler.Cancel(survivor,storage),"Tail survivor cleanup");
        Need(storage.state.empty(),"Tail final output leaked owner");
    }
    {
        Storage storage;ContinuousBatchScheduler scheduler(Config());auto ready=Seed(scheduler,storage,3);
        auto id=Submit(scheduler,2048,1);storage.cached[id]=2048;
        auto tick=scheduler.Schedule(storage);Budget(tick,storage);Ready(tick,ready);auto item=Prefill(tick).requests.at(0);
        Need(item.cachedTokens==2048&&item.prefillOffset==2048&&item.prefillTokens==0&&Prefill(tick).prefillChunkTokens==2,"Exact-prefix head-only plan");
        Need(scheduler.CompletePrefill(id,200,storage),"Head-only completion rejected");
        for(auto survivor:ready)Need(scheduler.Cancel(survivor,storage),"Head survivor cleanup");
    }
    {
        // A real exhausted budget: three ready decoders, global budget three.
        Storage storage;ContinuousBatchScheduler scheduler(Config(true,3));auto ready=Seed(scheduler,storage,3);
        auto id=Submit(scheduler),waiting=Submit(scheduler);auto tick=scheduler.Schedule(storage);
        Budget(tick,storage,3);Ready(tick,ready);Need(tick.batches.size()==1,"Zero budget scheduled positive prefill");
        auto lease=storage.Placement(id);Need(lease.rowEpoch!=0,"Deferral released reserved owner");
        Need(scheduler.CompleteDecode(ready.back(),201,true,storage),"Zero budget decoder finish");ready.pop_back();
        tick=scheduler.Schedule(storage);Budget(tick,storage,3);Ready(tick,ready);auto item=Prefill(tick).requests.at(0);
        Need(item.id==id&&item.prefillTokens==1&&item.rowEpoch==lease.rowEpoch,"Zero-budget deferral lost FIFO owner or lease");
        Need(scheduler.Cancel(id,storage)&&scheduler.Cancel(waiting,storage),"Deferral cleanup");
        for(auto survivor:ready)Need(scheduler.Cancel(survivor,storage),"Deferral survivor cleanup");
    }
    for(int mutation=0;mutation<5;++mutation)Reject([&]{
        auto cfg=Config();if(mutation==0)cfg.completePrefillFirst=false;
        if(mutation==1)cfg.prefillChunkTokens=0;if(mutation==2)cfg.prefillChunkSelector={};
        if(mutation==3)cfg.maxPrefillBatch=0;if(mutation==4){cfg.prefillChunkTokens=1;cfg.maxPrefillTokensPerTick=1;}
        ContinuousBatchScheduler scheduler(cfg);
    },"Invalid clipped scheduler config accepted");
    {
        Storage storage;ContinuousBatchScheduler scheduler(Config());
        Need(!scheduler.SetPrefillCompletionPriority(false)&&!scheduler.SetPrefillChunkTokens(0)&&!scheduler.SetPrefillGrouping(128,2,1024),"Incompatible clipped policy accepted");
        auto id=Submit(scheduler);Need(!scheduler.SetPrefillBudgetClip(false)&&!scheduler.SetPrefillChunkTokens(256),"Policy changed with tracked requests");
        Need(scheduler.Cancel(id,storage)&&scheduler.ForgetFinished(id),"Tracked request cleanup");
        Need(scheduler.SetPrefillBudgetClip(false)&&scheduler.SetPrefillCompletionPriority(false),"Policy disable failed after cleanup");
        Need(!scheduler.SetPrefillBudgetClip(true),"Clip enabled without completion-first");
        Need(scheduler.SetPrefillCompletionPriority(true)&&scheduler.SetPrefillBudgetClip(true),"Policy reenable failed");
    }
    std::cout<<"prefill_budget_clip PASS checks="<<checks<<" active=509/510/511 prepared=512 max_actual_tick=512 retry_prefix_epoch_fifo=true default_unchanged=true\n";
    return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
