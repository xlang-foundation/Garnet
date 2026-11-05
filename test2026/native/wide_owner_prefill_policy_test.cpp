// SPDX-License-Identifier: Apache-2.0
#include "../../src/backends/metal/prefill_profile_policy.h"
#include "../../src/runtime/scheduler/continuous_batch_scheduler.h"
#include "../../src/runtime/scheduler/request_row_leases.h"
#include "../../src/runtime/scheduler/packed_token_plan.h"
#include <iostream>
#include <map>
#include <stdexcept>
using namespace Garnet;
namespace {
void Need(bool v){if(!v)throw std::runtime_error("Wide owner profile policy invariant failed");}
std::vector<DenseOwnerPrefillFamily> Families() {
    std::vector<DenseOwnerPrefillFamily> f;
    for(size_t rows:{size_t(1),size_t(2)})for(size_t t=2;rows*t<=512;t*=2)f.push_back({rows,t,std::min(t,size_t(128))});
    return f;
}
struct Storage final:KVSequenceAllocator {
    struct Item{RequestRowLease lease;int length=0,cached=0;};
    RequestRowLeases rows{4};std::map<uint64_t,Item> state;std::map<uint64_t,int> cached;
    bool Reserve(uint64_t id,int count) override {
        if(state.size()==4||state.count(id)||count<0||count>4096)return false;
        int prefix=cached.count(id)?cached[id]:0;state.emplace(id,Item{rows.Claim(id),prefix,prefix});return true;
    }
    bool EnsureCapacity(uint64_t id,int count) override{return state.count(id)&&count>=0&&count<=4096;}
    bool CommitLength(uint64_t id,int count) override{if(!EnsureCapacity(id,count))return false;state.at(id).length=count;return true;}
    bool Release(uint64_t id) override{auto it=state.find(id);if(it==state.end())return false;Need(rows.Release(it->second.lease.row,it->second.lease.epoch));state.erase(it);return true;}
    int LogicalLength(uint64_t id) const override{auto it=state.find(id);return it==state.end()?-1:it->second.length;}
    std::vector<int> PageTable(uint64_t) const override{return {};}
    KVSequencePlacement Placement(uint64_t id) const override{auto it=state.find(id);return it==state.end()?KVSequencePlacement{}:KVSequencePlacement{int(it->second.lease.row),it->second.lease.epoch};}
    int RestoredPrefixTokens(uint64_t id) const override{return state.count(id)?state.at(id).cached:0;}
    bool PrefixPredictionReady(uint64_t id) const override{return RestoredPrefixTokens(id)>0;}
};
ContinuousBatchSchedulerConfig Config() {
    ContinuousBatchSchedulerConfig c;c.decodeBuckets={4};c.maxPrefillBatch=4;c.maxPrefillTokensPerTick=512;c.maxActiveRequests=16;c.prefillChunkTokens=128;
    c.prefillChunkSelector=[](const BatchPlan& p,int budget){
        std::vector<size_t> rows,left;
        for(const auto& r:p.requests){rows.push_back(size_t(r.requestRow));left.push_back(size_t(r.promptTokens-r.prefillOffset));}
        return int(SelectDenseOwnerPrefillChunk(rows,left,512,size_t(budget),Families()));
    };return c;
}
uint64_t Submit(ContinuousBatchScheduler& s,int prompt=2048,int output=8){GenerationRequest r;r.promptTokens=prompt;r.maxNewTokens=output;auto id=s.Submit(r);Need(id!=0);return id;}
const BatchPlan& Prefill(const SchedulerTick& t){for(const auto& b:t.batches)if(b.stage==RequestStage::Prefill)return b;throw std::runtime_error("Missing prefill plan");}
void Budget(const SchedulerTick& t,const Storage& storage){auto p=BuildPackedTokenPlan(t,storage,512,4096);Need(p.tokenCount<=512);}
}
int main(){try{
    auto families=Families();Need(families.size()==17);size_t cases=0;
    auto choose=[&](std::vector<size_t> rows,std::vector<size_t> left,size_t wanted,size_t budget=512,size_t maximum=512){Need(SelectDenseOwnerPrefillChunk(rows,left,maximum,budget,families)==wanted);++cases;};
    choose({0},{2048},512);choose({3},{2048},512);choose({1,2},{2048,2048},256);choose({3,2},{2048,2048},256);
    choose({0,2},{2048,2048},128);choose({0,1,2},{2048,2048,2048},128);choose({0,1,2,3},{2048,2048,2048,2048},128);
    choose({0},{263},256);choose({0},{7},8);choose({0},{1},2);choose({0},{0},2,0);
    choose({0,3},{0,2048},512);choose({0},{2048},256,511);choose({0,1},{2048,2048},128,510);
    choose({0,1},{2000,16},256,510);choose({0},{100},128);choose({0},{2048},64,512,64);choose({0},{1},0,0);
    auto narrow=families;narrow.erase(std::remove_if(narrow.begin(),narrow.end(),[](auto p){return p.chunk>128;}),narrow.end());
    Need(SelectDenseOwnerPrefillChunk({3},{2048},512,512,narrow)==128);++cases;
    for(int mutation=0;mutation<5;++mutation){bool rejected=false;auto f=families;try{
        if(mutation==0)SelectDenseOwnerPrefillChunk({0,0},{10,10},512,512,f);
        if(mutation==1)SelectDenseOwnerPrefillChunk({4},{10},512,512,f);
        if(mutation==2)SelectDenseOwnerPrefillChunk({0},{10},513,512,f);
        if(mutation==3)SelectDenseOwnerPrefillChunk({0},{10},512,513,f);
        if(mutation==4){f.push_back({2,512,128});SelectDenseOwnerPrefillChunk({0},{10},512,512,f);}
    }catch(const std::invalid_argument&){rejected=true;}Need(rejected);++cases;}
    {
        Storage storage;ContinuousBatchScheduler s(Config());auto a=Submit(s);auto tick=s.Schedule(storage);Budget(tick,storage);
        Need(Prefill(tick).prefillChunkTokens==512&&Prefill(tick).requests[0].prefillTokens==512);auto leaseA=storage.Placement(a);
        Need(s.CompletePrefillChunk(a,512,storage));auto b=Submit(s);tick=s.Schedule(storage);Budget(tick,storage);
        Need(Prefill(tick).prefillChunkTokens==256&&Prefill(tick).scheduledTokens==512);
        auto oldB=storage.Placement(b);Need(s.CompletePrefillChunk(a,256,storage)&&s.CompletePrefillChunk(b,256,storage)&&s.Cancel(b,storage));
        auto c=Submit(s,16,4);tick=s.Schedule(storage);Budget(tick,storage);auto leaseC=storage.Placement(c);
        Need(leaseC.requestRow==oldB.requestRow&&leaseC.rowEpoch!=oldB.rowEpoch&&Prefill(tick).prefillChunkTokens==256);
        Need(s.CompletePrefillChunk(a,256,storage)&&s.CompletePrefill(c,100,storage));
        for(int i=0;i<3;++i){tick=s.Schedule(storage);Budget(tick,storage);Need(tick.batches.size()==2&&Prefill(tick).prefillChunkTokens==256);
            Need(s.CompletePrefillChunk(a,256,storage)&&s.CompleteDecode(c,101+i,i==2,storage));}
        tick=s.Schedule(storage);Budget(tick,storage);Need(Prefill(tick).requests[0].prefillTokens==256);
        Need(s.CompletePrefill(a,200,storage)&&s.CompleteDecode(a,201,true,storage));
        auto d=Submit(s,20,1);storage.cached[d]=20;tick=s.Schedule(storage);Budget(tick,storage);auto leaseD=storage.Placement(d);
        Need(Prefill(tick).scheduledTokens==0&&Prefill(tick).prefillChunkTokens==2&&leaseD.requestRow==leaseA.requestRow&&leaseD.rowEpoch!=leaseA.rowEpoch);
        Need(s.CompletePrefill(d,202,storage)&&!s.CompletePrefillChunk(b,256,storage)&&storage.state.empty());++cases;
    }
    {
        Storage storage;ContinuousBatchScheduler s(Config());std::vector<uint64_t> ids;
        for(int i=0;i<5;++i)ids.push_back(Submit(s));auto tick=s.Schedule(storage);Budget(tick,storage);
        Need(Prefill(tick).requests.size()==4&&Prefill(tick).prefillChunkTokens==128);
        auto old=storage.Placement(ids[0]);for(int i=0;i<4;++i)Need(s.CompletePrefillChunk(ids[i],128,storage));
        Need(s.Cancel(ids[0],storage));tick=s.Schedule(storage);Budget(tick,storage);auto fresh=storage.Placement(ids[4]);
        Need(Prefill(tick).requests.size()==4&&Prefill(tick).prefillChunkTokens==128&&fresh.requestRow==old.requestRow&&fresh.rowEpoch!=old.rowEpoch);
        for(int i=1;i<5;++i)Need(s.Cancel(ids[i],storage));Need(storage.state.empty());++cases;
    }
    {
        Storage storage;ContinuousBatchScheduler s(Config());auto a=Submit(s),b=Submit(s),c=Submit(s);s.Schedule(storage);
        Need(s.CompletePrefillChunk(a,128,storage)&&s.CompletePrefillChunk(c,128,storage)&&s.Cancel(b,storage));
        auto tick=s.Schedule(storage);Budget(tick,storage);Need(Prefill(tick).prefillChunkTokens==128&&Prefill(tick).requests.size()==2);
        Need(s.Cancel(a,storage)&&s.Cancel(c,storage));++cases;
    }
    {
        Storage storage;auto cfg=Config();int calls=0;auto select=cfg.prefillChunkSelector;
        cfg.prefillChunkSelector=[&](const BatchPlan& p,int n){return calls++?select(p,n):0;};
        ContinuousBatchScheduler s(cfg);auto id=Submit(s);Need(s.Schedule(storage).batches.empty());auto lease=storage.Placement(id);
        auto tick=s.Schedule(storage);Budget(tick,storage);Need(Prefill(tick).prefillChunkTokens==512&&storage.Placement(id).rowEpoch==lease.rowEpoch);Need(s.Cancel(id,storage));++cases;
    }
    for(int invalid:{512,513}){
        Storage storage;auto cfg=Config();cfg.prefillChunkSelector=[&](const BatchPlan&,int){return invalid;};
        ContinuousBatchScheduler s(cfg);auto a=Submit(s),b=Submit(s);bool rejected=false;
        try{s.Schedule(storage);}catch(const std::logic_error&){rejected=true;}Need(rejected&&s.Cancel(a,storage)&&s.Cancel(b,storage)&&storage.state.empty());++cases;
    }
    std::cout<<"wide_owner_prefill_policy PASS cases="<<cases<<" prepared_families=17 global_budget=512 stable_rows_and_prefix=true\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
