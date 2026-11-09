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
int main(){try{
 ContinuousBatchSchedulerConfig cfg;cfg.decodeBuckets={4};cfg.maxPrefillBatch=4;cfg.maxPrefillTokensPerTick=512;cfg.prefillChunkTokens=128;
 auto submit=[](ContinuousBatchScheduler& scheduler,int length,bool vision=false,int delta=0){GenerationRequest request;request.promptTokens=length;request.maxNewTokens=8;request.hasVision=vision;request.positionDelta=delta;auto id=scheduler.Submit(request);Check(id>0);return id;};
 {
  FixedRows storage(4096);ContinuousBatchScheduler scheduler(cfg);Check(!scheduler.SetPrefillGrouping(128,1,8192)&&!scheduler.SetPrefillGrouping(128,17,9000)&&!scheduler.SetPrefillGrouping(128,16,8191));Check(scheduler.SetPrefillGrouping(128,16,8192));std::vector<uint64_t> ids;for(int row=0;row<4;++row)ids.push_back(submit(scheduler,2051));
  auto tick=scheduler.Schedule(storage);Check(tick.batches.size()==1);const auto& group=tick.batches[0];Check(group.prefillGroupChunks==16&&group.prefillChunkTokens==128&&group.scheduledTokens==8192);auto packed=BuildPackedTokenPlan(tick,storage,8192,4096);Check(packed.tokenCount==8192&&packed.segments.size()==4);
  Check(!scheduler.SetPrefillGrouping(128,2,1024));for(auto id:ids){Check(!scheduler.CompletePrefill(id,8,storage));Check(!scheduler.CompletePrefillChunk(id,2047,storage));Check(scheduler.CompletePrefillChunk(id,2048,storage));Check(!scheduler.CompletePrefillChunk(id,2048,storage));Check(storage.LogicalLength(id)==2048);}
  tick=scheduler.Schedule(storage);Check(tick.batches.size()==1&&tick.batches[0].prefillGroupChunks==0&&tick.batches[0].scheduledTokens==12);for(auto id:ids){Check(!scheduler.CompletePrefillChunk(id,3,storage));Check(scheduler.CompletePrefill(id,17,storage));}tick=scheduler.Schedule(storage);Check(tick.batches.size()==1&&tick.batches[0].stage==RequestStage::Decode&&tick.batches[0].requests.size()==4);
 }
 {
  FixedRows storage(4096);ContinuousBatchScheduler scheduler(cfg);Check(scheduler.SetPrefillGrouping(128,16,8192));std::vector<uint64_t> ids;for(int row=0;row<4;++row)ids.push_back(submit(scheduler,4095));auto first=scheduler.Schedule(storage);auto old=storage.Placement(ids[0]);std::vector<KVSequencePlacement> survivors;for(int row=1;row<4;++row)survivors.push_back(storage.Placement(ids[row]));Check(scheduler.Cancel(ids[0],storage));for(int row=1;row<4;++row){Check(scheduler.AbandonPrefillChunk(ids[row]));Check(!scheduler.CompletePrefillChunk(ids[row],2048,storage));Check(storage.LogicalLength(ids[row])==0);}auto next=submit(scheduler,4095);auto tick=scheduler.Schedule(storage);Check(tick.batches.size()==1&&tick.batches[0].prefillGroupChunks==16);auto replacement=storage.Placement(next);Check(replacement.requestRow==old.requestRow&&replacement.rowEpoch!=old.rowEpoch);for(int row=1;row<4;++row){auto got=storage.Placement(ids[row]);Check(got.requestRow==survivors[row-1].requestRow&&got.rowEpoch==survivors[row-1].rowEpoch);}
 }
 for(int kind=0;kind<4;++kind){
  FixedRows storage(4096);ContinuousBatchScheduler scheduler(cfg);Check(scheduler.SetPrefillGrouping(128,16,8192));std::vector<uint64_t> ids;for(int row=0;row<4;++row)ids.push_back(submit(scheduler,row==0&&kind==0?128:2048,row==0&&kind==2,row==0&&kind==3?1:0));if(kind==1)storage.SeedPrefix(ids[0],2048);auto tick=scheduler.Schedule(storage);Check(tick.batches.size()==1&&tick.batches[0].prefillGroupChunks==0&&tick.batches[0].scheduledTokens<=512);
  if(kind<2){Check(scheduler.CompletePrefill(ids[0],42,storage));for(int row=1;row<4;++row)Check(scheduler.CompletePrefillChunk(ids[row],128,storage));tick=scheduler.Schedule(storage);Check(tick.batches.size()==2&&tick.batches[0].stage==RequestStage::Decode&&tick.batches[0].requests.size()==1&&tick.batches[1].prefillGroupChunks==0&&tick.batches[1].scheduledTokens==384);}
 }
 std::cout<<"Native grouped scheduling policy PASS: shared8192 budget, actual128 chunks, tails, decoder fairness, head-only cache, vision/delta fallback, abandon/cancel and epoch reuse\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
