// SPDX-License-Identifier: Apache-2.0
#include "../../src/runtime/scheduler/continuous_batch_scheduler.h"
#include "../../src/runtime/scheduler/request_row_leases.h"
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
    std::cout<<"portable_continuous_scheduler PASS CPU allocation interface, priority order, stable A/B/C/D rows, queued E refill and cancellation\n";
}
