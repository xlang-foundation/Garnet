// SPDX-License-Identifier: Apache-2.0
#pragma once
#import <Metal/Metal.h>
#include <vector>
#include <unordered_set>
#include <stdexcept>
#include <cstdint>
// Caller serializes access and exclusively owns the queue. No model arithmetic.
// The native context aborts/drains GPU work before destroying this plan.
namespace Garnet {
template<bool TestFaults=false>
class BoundedAffineResidencyPlan {
 id<MTLDevice> device_=nil;id<MTLCommandQueue> queue_=nil;
 id<MTLResidencySet> set_=nil;
 uint64_t payload_=0,budget_=0,renewals_=0;
public:
 BoundedAffineResidencyPlan()=default;
 BoundedAffineResidencyPlan(const BoundedAffineResidencyPlan&)=delete;
 BoundedAffineResidencyPlan& operator=(const BoundedAffineResidencyPlan&)=delete;
 ~BoundedAffineResidencyPlan(){CloseNoThrow();}
 bool Ready()const{return set_!=nil;}
 uint64_t PayloadBytes()const{return payload_;}
 uint64_t BudgetBytes()const{return budget_;}
 uint64_t Renewals()const{return renewals_;}
 uint64_t ChargedBytes()const{return set_?set_.allocatedSize:0;}
 size_t Count()const{return set_?set_.allocationCount:0;}
 void Prepare(id<MTLDevice> device,id<MTLCommandQueue> queue,
              const std::vector<id<MTLBuffer>>& buffers,uint64_t budget,int fault=0){
  if(Ready())throw std::logic_error("Residency plan already prepared");
  if(!device||!queue||queue.device!=device||buffers.empty())throw std::invalid_argument("Residency requires device, matching queue and buffers");
  constexpr uint64_t reserve=256ull*1024*1024;
  if(!budget||device.recommendedMaxWorkingSetSize<=reserve||budget>device.recommendedMaxWorkingSetSize-reserve)throw std::invalid_argument("Residency budget exceeds device envelope");
  if(device.currentAllocatedSize>device.recommendedMaxWorkingSetSize-reserve)throw std::invalid_argument("Residency process allocation exceeds device envelope");
  if(@available(macOS 15.0,*)){
   std::unordered_set<uintptr_t> seen;std::vector<id<MTLBuffer>> unique;uint64_t bytes=0;
   for(id<MTLBuffer> b:buffers){
    if(!b||b.device!=device||b.heap||!b.length||!b.allocatedSize)throw std::invalid_argument("Residency requires standalone same-device buffers");
    const auto key=reinterpret_cast<uintptr_t>((__bridge void*)b);
    if(seen.insert(key).second){if(b.allocatedSize>budget-bytes)throw std::invalid_argument("Residency payload exceeds budget");bytes+=b.allocatedSize;unique.push_back(b);}
   }
   auto desc=[MTLResidencySetDescriptor new];desc.label=@"Garnet bounded immutable residency prototype";desc.initialCapacity=unique.size();NSError* error=nil;
   auto candidate=[device newResidencySetWithDescriptor:desc error:&error];
if constexpr(TestFaults){if(fault==1)candidate=nil;}
   if(!candidate)throw std::runtime_error("Metal residency set creation failed");
   for(id<MTLBuffer> b:unique)[candidate addAllocation:b];[candidate commit];
   if(candidate.allocationCount!=unique.size()||candidate.allocatedSize>budget||(TestFaults&&fault==2)){[candidate removeAllAllocations];[candidate commit];throw std::runtime_error("Residency committed charge exceeds budget/count");}
   [queue addResidencySet:candidate];[candidate requestResidency];
   device_=device;queue_=queue;set_=candidate;payload_=bytes;budget_=budget;renewals_=1;
  }else throw std::runtime_error("Metal residency requires macOS15");
 }
 void Renew(){if(!Ready())throw std::logic_error("Residency plan is not ready");[set_ requestResidency];++renewals_;}
 // Queue is exclusively owned by the caller. Appending a marker drains all
 // prior GPU references before releasing the queue's residency registration.
 void Close(){
  if(!Ready())return;
  auto marker=[queue_ commandBuffer];if(!marker)throw std::runtime_error("Cannot drain residency queue");
  [marker commit];[marker waitUntilCompleted];
  if(marker.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error("Residency queue drain failed");
  [queue_ removeResidencySet:set_];[set_ endResidency];[set_ removeAllAllocations];[set_ commit];
  set_=nil;queue_=nil;device_=nil;payload_=0;budget_=0;renewals_=0;
 }
 void CloseNoThrow()noexcept{try{Close();}catch(...){// The context destructor has already drained/aborted. A marker-allocation
   // failure must still detach and release the residency registration.
   if(set_){[queue_ removeResidencySet:set_];[set_ endResidency];[set_ removeAllAllocations];[set_ commit];}
   set_=nil;queue_=nil;device_=nil;payload_=budget_=renewals_=0;}}
};

using BoundedAffineResidency=BoundedAffineResidencyPlan<false>;
} // namespace Garnet
