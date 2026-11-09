// SPDX-License-Identifier: Apache-2.0
#include "../../src/runtime/kv/prefix_cache.h"
#include <atomic>
#include <iostream>

using namespace Garnet;
namespace {
void Require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
struct State final:PrefixSnapshot {
    int marker;
    std::shared_ptr<std::atomic<int>> destroyed;
    State(PrefixCacheIdentity identity,size_t tokens,int value,size_t bytes=64,
          std::shared_ptr<std::atomic<int>> counter={})
        :PrefixSnapshot(std::move(identity),tokens,bytes),marker(value),destroyed(std::move(counter)) {}
    ~State(){if(destroyed)++*destroyed;}
};
PrefixCacheIdentity Identity(){return {"weights-v1","metal-q4-kv-bf16","tokenizer-v1","mrope-v1","tenant-A",""};}
void Matches() {
    auto id=Identity();PrefixCache cache(8,65536);
    Require(cache.Insert({1,2},std::make_shared<State>(id,2,20)),"insert short checkpoint failed");
    Require(cache.Insert({1,2,3,4},std::make_shared<State>(id,4,40)),"insert long checkpoint failed");
    Require(cache.Lookup(id,{1,2,3,4,5}).tokens==4,"append must reuse longest stored checkpoint");
    Require(cache.Lookup(id,{1,2,9,4}).tokens==2,"edit must fall back before changed token");
    Require(cache.Lookup(id,{1,2,3,4}).tokens==4,"exact prompt prefix did not match");
    Require(!cache.Lookup(id,{1}),"short prompt matched a future recurrent state");
    auto different=id;different.model="other-weights";
    Require(!cache.Lookup(different,{1,2,3,4}),"foreign model reused prefix");
    different=id;different.executionProfile="other-kv-format";
    Require(!cache.Lookup(different,{1,2,3,4}),"foreign cache format reused prefix");
    different=id;different.tokenizer="other-tokenizer";
    Require(!cache.Lookup(different,{1,2,3,4}),"foreign tokenizer reused prefix");
    different=id;different.positionPolicy="other-rope";
    Require(!cache.Lookup(different,{1,2,3,4}),"foreign position policy reused prefix");
    different=id;different.isolation="tenant-B";
    Require(!cache.Lookup(different,{1,2,3,4}),"cross-tenant prefix reuse");
    different=id;different.multimodal="different-image-digest";
    Require(!cache.Lookup(different,{1,2,3,4}),"different multimodal content reused prefix");
    Require(!cache.Lookup(id,{9,2,3,4}),"changed first token reused state");
    Require(cache.Lookup(id,{1,2,3}).tokens==2,"recurrent state was truncated from a later boundary");
}
void EvictionAndLifetime() {
    auto id=Identity();PrefixCache cache(2,65536);auto destroyed=std::make_shared<std::atomic<int>>(0);
    cache.Insert({1},std::make_shared<State>(id,1,10,64,destroyed));
    cache.Insert({2},std::make_shared<State>(id,1,20,64,destroyed));
    auto active=cache.Lookup(id,{1,9});
    cache.Insert({3},std::make_shared<State>(id,1,30,64,destroyed));
    Require(!cache.Lookup(id,{2}),"LRU entry was not evicted");
    Require(*destroyed==1,"eviction did not release unused snapshot");
    cache.Insert({1},std::make_shared<State>(id,1,11,64,destroyed));
    Require(std::static_pointer_cast<const State>(active.snapshot)->marker==10,"replacement mutated an active snapshot");
    Require(std::static_pointer_cast<const State>(cache.Lookup(id,{1}).snapshot)->marker==11,"replacement not visible to new requests");
    cache.Clear();Require(cache.Stats().entries==0&&cache.Stats().accountedBytes==0,"clear leaked cache accounting");
    Require(std::static_pointer_cast<const State>(active.snapshot)->marker==10,"clear invalidated active inference owner");
    Require(*destroyed==3,"clear released active owner or retained unused owner");
    active={};Require(*destroyed==4,"last inference owner did not release snapshot");
}
void MaintenanceRecency() {
    auto id=Identity();PrefixCache cache(2,65536);
    cache.Insert({1},std::make_shared<State>(id,1,10));
    cache.Insert({2},std::make_shared<State>(id,1,20));
    const auto before=cache.Stats();auto foreign=id;foreign.isolation="other-tenant";
    Require(!cache.Touch(foreign,{1})&&!cache.Touch(id,{1,9})&&!cache.Touch(id,{}),"maintenance matched foreign or inexact key");
    Require(cache.Touch(id,{1}),"exact maintenance key missing");
    const auto after=cache.Stats();
    Require(after.hits==before.hits&&after.misses==before.misses&&after.evictions==before.evictions&&
        after.replacements==before.replacements&&after.entries==before.entries&&after.accountedBytes==before.accountedBytes,
        "maintenance changed request counters or storage");
    cache.Insert({3},std::make_shared<State>(id,1,30));
    Require(!cache.Lookup(id,{2})&&cache.Lookup(id,{1}),"maintenance did not preserve expected eviction order");
    cache.Clear();Require(!cache.Touch(id,{1}),"maintenance resurrected retired snapshot");
}
void BudgetsAndInvalidBoundaries() {
    auto id=Identity();PrefixCache cache(4,1024);
    Require(!cache.Insert({1},std::make_shared<State>(id,1,0,SIZE_MAX)),"oversized snapshot accepted");
    Require(cache.Stats().entries==0,"oversized insertion changed cache");
    bool rejected=false;
    try{cache.Insert({1},std::make_shared<State>(id,2,0));}catch(const std::invalid_argument&){rejected=true;}
    Require(rejected,"state/token boundary mismatch accepted");
    for(int64_t i=0;i<20;++i)cache.Insert({i},std::make_shared<State>(id,1,int(i),128));
    Require(cache.Stats().accountedBytes<=1024&&cache.Stats().entries<=4,"cache exceeded memory/cardinality budget");
    PrefixCache disabled(0,1024);
    Require(!disabled.Insert({1},std::make_shared<State>(id,1,0)),"disabled cache retained state");
}
}
int main(){try{Matches();EvictionAndLifetime();MaintenanceRecency();BudgetsAndInvalidBoundaries();
    std::cout<<"native prefix matching, checkpoint, isolation, LRU and owning-lease gates passed\n";return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<"\n";return 1;}}
