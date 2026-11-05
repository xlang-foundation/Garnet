// SPDX-License-Identifier: Apache-2.0
#include "../../src/runtime/memory/scratch_alias_group_plan.h"
#include <iostream>
using namespace Garnet;
static void Check(bool value){if(!value)throw std::runtime_error("scratch alias group invariant failed");}
int main() {
    constexpr auto absent=size_t(-1);
    const std::vector<ScratchAliasMember> members={
        {0,100,2,32,2,0,2,true,true,true},
        {1,100,2,32,5,absent,5,true,false,false},
        {2,200,2,32,8,6,8,true,true,true},
        {3,300,2,32,7,5,7,true,true,true}};
    const auto groups=PlanScratchAliasGroups(members);Check(groups.size()==3);
    Check(groups[0].slots==std::vector<size_t>({0,1}));
    Check(groups[0].lifetime.firstWrite==0&&groups[0].lifetime.lastUse==5);
    std::vector<ScratchLifetime> lifetimes;for(const auto& group:groups)lifetimes.push_back(group.lifetime);
    const auto plan=PlanScratchLifetimes(lifetimes);std::map<size_t,size_t> lanes;
    for(const auto& assignment:plan)lanes[assignment.slot]=assignment.lane;
    Check(lanes[0]==lanes[2]);Check(lanes[0]!=lanes[3]); // Alias last read at5 overlaps a write at5.
    auto reversed=members;std::reverse(reversed.begin(),reversed.end());
    const auto again=PlanScratchAliasGroups(reversed);Check(again.size()==groups.size());
    for(size_t i=0;i<groups.size();++i)Check(again[i].slots==groups[i].slots&&again[i].lifetime.slot==groups[i].lifetime.slot);
    // Every view participates: protected ownership, physical-layout mismatch,
    // same-op input/output, early reads and a partial/nonresident first writer.
    for(int kind=0;kind<6;++kind) {
        auto bad=members;
        if(kind==0)bad[1].allowed=false;
        if(kind==1)bad[1].columns=64;
        if(kind==2)bad[1].firstRead=0;
        if(kind==3)bad[0].completeFirstWrite=false;
        if(kind==4)bad[0].residentFirstWrite=false;
        if(kind==5)bad[1].firstWrite=0;
        const auto filtered=PlanScratchAliasGroups(bad);Check(filtered.size()==2);
        for(const auto& group:filtered)Check(group.lifetime.slot!=0);
    }
    for(int kind=0;kind<5;++kind) {
        auto bad=members;
        if(kind==0)bad[1].slot=0;
        if(kind==1)bad[1].storage=0;
        if(kind==2)bad[1].rows=0;
        if(kind==3)bad[1].lastUse=absent;
        if(kind==4)bad[1].lastUse=4;
        bool rejected=false;try{PlanScratchAliasGroups(bad);}catch(const std::invalid_argument&){rejected=true;}
        Check(rejected);
    }
    Check(PlanScratchAliasGroups({}).empty());
    std::cout<<"scratch alias plan PASS storage union, exact overlap boundary, protected views, writer initialization, deterministic groups and invalid metadata\n";
}
