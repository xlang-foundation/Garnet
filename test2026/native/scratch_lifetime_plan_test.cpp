// SPDX-License-Identifier: Apache-2.0
#include "../../src/runtime/memory/scratch_lifetime_plan.h"
#include <iostream>
#include <stdexcept>
using namespace Garnet;
static void Check(bool value){if(!value)throw std::runtime_error("scratch lifetime invariant failed");}
int main() {
    const std::vector<ScratchLifetime> values={
        {3,16,5120,6,8},{0,16,5120,0,3},{2,16,5120,4,5},
        {1,16,5120,3,4},{4,16,17408,0,8},{5,16,5120,4,4}};
    const auto plan=PlanScratchLifetimes(values);
    std::map<size_t,size_t> lane;
    for(const auto& entry:plan)lane[entry.slot]=entry.lane;
    Check(lane[0]!=lane[1]); // Last read at3 cannot alias first write at3.
    Check(lane[0]==lane[2]&&lane[2]==lane[3]); // Disjoint lifetimes reuse.
    Check(lane[5]!=lane[1]&&lane[5]!=lane[2]);
    Check(lane[4]==0); // Other shapes have independent pools.
    for(const auto& a:values)for(const auto& b:values)
        if(a.slot!=b.slot&&a.rows==b.rows&&a.columns==b.columns&&lane[a.slot]==lane[b.slot])
            Check(a.lastUse<b.firstWrite||b.lastUse<a.firstWrite);
    auto reversed=values;std::reverse(reversed.begin(),reversed.end());
    const auto repeat=PlanScratchLifetimes(reversed);
    Check(repeat.size()==plan.size());
    for(size_t i=0;i<plan.size();++i)Check(repeat[i].slot==plan[i].slot&&repeat[i].lane==plan[i].lane);
    for(const auto& invalid:std::vector<std::vector<ScratchLifetime>>{
        {{0,1,1,2,1}},{{0,0,1,0,1}},{{0,1,1,0,size_t(-1)}},{{0,1,1,0,1},{0,1,1,2,3}}}) {
        bool rejected=false;try{PlanScratchLifetimes(invalid);}catch(const std::invalid_argument&){rejected=true;}
        Check(rejected);
    }
    Check(PlanScratchLifetimes({}).empty());
    std::cout<<"scratch lifetime plan PASS overlap boundaries, shape isolation, deterministic reuse and invalid inputs\n";
}
