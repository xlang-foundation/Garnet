// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <algorithm>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace Garnet {
struct ScratchLifetime {
    size_t slot,rows,columns,firstWrite,lastUse;
};
struct ScratchLane {size_t slot,lane;};
// Equal-shape buffers may alias only when one lifetime ends strictly before
// the next writer. Same-operation reads and writes therefore never alias.
// Callers must exclude inputs, outputs, weights, existing aliases and buffers
// whose first writer does not initialize their complete logical contents.
inline std::vector<ScratchLane> PlanScratchLifetimes(std::vector<ScratchLifetime> values) {
    std::unordered_set<size_t> unique;
    for(const auto& value:values)
        if(!value.rows||!value.columns||value.firstWrite>value.lastUse||
            value.lastUse==size_t(-1)||!unique.insert(value.slot).second)
            throw std::invalid_argument("Invalid scratch lifetime");
    std::sort(values.begin(),values.end(),[](const auto& a,const auto& b){
        return a.firstWrite!=b.firstWrite?a.firstWrite<b.firstWrite:a.slot<b.slot;
    });
    std::map<std::pair<size_t,size_t>,std::vector<size_t>> ends;
    std::vector<ScratchLane> result;
    for(const auto& value:values) {
        auto& lanes=ends[{value.rows,value.columns}];
        size_t lane=0;
        while(lane<lanes.size()&&lanes[lane]>=value.firstWrite)++lane;
        if(lane==lanes.size())lanes.push_back(value.lastUse);
        else lanes[lane]=value.lastUse;
        result.push_back({value.slot,lane});
    }
    return result;
}
}
