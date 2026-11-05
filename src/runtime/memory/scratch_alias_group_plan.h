// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "scratch_lifetime_plan.h"
#include <cstdint>
#include <limits>
#include <utility>
namespace Garnet {
// The caller enumerates EVERY slot sharing a storage identity, and establishes
// allowed only for the same default physical layout with no protected owner.
// Logical element count alone is not a physical-layout proof.
struct ScratchAliasMember {
    size_t slot;
    uintptr_t storage;
    size_t rows,columns,firstRead,firstWrite,lastUse;
    bool allowed,completeFirstWrite,residentFirstWrite;
};
struct ScratchAliasGroup {
    ScratchLifetime lifetime;
    std::vector<size_t> slots;
};
inline std::vector<ScratchAliasGroup> PlanScratchAliasGroups(const std::vector<ScratchAliasMember>& members) {
    const size_t absent=std::numeric_limits<size_t>::max();
    std::map<uintptr_t,std::vector<const ScratchAliasMember*>> storage;
    std::unordered_set<size_t> slots;
    for(const auto& member:members) {
        if(!member.storage||!member.rows||!member.columns||!slots.insert(member.slot).second||
           member.lastUse==absent||(member.firstWrite!=absent&&member.firstWrite>member.lastUse)||
           (member.firstRead!=absent&&member.firstRead>member.lastUse))
            throw std::invalid_argument("Invalid scratch alias member");
        storage[member.storage].push_back(&member);
    }
    std::vector<ScratchAliasGroup> result;
    for(const auto& entry:storage) {
        const auto& group=entry.second;const auto& first=*group.front();
        size_t representative=absent,firstRead=absent,firstWrite=absent,lastUse=0;
        bool allowed=true;
        for(const auto* member:group) {
            allowed=allowed&&member->allowed&&member->rows==first.rows&&member->columns==first.columns;
            representative=std::min(representative,member->slot);
            firstRead=std::min(firstRead,member->firstRead);
            firstWrite=std::min(firstWrite,member->firstWrite);
            lastUse=std::max(lastUse,member->lastUse);
        }
        if(!allowed||firstWrite==absent||firstRead<=firstWrite)continue;
        // The initializing writer must also run in resident execution. A full
        // standalone-only write cannot initialize a later partial resident write.
        for(const auto* member:group)if(member->firstWrite==firstWrite)
            allowed=allowed&&member->completeFirstWrite&&member->residentFirstWrite;
        if(!allowed)continue;
        ScratchAliasGroup planned{{representative,first.rows,first.columns,firstWrite,lastUse},{}};
        for(const auto* member:group)planned.slots.push_back(member->slot);
        std::sort(planned.slots.begin(),planned.slots.end());
        result.push_back(std::move(planned));
    }
    std::sort(result.begin(),result.end(),[](const auto& a,const auto& b){return a.lifetime.slot<b.lifetime.slot;});
    return result;
}
}
