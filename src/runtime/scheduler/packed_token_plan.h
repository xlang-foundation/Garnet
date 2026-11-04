// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "continuous_batch_scheduler.h"
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace Garnet {
// Metadata for a future layer executor. Building this plan does not execute
// projections or acknowledge scheduler progress. Leases must be checked again
// by the executor immediately before touching mutable device state.
struct PackedTokenSegment {
    ScheduledRequest request;
    RequestStage stage=RequestStage::Queued;
    int packedOffset=0;
    int tokenCount=0;
    int absolutePosition=0;
};
struct PackedTokenPlan {
    std::vector<PackedTokenSegment> segments;
    int tokenCount=0;
};

inline PackedTokenPlan BuildPackedTokenPlan(const SchedulerTick& tick,
    const KVSequenceAllocator& storage,int maximumTokens,int contextCapacity) {
    if(maximumTokens<=0||contextCapacity<=0)
        throw std::invalid_argument("Invalid packed token capacity");
    PackedTokenPlan result;
    std::unordered_set<std::uint64_t> requests;
    std::unordered_set<int> rows;
    // Decode first is an execution convention, independent of tick ordering.
    for(auto stage:{RequestStage::Decode,RequestStage::Prefill}) {
        for(const auto& batch:tick.batches) {
            if(batch.stage!=stage)continue;
            for(const auto& request:batch.requests) {
                const int count=stage==RequestStage::Decode?1:request.prefillTokens;
                const int position=stage==RequestStage::Decode?
                    request.slotPosition:request.prefillOffset;
                const auto placement=storage.Placement(request.id);
                if(!request.id||request.requestRow<0||!request.rowEpoch||
                    placement.requestRow!=request.requestRow||placement.rowEpoch!=request.rowEpoch||
                    !requests.insert(request.id).second||!rows.insert(request.requestRow).second)
                    throw std::invalid_argument("Invalid or repeated packed row lease");
                if(count<=0||position<0||position>contextCapacity||count>contextCapacity-position||
                    count>maximumTokens-result.tokenCount||
                    request.contextLength!=position+count||request.slotPosition!=position||
                    storage.LogicalLength(request.id)!=position)
                    throw std::invalid_argument("Invalid packed token range");
                if(stage==RequestStage::Prefill&&
                    (request.promptTokens<position||count>request.promptTokens-position))
                    throw std::invalid_argument("Packed prefill exceeds prompt");
                // Position deltas can be negative for multimodal RoPE. Check
                // representability without imposing text-only semantics.
                const auto absolute=std::int64_t(position)+request.positionDelta;
                if(absolute<std::numeric_limits<int>::min()||absolute>std::numeric_limits<int>::max())
                    throw std::invalid_argument("Packed absolute position overflow");
                result.segments.push_back({request,stage,result.tokenCount,count,int(absolute)});
                result.tokenCount+=count;
            }
        }
    }
    for(const auto& batch:tick.batches)
        if(batch.stage!=RequestStage::Decode&&batch.stage!=RequestStage::Prefill)
            throw std::invalid_argument("Unsupported packed stage");
    return result;
}
}
