// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>
namespace Garnet {
inline std::vector<size_t> ChunkAttentionContextBuckets(size_t capacity) {
    if(!capacity||capacity>4096)throw std::invalid_argument("Unsupported attention context envelope");
    std::vector<size_t> result;
    for(size_t bound=std::min(size_t(128),capacity);;bound=std::min(bound*2,capacity)) {
        result.push_back(bound);if(bound==capacity)return result;
    }
}
struct ChunkAttentionPolicy {
    size_t capacity=0,activeTokens=0;
    bool known=false,scalar=false;
    // Physical request-row ranges, never compacted KV owners or token rows.
    std::vector<std::pair<size_t,size_t>> activeRequestSpans;
};
// Only completed host-readable control buffers may be supplied. GPU-produced
// pending controls use null pointers and retain the maximum compiled extent.
inline ChunkAttentionPolicy SelectChunkAttentionPolicy(size_t capacity,size_t batch,size_t chunk,
    const float* slots,size_t slotStride,const float* active,size_t activeStride,size_t maximumChunk=256) {
    const auto buckets=ChunkAttentionContextBuckets(capacity);
    if(!maximumChunk||maximumChunk>512||!chunk||chunk>maximumChunk)
        throw std::invalid_argument("Invalid compiled attention chunk bound");
    if(!slots&&!active)return {capacity,0,false,false};
    if(!slots||!active||!batch||batch>64||slotStride<chunk||activeStride<chunk)
        throw std::invalid_argument("Invalid attention control layout");
    size_t visible=1,count=0;std::vector<std::pair<size_t,size_t>> spans;
    for(size_t b=0;b<batch;++b) {
      const size_t before=count;
      for(size_t t=0;t<chunk;++t) {
        const float flag=active[b*activeStride+t],position=slots[b*slotStride+t];
        if(!std::isfinite(flag)||(flag!=0&&flag!=1)||!std::isfinite(position)||position<0||std::floor(position)!=position||
            (flag!=0&&position>=float(capacity)))throw std::invalid_argument("Invalid attention slot or active flag");
        if(flag!=0){++count;visible=std::max(visible,size_t(position)+1);}
      }
      if(count!=before) {
        if(!spans.empty()&&spans.back().first+spans.back().second==b)++spans.back().second;
        else spans.emplace_back(b,1);
      }
    }
    return {*std::lower_bound(buckets.begin(),buckets.end(),visible),count,true,count<=8,std::move(spans)};
}
}
