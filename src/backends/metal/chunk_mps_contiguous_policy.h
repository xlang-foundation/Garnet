// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cmath>
#include <cstddef>
namespace Garnet {
// Call only for completed, shared, nonpending controls. Unknown GPU controls
// must retain generic selection; this helper never synchronizes or reads them.
inline bool ChunkMPSContiguousPrefix(size_t batch,size_t chunk,size_t capacity,
    const float* slots,size_t slotStride,const float* active,size_t activeStride) {
    if(!batch||batch>4||!chunk||chunk>512||!capacity||capacity>4096||
       !slots||!active||slotStride<chunk||activeStride<chunk)return false;
    for(size_t b=0;b<batch;++b) {
        size_t count=0;
        while(count<chunk&&active[b*activeStride+count]!=0) {
            if(!std::isfinite(active[b*activeStride+count]))return false;
            ++count;
        }
        for(size_t t=count;t<chunk;++t)if(active[b*activeStride+t]!=0)return false;
        if(!count)continue;
        float first=slots[b*slotStride];
        if(count>capacity||!std::isfinite(first)||first<0||first!=std::floor(first)||
           double(first)+double(count)>double(capacity))return false;
        for(size_t t=0;t<count;++t)
            if(slots[b*slotStride+t]!=first+float(t))return false;
    }
    return true;
}
}
