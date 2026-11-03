// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <vector>

namespace Garnet {
struct KVSequencePlacement {
    int requestRow=-1;
    uint64_t rowEpoch=0;
};
// Scheduling consumes logical capacity and placement, not device allocation
// APIs. Paged CUDA and fixed-row hybrid backends retain their storage owners.
class KVSequenceAllocator {
public:
    virtual ~KVSequenceAllocator()=default;
    virtual bool Reserve(uint64_t sequence,int capacity)=0;
    virtual bool EnsureCapacity(uint64_t sequence,int capacity)=0;
    virtual bool CommitLength(uint64_t sequence,int length)=0;
    virtual bool Release(uint64_t sequence)=0;
    virtual int LogicalLength(uint64_t sequence) const=0;
    virtual std::vector<int> PageTable(uint64_t sequence) const=0;
    virtual KVSequencePlacement Placement(uint64_t) const {return {};}
};
}
