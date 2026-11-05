// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <atomic>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace Garnet {
struct RequestRowLease {
    size_t row=0;
    uint64_t request=0,epoch=0;
};

// Stable physical row ownership for inference backends with row-local state.
// Policy remains in ContinuousBatchScheduler. Epochs are process-wide and never
// reset on model rebuild, so late cancellation cannot clear a replacement row.
class RequestRowLeases {
    inline static std::atomic<uint64_t> nextEpoch_{1};
    std::vector<RequestRowLease> rows_;
    mutable std::mutex mutex_;
    static uint64_t Epoch() {
        auto value=nextEpoch_.load();
        for(;;) {
            if(value>=uint64_t(std::numeric_limits<int64_t>::max()))
                throw std::overflow_error("Native request row epochs exhausted");
            if(nextEpoch_.compare_exchange_weak(value,value+1))return value;
        }
    }
public:
    explicit RequestRowLeases(size_t rows):rows_(rows) {
        if(!rows)throw std::invalid_argument("Native request rows require positive capacity");
    }
    RequestRowLease Claim(uint64_t request) {
        std::lock_guard<std::mutex> lock(mutex_);
        if(!request)throw std::invalid_argument("Native request identity must be positive");
        size_t free=rows_.size();
        for(size_t row=0;row<rows_.size();++row) {
            if(rows_[row].request==request)throw std::invalid_argument("Native request already owns a row");
            if(!rows_[row].request&&free==rows_.size())free=row;
        }
        if(free==rows_.size())throw std::runtime_error("Native request rows are full");
        return rows_[free]=RequestRowLease{free,request,Epoch()};
    }
    bool Matches(size_t row,uint64_t epoch) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return row<rows_.size()&&rows_[row].request&&rows_[row].epoch==epoch;
    }
    bool Owned(size_t row) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return row<rows_.size()&&rows_[row].request;
    }
    bool Release(size_t row,uint64_t epoch) {
        std::lock_guard<std::mutex> lock(mutex_);
        if(row>=rows_.size()||!rows_[row].request||rows_[row].epoch!=epoch)return false;
        rows_[row]={};return true;
    }
    void Clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        for(auto& row:rows_)row={};
    }
};
}
