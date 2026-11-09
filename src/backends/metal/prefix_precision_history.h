// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <algorithm>
#include <cstddef>
#include <stdexcept>
namespace Garnet {
// Producer history only; this never slices a recurrent snapshot or asserts
// numerical equality between different prefill shapes/algorithms.
class PrefixPrecisionHistory {
    size_t total_=0,prefill_=0;
public:
    static constexpr size_t kMaximumTokens=4096;
    void Validate(size_t previous) const {
        if(previous!=total_||prefill_>total_||total_>kMaximumTokens)
            throw std::logic_error("Native prefix producer history differs from tracked tokens");
    }
    void Append(size_t previous,size_t count,bool prefill) {
        Validate(previous);
        if(count>kMaximumTokens-total_)throw std::invalid_argument("Prefix producer history exceeds context profile");
        if(prefill&&prefill_==total_)prefill_+=count;
        total_+=count;
    }
    void RestorePure(size_t count) {
        if(count>kMaximumTokens)throw std::invalid_argument("Restored prefix producer history exceeds profile");
        total_=prefill_=count;
    }
    void Truncate(size_t count) {
        if(count>total_)throw std::invalid_argument("Cannot grow prefix producer history by truncation");
        prefill_=std::min(prefill_,count);total_=count;
    }
    void Reset(){total_=prefill_=0;}
    size_t PrefillTokens() const {return prefill_;}
    size_t TotalTokens() const {return total_;}
    bool CaptureAllowed(size_t previous) const {Validate(previous);return total_!=0&&prefill_==total_;}
};
}
