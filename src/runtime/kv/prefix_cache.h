// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace Garnet {
struct PrefixCacheIdentity {
    std::string model,executionProfile,tokenizer,positionPolicy,isolation,multimodal;
    bool operator==(const PrefixCacheIdentity& other) const {
        return model==other.model&&executionProfile==other.executionProfile&&
            tokenizer==other.tokenizer&&positionPolicy==other.positionPolicy&&
            isolation==other.isolation&&multimodal==other.multimodal;
    }
};

// Inference backends supply immutable owning snapshots, not borrowed live row
// views. Hybrid snapshots include convolution/recurrent state at this exact
// boundary. Exact-prompt hits also need final hidden state to resume the head.
class PrefixSnapshot {
    PrefixCacheIdentity identity_;
    size_t tokens_,retainedBytes_;
protected:
    PrefixSnapshot(PrefixCacheIdentity identity,size_t tokens,size_t retainedBytes)
        :identity_(std::move(identity)),tokens_(tokens),retainedBytes_(retainedBytes) {
        if(identity_.model.empty()||identity_.executionProfile.empty()||identity_.tokenizer.empty()||
           identity_.positionPolicy.empty()||!tokens||!retainedBytes)
            throw std::invalid_argument("Incomplete native prefix snapshot identity/state");
    }
public:
    virtual ~PrefixSnapshot()=default;
    const PrefixCacheIdentity& Identity() const {return identity_;}
    size_t Tokens() const {return tokens_;}
    size_t RetainedBytes() const {return retainedBytes_;}
};

struct PrefixCacheHit {
    size_t tokens=0;
    std::shared_ptr<const PrefixSnapshot> snapshot;
    explicit operator bool() const {return bool(snapshot);}
};
struct PrefixCacheStats {
    size_t entries=0,accountedBytes=0;
    uint64_t hits=0,misses=0,evictions=0,replacements=0;
};

// Backend-independent inference policy. Matching is exact, so hash collisions
// cannot alias state. It selects the longest STORED checkpoint, not an arbitrary
// truncation of a later recurrent state. Backend leases retain KV storage after
// request completion; this index never treats a request ID as cache identity.
// Integration must still supply those leases and native capture/restore hooks.
class PrefixCache {
    struct Entry {
        std::vector<int64_t> tokens;
        std::shared_ptr<const PrefixSnapshot> snapshot;
        size_t charge;
    };
    size_t maxEntries_,maxBytes_;
    std::list<Entry> entries_; // LRU first, MRU last.
    PrefixCacheStats stats_;
    mutable std::mutex mutex_;

    bool Charge(const std::vector<int64_t>& tokens,const PrefixSnapshot& state,size_t& bytes) const {
        bytes=state.RetainedBytes();
        if(bytes>maxBytes_||tokens.size()>SIZE_MAX/sizeof(int64_t))return false;
        auto add=[&](size_t count){if(count>maxBytes_-bytes)return false;bytes+=count;return true;};
        if(!add(sizeof(Entry)+2*sizeof(void*)+sizeof(PrefixSnapshot))||
           !add(tokens.size()*sizeof(int64_t)))return false;
        const auto& id=state.Identity();
        for(const auto* key:{&id.model,&id.executionProfile,&id.tokenizer,&id.positionPolicy,&id.isolation,&id.multimodal})
            if(!add(key->size()))return false;
        return true;
    }
public:
    PrefixCache(size_t maxEntries,size_t maxBytes):maxEntries_(maxEntries),maxBytes_(maxBytes) {}
    PrefixCache(const PrefixCache&)=delete;
    PrefixCache& operator=(const PrefixCache&)=delete;

    PrefixCacheHit Lookup(const PrefixCacheIdentity& identity,const std::vector<int64_t>& tokens) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto best=entries_.end();
        for(auto entry=entries_.begin();entry!=entries_.end();++entry) {
            if(entry->tokens.size()>tokens.size()||
               (best!=entries_.end()&&entry->tokens.size()<=best->tokens.size())||
               !(entry->snapshot->Identity()==identity))continue;
            if(std::equal(entry->tokens.begin(),entry->tokens.end(),tokens.begin()))best=entry;
        }
        if(best==entries_.end()){++stats_.misses;return {};}
        PrefixCacheHit hit{best->tokens.size(),best->snapshot};
        entries_.splice(entries_.end(),entries_,best);++stats_.hits;return hit;
    }

    bool Insert(const std::vector<int64_t>& tokens,std::shared_ptr<const PrefixSnapshot> snapshot) {
        if(!snapshot||tokens.empty()||snapshot->Tokens()!=tokens.size())
            throw std::invalid_argument("Native prefix tokens do not match captured state boundary");
        size_t charge;
        if(!maxEntries_||!Charge(tokens,*snapshot,charge))return false;
        // Allocate before eviction. Drop retired backend owners after unlocking,
        // avoiding cache/allocator lock inversion in snapshot destructors.
        std::list<Entry> candidate,retired;
        candidate.push_back({tokens,std::move(snapshot),charge});
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto duplicate=std::find_if(entries_.begin(),entries_.end(),[&](const Entry& entry){
                return entry.tokens==tokens&&entry.snapshot->Identity()==candidate.front().snapshot->Identity();
            });
            if(duplicate!=entries_.end()) {
                stats_.accountedBytes-=duplicate->charge;
                retired.splice(retired.end(),entries_,duplicate);++stats_.replacements;
            }
            while(!entries_.empty()&&(entries_.size()>=maxEntries_||stats_.accountedBytes>maxBytes_-charge)) {
                stats_.accountedBytes-=entries_.front().charge;
                retired.splice(retired.end(),entries_,entries_.begin());++stats_.evictions;
            }
            entries_.splice(entries_.end(),candidate);stats_.accountedBytes+=charge;
        }
        return true;
    }

    void Clear() {
        std::list<Entry> retired;
        {std::lock_guard<std::mutex> lock(mutex_);retired.splice(retired.end(),entries_);stats_.accountedBytes=0;}
        // Outstanding hits remain valid until their inference owners release them.
    }
    PrefixCacheStats Stats() const {
        std::lock_guard<std::mutex> lock(mutex_);auto result=stats_;result.entries=entries_.size();return result;
    }
};
}
