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
#include <unordered_map>

namespace Garnet {
struct PrefixCacheIdentity {
    std::string model,executionProfile,tokenizer,positionPolicy,isolation,multimodal;
    bool operator==(const PrefixCacheIdentity& other) const {
        return model==other.model&&executionProfile==other.executionProfile&&
            tokenizer==other.tokenizer&&positionPolicy==other.positionPolicy&&
            isolation==other.isolation&&multimodal==other.multimodal;
    }
};

struct PrefixSharedResource {const void* identity=nullptr;size_t bytes=0;};

// Inference backends supply immutable owning snapshots, not borrowed live row
// views. Hybrid snapshots include convolution/recurrent state at this exact
// boundary. Exact-prompt hits also need final hidden state to resume the head.
class PrefixSnapshot {
    PrefixCacheIdentity identity_;
    size_t tokens_,retainedBytes_;
protected:
    void FinalizeRetainedBytes(size_t bytes) {
        if(!bytes)throw std::invalid_argument("Empty private prefix payload");
        retainedBytes_=bytes;
    }
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
    // Opaque identities name immutable owning resources. Only the shared
    // accounting mode accepts these; retained bytes then describe private data.
    virtual const std::vector<PrefixSharedResource>& SharedResources() const {
        static const std::vector<PrefixSharedResource> empty;return empty;
    }
};

struct PrefixCacheHit {
    size_t tokens=0;
    std::shared_ptr<const PrefixSnapshot> snapshot;
    explicit operator bool() const {return bool(snapshot);}
};
// Resource-reuse hint only: commonTokens may be inside a later snapshot.
// It must never be used as a partially restored recurrent checkpoint.
struct PrefixCacheReuseHint {
    size_t commonTokens=0;std::shared_ptr<const PrefixSnapshot> snapshot;
};
struct PrefixCacheStats {
    size_t entries=0,accountedBytes=0,privateBytes=0,resourceBytes=0,resourceCount=0;
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
    bool sharedAccounting_=false;
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
    bool SharedCharge(const std::vector<const Entry*>& entries,size_t& total,size_t& privateBytes,size_t& resources,size_t& resourceBytes) const {
        total=privateBytes=resources=resourceBytes=0;
        std::unordered_map<const void*,size_t> unique;
        auto add=[&](size_t bytes){if(bytes>maxBytes_-total)return false;total+=bytes;return true;};
        for(const auto* entry:entries) {
            if(!add(entry->charge))return false;privateBytes+=entry->charge;
            for(const auto& resource:entry->snapshot->SharedResources()) {
                if(!resource.identity||!resource.bytes)throw std::invalid_argument("Invalid shared prefix resource");
                auto old=unique.find(resource.identity);
                if(old!=unique.end()) {
                    if(old->second!=resource.bytes)throw std::invalid_argument("Inconsistent shared prefix resource charge");
                }else {
                    if(!add(resource.bytes))return false;
                    unique.emplace(resource.identity,resource.bytes);resourceBytes+=resource.bytes;
                }
            }
        }
        resources=unique.size();return true;
    }
    bool InsertShared(const std::vector<int64_t>& tokens,std::shared_ptr<const PrefixSnapshot> snapshot) {
        size_t charge;if(!maxEntries_||!Charge(tokens,*snapshot,charge))return false;
        std::list<Entry> candidate,retired;candidate.push_back({tokens,std::move(snapshot),charge});
        size_t total,privateBytes,resources,resourceBytes;
        if(!SharedCharge({&candidate.front()},total,privateBytes,resources,resourceBytes))return false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto duplicate=std::find_if(entries_.begin(),entries_.end(),[&](const Entry& entry){
                return entry.tokens==tokens&&entry.snapshot->Identity()==candidate.front().snapshot->Identity();
            });
            std::vector<const Entry*> keep;keep.reserve(entries_.size()+1);
            for(const auto& entry:entries_)if(&entry!=(duplicate==entries_.end()?nullptr:&*duplicate))keep.push_back(&entry);
            keep.push_back(&candidate.front());size_t remove=0;
            while(keep.size()>maxEntries_||!SharedCharge(keep,total,privateBytes,resources,resourceBytes)) {
                if(keep.size()==1)return false;
                keep.erase(keep.begin());++remove;
            }
            // All allocations/validation above precede index mutation. Retired
            // backend owners are released after the cache mutex is unlocked.
            if(duplicate!=entries_.end()){retired.splice(retired.end(),entries_,duplicate);++stats_.replacements;}
            while(remove){retired.splice(retired.end(),entries_,entries_.begin());++stats_.evictions;--remove;}
            entries_.splice(entries_.end(),candidate);
            stats_.accountedBytes=total;stats_.privateBytes=privateBytes;
            stats_.resourceBytes=resourceBytes;stats_.resourceCount=resources;
        }
        return true;
    }
public:
    PrefixCache(size_t maxEntries,size_t maxBytes,bool sharedAccounting=false)
        :maxEntries_(maxEntries),maxBytes_(maxBytes),sharedAccounting_(sharedAccounting) {}
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

    PrefixCacheReuseHint FindCommonPrefix(const PrefixCacheIdentity& identity,const std::vector<int64_t>& tokens) const {
        std::lock_guard<std::mutex> lock(mutex_);PrefixCacheReuseHint hint;
        for(const auto& entry:entries_) {
            if(!(entry.snapshot->Identity()==identity))continue;
            size_t common=0,limit=std::min(tokens.size(),entry.tokens.size());
            while(common<limit&&tokens[common]==entry.tokens[common])++common;
            if(common&&common>=hint.commonTokens)hint={common,entry.snapshot};
        }
        return hint; // No hit/miss or LRU changes for maintenance.
    }

    // Maintenance is not a request lookup: refresh an exact owning checkpoint
    // without changing hit/miss counters or accepting a shorter prefix.
    bool Touch(const PrefixCacheIdentity& identity,const std::vector<int64_t>& tokens) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto entry=std::find_if(entries_.begin(),entries_.end(),[&](const Entry& value){
            return value.tokens==tokens&&value.snapshot->Identity()==identity;
        });
        if(entry==entries_.end())return false;
        entries_.splice(entries_.end(),entries_,entry);return true;
    }

    bool Insert(const std::vector<int64_t>& tokens,std::shared_ptr<const PrefixSnapshot> snapshot) {
        if(!snapshot||tokens.empty()||snapshot->Tokens()!=tokens.size())
            throw std::invalid_argument("Native prefix tokens do not match captured state boundary");
        if(sharedAccounting_)return InsertShared(tokens,std::move(snapshot));
        if(!snapshot->SharedResources().empty())throw std::invalid_argument("Shared resources require shared cache accounting");
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
        {std::lock_guard<std::mutex> lock(mutex_);retired.splice(retired.end(),entries_);stats_.accountedBytes=stats_.privateBytes=stats_.resourceBytes=stats_.resourceCount=0;}
        // Outstanding hits remain valid until their inference owners release them.
    }
    PrefixCacheStats Stats() const {
        std::lock_guard<std::mutex> lock(mutex_);auto result=stats_;result.entries=entries_.size();if(!sharedAccounting_)result.privateBytes=result.accountedBytes;return result;
    }
};
}
