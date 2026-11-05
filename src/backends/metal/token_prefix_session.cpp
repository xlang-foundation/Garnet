// SPDX-License-Identifier: Apache-2.0
#include "decoder_session.h"
#ifdef GARNET_METAL_TOKEN_PREFIX_CACHE
#include <algorithm>
#include <cmath>
#include <climits>
#include <sstream>
#include <stdexcept>
namespace Garnet {
namespace {
struct HybridRowSnapshot final : PrefixSnapshot {
    std::vector<std::array<std::vector<float>,2>> states;
    std::vector<float> hidden;
    HybridRowSnapshot(PrefixCacheIdentity identity,size_t tokens,size_t bytes)
        :PrefixSnapshot(std::move(identity),tokens,bytes) {}
};
std::vector<int64_t> Tokens(X::Value value,X3PackageHost* host) {
    if(value.runtime()!=host->runtime||!value.IsList())throw std::invalid_argument("Native prefix requires integer token IDs");
    std::vector<int64_t> result;result.reserve(value.Size());
    for(uint64_t i=0;i<value.Size();++i) {
        auto token=value.Get(i);
        if(!token.IsInt64()||token.ToInt64()<0||token.ToInt64()>INT_MAX)
            throw std::invalid_argument("Native prefix token ID out of range");
        result.push_back(token.ToInt64());
    }
    return result;
}
}
void MetalDecoderSession::ValidateTokenOwners(std::vector<std::unique_lock<std::mutex>>& locks) {
    std::vector<MetalCompiledGraph*> graphs;
    for(size_t i=0;i<states_.size();++i)graphs.push_back(owners_[i].NativeData<MetalCompiledGraph>());
    if(head_)graphs.push_back(head_);
    auto ordered=graphs;std::sort(ordered.begin(),ordered.end(),std::less<MetalCompiledGraph*>());
    if(std::adjacent_find(ordered.begin(),ordered.end())!=ordered.end())throw std::logic_error("Duplicate token-prefix owners");
    for(auto graph:ordered)locks.emplace_back(graph->mutex_);
    for(size_t i=0;i<states_.size();++i)if(!graphs[i]->ready_||graphs[i]->generation_!=states_[i].tokenOwnerGeneration)
        throw std::logic_error("Native token-prefix layer owner was closed or rebuilt");
    if(!head_||!head_->ready_||head_->generation_!=headGeneration_)
        throw std::logic_error("Native token-prefix head owner was closed or rebuilt");
}
bool MetalDecoderSession::ConfigureTokenPrefix(X::Value identity,int64_t maxEntries,int64_t maxBytes) {
    std::lock_guard<std::mutex> lock(mutex_);
#ifdef GARNET_METAL_MTP_BRIDGE
    if(mtp_)throw std::invalid_argument("MTP auxiliary prefix snapshots are not prepared");
#endif
    if(!gpu_||poisoned_||!head_||!requestRows_||tokenCache_||maxEntries<1||maxEntries>256||maxBytes<1024||maxBytes>512LL*1024*1024||
       identity.runtime()!=Host()->runtime||!identity.IsList()||identity.Size()!=6)
        throw std::invalid_argument("Token-prefix configuration requires a ready session, six identity strings and bounded host budget");
    for(size_t row=0;row<states_[0].rows;++row)if(requestRows_->Owned(row)||(finalHiddenReady_.size()>row&&finalHiddenReady_[row]))
        throw std::invalid_argument("Configure token prefixes before executing or claiming requests");
    std::array<std::string,6> fields;
    for(size_t i=0;i<6;++i) {
        auto value=identity.Get(i);
        if(!value.IsString()||value.ToString().empty()||value.ToString().size()>4096)
            throw std::invalid_argument("Incomplete or excessive token-prefix identity");
        fields[i]=value.ToString();
    }
    if(fields[3]!="text_zero_based_3axis_v1"||fields[5]!="text")
        throw std::invalid_argument("Initial hybrid token-prefix profile requires plain text positions");
    std::vector<std::unique_lock<std::mutex>> locks;ValidateTokenOwners(locks);
    size_t capacity=SIZE_MAX;
    for(const auto& state:states_)for(size_t part=0;part<2;++part)if(state.tokenCompact[part]) {
        const auto& shape=state.shape[part];
        if(shape.size()!=4||shape[2]<1)throw std::invalid_argument("Token-prefix KV layout is not bounded");
        capacity=std::min(capacity,size_t(shape[2]));
    }
    if(capacity==SIZE_MAX||capacity>4096)throw std::invalid_argument("Token-prefix profile requires proved bounded KV capacity up to4096");
    tokenCapacity_=capacity;
    tokenIdentity_={fields[0],fields[1],fields[2],fields[3],fields[4],fields[5]};
    tokenCache_=std::make_unique<PrefixCache>(size_t(maxEntries),size_t(maxBytes));tokenCacheBudget_=size_t(maxBytes);
    return true;
}
X::Value MetalDecoderSession::RunTokens(X::Value graphs,X::Value hidden,X::Value positions,X::Value slots,
    X::Value active,int64_t predictionToken,X::Value epochs,X::Value tokens) {
    std::lock_guard<std::mutex> lock(mutex_);
    if(!gpu_||poisoned_||!tokenCache_||!requestRows_||epochs.runtime()!=Host()->runtime||
       !epochs.IsList()||epochs.Size()!=states_[0].rows)
        throw std::invalid_argument("Tracked execution requires configured token prefixes and row leases");
    auto ids=Tokens(tokens,Host());std::vector<uint64_t> expected;
    for(uint64_t row=0;row<epochs.Size();++row) {
        auto value=epochs.Get(row);
        if(!value.IsInt64()||value.ToInt64()<0)throw std::invalid_argument("Invalid tracked row epoch");
        auto epoch=uint64_t(value.ToInt64());
        if(epoch&&!requestRows_->Matches(size_t(row),epoch))throw std::invalid_argument("Stale tracked request epoch");
        expected.push_back(epoch);
    }
    return RunUnlocked(graphs,hidden,positions,slots,active,predictionToken,&expected,&ids);
}
int64_t MetalDecoderSession::TrackedTokenCount(int64_t row,int64_t epoch) {
    std::lock_guard<std::mutex> lock(mutex_);
    if(!gpu_||poisoned_||!tokenCache_||!requestRows_||row<0||epoch<=0||size_t(row)>=rowTokens_.size()||
       size_t(row)>=finalHiddenReady_.size()||!requestRows_->Matches(size_t(row),uint64_t(epoch))||
       !rowTokensValid_[size_t(row)]||!finalHiddenReady_[size_t(row)])return -1;
    return int64_t(rowTokens_[size_t(row)].size());
}
bool MetalDecoderSession::CacheTokenPrefix(int64_t row,int64_t epoch) {
    std::lock_guard<std::mutex> lock(mutex_);
    if(!gpu_||poisoned_||!tokenCache_||row<0||epoch<=0||!requestRows_->Matches(size_t(row),uint64_t(epoch))||
       !rowTokensValid_[size_t(row)]||rowTokens_[size_t(row)].empty()||!finalHiddenReady_[size_t(row)])
        throw std::invalid_argument("Capture requires a completed tracked prefix and current request epoch");
    std::vector<std::unique_lock<std::mutex>> locks;ValidateTokenOwners(locks);
    const size_t tokens=rowTokens_[size_t(row)].size();
    size_t bytes=sizeof(HybridRowSnapshot)+states_.size()*sizeof(std::array<std::vector<float>,2>);
    auto charge=[&](size_t count) {
        if(bytes>tokenCacheBudget_||count>(tokenCacheBudget_-bytes)/sizeof(float))return false;
        bytes+=count*sizeof(float);return true;
    };
    if(!charge(rowFinalWidth_))return false;
    for(const auto& state:states_)for(size_t part=0;part<2;++part) {
        size_t count=state.columns[part];
        if(state.tokenCompact[part]) {
            const auto& shape=state.shape[part];
            if(shape.size()!=4||tokens>size_t(shape[2]))throw std::invalid_argument("Captured token boundary exceeds native KV capacity");
            count=size_t(shape[1])*tokens*size_t(shape[3]);
        }
        if(!charge(count))return false;
    }
    auto snapshot=std::make_shared<HybridRowSnapshot>(tokenIdentity_,tokens,bytes);
    snapshot->states.reserve(states_.size());
    for(const auto& state:states_) {
        std::array<std::vector<float>,2> layer;
        for(size_t part=0;part<2;++part) {
            const auto& shape=state.shape[part];
            layer[part]=state.tokenCompact[part]?gpu_->ReadRowPrefixGroups(state.matrix[part],size_t(row),size_t(shape[1]),size_t(shape[2]),size_t(shape[3]),tokens):
                gpu_->ReadRow(state.matrix[part],size_t(row));
        }
        snapshot->states.push_back(std::move(layer));
    }
    snapshot->hidden=gpu_->ReadRow(rowFinalHidden_,size_t(row));
    return tokenCache_->Insert(rowTokens_[size_t(row)],std::move(snapshot));
}
int64_t MetalDecoderSession::RestoreTokenPrefix(int64_t row,int64_t epoch,X::Value tokens) {
    std::lock_guard<std::mutex> lock(mutex_);
    if(!gpu_||poisoned_||!tokenCache_||row<0||epoch<=0||!requestRows_->Matches(size_t(row),uint64_t(epoch))||
       !rowTokensValid_[size_t(row)]||!rowTokens_[size_t(row)].empty()||
       (size_t(row)<finalHiddenReady_.size()&&finalHiddenReady_[size_t(row)]))
        throw std::invalid_argument("Prefix restore requires a fresh current request row");
    auto ids=Tokens(tokens,Host());
    if(ids.size()>tokenCapacity_)throw std::invalid_argument("Prefix query exceeds the native context profile");
    std::vector<std::unique_lock<std::mutex>> locks;ValidateTokenOwners(locks);
    auto hit=tokenCache_->Lookup(tokenIdentity_,ids);if(!hit)return 0;
    auto snapshot=std::dynamic_pointer_cast<const HybridRowSnapshot>(hit.snapshot);
    if(!snapshot||snapshot->states.size()!=states_.size())throw std::logic_error("Hybrid snapshot type/layer mismatch");
    const size_t width=size_t(head_->inputShapes_[0].back());
    if(snapshot->hidden.size()!=width)throw std::logic_error("Hybrid snapshot hidden width mismatch");
    for(size_t layer=0;layer<states_.size();++layer)for(size_t part=0;part<2;++part) {
        const auto& state=states_[layer];const auto& shape=state.shape[part];
        const size_t count=state.tokenCompact[part]?size_t(shape[1])*hit.tokens*size_t(shape[3]):state.columns[part];
        if(snapshot->states[layer][part].size()!=count)throw std::logic_error("Hybrid snapshot state layout mismatch");
        gpu_->ValidateCompletedRow(state.matrix[part],size_t(row));
    }
    std::vector<int64_t> history(ids.begin(),ids.begin()+hit.tokens);
    if(!rowFinalHidden_.Valid()) {
        auto bank=gpu_->Upload(std::vector<float>(states_[0].rows*width,0),states_[0].rows,width);
        finalHiddenReady_.assign(states_[0].rows,0);rowFinalHidden_=std::move(bank);rowFinalWidth_=width;
    }
    gpu_->ValidateCompletedRow(rowFinalHidden_,size_t(row));
    ResetRowUnlocked(row);poisoned_=true;
    try {
        for(size_t layer=0;layer<states_.size();++layer)for(size_t part=0;part<2;++part) {
            const auto& state=states_[layer];const auto& shape=state.shape[part];const auto& data=snapshot->states[layer][part];
            if(state.tokenCompact[part])gpu_->WriteRowPrefixGroups(state.matrix[part],size_t(row),size_t(shape[1]),size_t(shape[2]),size_t(shape[3]),hit.tokens,data);
            else gpu_->WriteRow(state.matrix[part],size_t(row),data);
        }
        gpu_->WriteRow(rowFinalHidden_,size_t(row),snapshot->hidden);
        rowTokens_[size_t(row)]=std::move(history);rowTokensValid_[size_t(row)]=1;finalHiddenReady_[size_t(row)]=1;
        poisoned_=false;return int64_t(hit.tokens);
    } catch(...) {throw;}
}
std::string MetalDecoderSession::TokenPrefixStatsJson() {
    std::lock_guard<std::mutex> lock(mutex_);
    if(!tokenCache_)throw std::logic_error("Token prefix cache is not configured");
    const auto stats=tokenCache_->Stats();std::ostringstream out;
    out<<"{\"entries\":"<<stats.entries<<",\"accounted_host_bytes\":"<<stats.accountedBytes
       <<",\"hits\":"<<stats.hits<<",\"misses\":"<<stats.misses<<",\"evictions\":"<<stats.evictions
       <<",\"replacements\":"<<stats.replacements<<"}";return out.str();
}
bool MetalDecoderSession::ClearTokenPrefixes() {
    std::lock_guard<std::mutex> lock(mutex_);
    if(!tokenCache_)throw std::logic_error("Token prefix cache is not configured");
    tokenCache_->Clear();return true;
}
}
#endif
