// SPDX-License-Identifier: Apache-2.0
#include "decoder_session.h"
#include "token_prefix_profile.h"
#ifdef GARNET_METAL_PREFIX_SHARED_BLOCKS
#include "prefix_shared_blocks.h"
#endif
#ifdef GARNET_METAL_TOKEN_PREFIX_CACHE
#include <algorithm>
#include <cmath>
#include <climits>
#include <sstream>
#include <stdexcept>
namespace Garnet {
namespace {
struct HybridRowSnapshot final : PrefixSnapshot {
#ifdef GARNET_METAL_PREFIX_PREFILL_ONLY
    size_t prefillTokens=0;
#endif
    std::vector<std::array<std::vector<float>,2>> states;
    std::vector<float> hidden;
#ifdef GARNET_METAL_PREFIX_SHARED_BLOCKS
    using Blocks=std::vector<std::shared_ptr<const DensePrefixBlock>>;
    std::vector<std::array<Blocks,2>> dense;
    std::vector<PrefixSharedResource> resources;
    const std::vector<PrefixSharedResource>& SharedResources() const override {return resources;}
    void FinalizePrivate() {
        size_t bytes=sizeof(HybridRowSnapshot);
        auto add=[&](size_t count,size_t width){if(count>(SIZE_MAX-bytes)/width)throw std::overflow_error("Private prefix metadata overflow");bytes+=count*width;};
        add(states.capacity(),sizeof(decltype(states)::value_type));
        for(const auto& layer:states)for(const auto& part:layer)add(part.capacity(),sizeof(float));
        add(dense.capacity(),sizeof(decltype(dense)::value_type));
        for(const auto& layer:dense)for(const auto& part:layer)add(part.capacity(),sizeof(std::shared_ptr<const DensePrefixBlock>));
        add(resources.capacity(),sizeof(PrefixSharedResource));add(hidden.capacity(),sizeof(float));
        FinalizeRetainedBytes(bytes);
    }
#endif
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
    bool trackingOnly=false;
#ifdef GARNET_METAL_DIAGNOSTIC_LAYER_BULK
    trackingOnly=maxEntries==0&&maxBytes==0;
#endif
    if(!gpu_||poisoned_||!head_||!requestRows_||tokenCache_||(!trackingOnly&&(maxEntries<1||maxEntries>256||maxBytes<1024||uint64_t(maxBytes)>kTokenPrefixHostLimitBytes))||
       identity.runtime()!=Host()->runtime||!identity.IsList()||identity.Size()!=6)
        throw std::invalid_argument("Token-prefix configuration requires a ready session, six identity strings and bounded host budget");
#if GARNET_METAL_TOKEN_PREFIX_HOST_MIB > 512
    const auto memory=MetalContext::MemoryStats();
    if(!trackingOnly&&(memory.taskStatus!=0||!TokenPrefixHostMemoryFits(memory.allocated,memory.physical,memory.recommended,uint64_t(maxBytes),2)))
        throw std::runtime_error("Prefix host budget and replacement reserve exceed the task memory envelope");
#endif
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
#if defined(GARNET_METAL_PREFIX_PREFILL_ONLY)||defined(GARNET_METAL_COMMITTED_KV_MPS)||defined(GARNET_METAL_CHUNK_HALF_ATTENTION)||defined(GARNET_METAL_MISTRAL_VERIFY_SPLIT_SCORES)
    const std::string producerDomain=
#ifdef GARNET_METAL_PREFIX_PREFILL_ONLY
        "|native-prefill-snapshots-v1"
#endif
#ifdef GARNET_METAL_COMMITTED_KV_MPS
        "|committed-mps-fp32-v1"
#endif
#ifdef GARNET_METAL_CHUNK_HALF_ATTENTION
        "|fp16-prefill-all-p16-gpupack-v1"
#endif
#ifdef GARNET_METAL_FUSED_MISTRAL_MLP
        "|fused-mistral-mlp-fast-v1"
#endif
#ifdef GARNET_METAL_FUSED_Q4_DECODE
        "|fused-q4-decode-rows4-default-v1"
#endif
#ifdef GARNET_METAL_MISTRAL_VERIFY_SPLIT_SCORES
        "|mistral-verify-split-scores128-default-v1"
#endif
        ;
    if(fields[1].size()>4096-producerDomain.size())throw std::invalid_argument("Prefix execution profile leaves no room for native producer domain");
    fields[1]+=producerDomain;
#endif
    std::vector<std::unique_lock<std::mutex>> locks;ValidateTokenOwners(locks);
    size_t capacity=SIZE_MAX;
    for(const auto& state:states_)for(size_t part=0;part<2;++part)if(state.tokenCompact[part]) {
        const auto& shape=state.shape[part];
        if(shape.size()!=4||shape[2]<1)throw std::invalid_argument("Token-prefix KV layout is not bounded");
        capacity=std::min(capacity,size_t(shape[2]));
    }
    if(capacity==SIZE_MAX||capacity>4096)throw std::invalid_argument("Token-prefix profile requires proved bounded KV capacity up to4096");
    // Prefix keys describe input tokens, whose vocabulary can exceed an
    // untied proposal head's output rows. Resolve the immutable input domain
    // at preparation, without loading another copy of the embedding weights.
    size_t vocabulary=head_->slots_.at(head_->output_).columns;
    std::string embeddingName="model.language_model.embed_tokens.weight";
    auto embedding=head_->FindWeightMetadata(embeddingName);
    if(!embedding) {
        embeddingName="model.embed_tokens.weight";
        embedding=head_->FindWeightMetadata(embeddingName);
    }
    if(embedding) {
        if(embedding->shape.size()!=2||embedding->shape[0]<=0||uint64_t(embedding->shape[0])>INT_MAX)
            throw std::invalid_argument("Token-prefix input embedding vocabulary is invalid");
        size_t width=0;
#ifdef GARNET_METAL_MLX_AFFINE_INT4
        if(embedding->dataType=="U32")
            width=AffineInt4Metadata::Resolve(head_->index_,head_->mlxCheckpoint_,embeddingName).layout.columns;
        else
#endif
        {
            const size_t bytes=embedding->dataType=="F32"?4:
                (embedding->dataType=="BF16"||embedding->dataType=="F16"?2:0);
            if(!bytes||embedding->shape[1]<=0||uint64_t(embedding->shape[1])>SIZE_MAX/bytes||
               uint64_t(embedding->shape[0])>SIZE_MAX/(size_t(embedding->shape[1])*bytes)||
               embedding->dataSize!=size_t(embedding->shape[0])*size_t(embedding->shape[1])*bytes)
                throw std::invalid_argument("Token-prefix input embedding layout is invalid");
            width=size_t(embedding->shape[1]);
        }
        if(width!=size_t(head_->inputShapes_[0].back()))
            throw std::invalid_argument("Token-prefix input embedding width differs from the decoder");
        vocabulary=size_t(embedding->shape[0]);
    }
    if(!vocabulary||vocabulary>INT_MAX)throw std::invalid_argument("Token-prefix vocabulary is not bounded");
    tokenCapacity_=capacity;
    tokenVocabulary_=vocabulary;
    tokenIdentity_={fields[0],fields[1],fields[2],fields[3],fields[4],fields[5]};
    tokenCache_=std::make_unique<PrefixCache>(size_t(maxEntries),size_t(maxBytes)
#ifdef GARNET_METAL_PREFIX_SHARED_BLOCKS
        ,true
#endif
    );tokenCacheBudget_=size_t(maxBytes);
#ifdef GARNET_METAL_PREFIX_DIRECT_CAPTURE
    tokenPrefixComparedBytes_=tokenPrefixCopiedBytes_=tokenPrefixReusedBlocks_=tokenPrefixNewBlocks_=0;
#endif
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
#ifdef GARNET_METAL_PREDICTION_CONFIDENCE
X::Value MetalDecoderSession::RunTokensConfidence(X::Value graphs,X::Value hidden,X::Value positions,X::Value slots,
    X::Value active,int64_t predictionToken,X::Value epochs,X::Value tokens) {
    std::lock_guard<std::mutex> lock(mutex_);
    if(!gpu_||poisoned_||!tokenCache_||!requestRows_||epochs.runtime()!=Host()->runtime||
       !epochs.IsList()||epochs.Size()!=states_[0].rows)
        throw std::invalid_argument("Tracked execution requires configured token prefixes and row leases");
    if(!confidence_.Valid()||predictionToken!=0||X::Tensor(hidden).Info().rank!=2)
        throw std::invalid_argument("Confidence tracked execution requires prepared output, base token rows and prediction index0");
    auto ids=Tokens(tokens,Host());std::vector<uint64_t> expected;
    for(uint64_t row=0;row<epochs.Size();++row) {
        auto value=epochs.Get(row);
        if(!value.IsInt64()||value.ToInt64()<0)throw std::invalid_argument("Invalid tracked row epoch");
        auto epoch=uint64_t(value.ToInt64());
        if(epoch&&!requestRows_->Matches(size_t(row),epoch))throw std::invalid_argument("Stale tracked request epoch");
        expected.push_back(epoch);
    }
    return RunUnlocked(graphs,hidden,positions,slots,active,predictionToken,&expected,&ids,nullptr,true);
}
#endif

int64_t MetalDecoderSession::TrackedTokenCount(int64_t row,int64_t epoch) {
    std::lock_guard<std::mutex> lock(mutex_);
    if(!gpu_||poisoned_||!tokenCache_||!requestRows_||row<0||epoch<=0||size_t(row)>=rowTokens_.size()||
       size_t(row)>=finalHiddenReady_.size()||!requestRows_->Matches(size_t(row),uint64_t(epoch))||
       !rowTokensValid_[size_t(row)]||!finalHiddenReady_[size_t(row)])return -1;
    return int64_t(rowTokens_[size_t(row)].size());
}
#ifdef GARNET_METAL_PREFIX_PREFILL_ONLY
int64_t MetalDecoderSession::TrackedPrefillPrefixTokens(int64_t row,int64_t epoch) {
    std::lock_guard<std::mutex> lock(mutex_);
    if(!gpu_||poisoned_||!tokenCache_||!requestRows_||row<0||epoch<=0||size_t(row)>=rowTokens_.size()||
       !requestRows_->Matches(size_t(row),uint64_t(epoch))||!rowTokensValid_[size_t(row)]||
       size_t(row)>=finalHiddenReady_.size()||!finalHiddenReady_[size_t(row)])return -1;
    tokenPrecisionHistory_.at(size_t(row)).Validate(rowTokens_[size_t(row)].size());
    return int64_t(tokenPrecisionHistory_[size_t(row)].PrefillTokens());
}
#endif
bool MetalDecoderSession::CacheTokenPrefix(int64_t row,int64_t epoch) {
    std::lock_guard<std::mutex> lock(mutex_);
    if(!gpu_||poisoned_||!tokenCache_||row<0||epoch<=0||!requestRows_->Matches(size_t(row),uint64_t(epoch))||
       !rowTokensValid_[size_t(row)]||rowTokens_[size_t(row)].empty()||!finalHiddenReady_[size_t(row)])
        throw std::invalid_argument("Capture requires a completed tracked prefix and current request epoch");
    std::vector<std::unique_lock<std::mutex>> locks;ValidateTokenOwners(locks);
    const size_t tokens=rowTokens_[size_t(row)].size();
#ifdef GARNET_METAL_PREFIX_PREFILL_ONLY
    if(!tokenPrecisionHistory_.at(size_t(row)).CaptureAllowed(tokens)){
        if(tokenCacheBudget_&&tokenPrefillSkippedMixed_!=UINT64_MAX)++tokenPrefillSkippedMixed_;
        return false;
    }
#endif
#ifdef GARNET_METAL_PREFIX_SHARED_BLOCKS
    // Estimate a complete owning candidate before allocation. Only byte-equal
    // blocks are reused, so unrelated FP16/FP32 precision histories cannot alias.
    size_t estimate=sizeof(HybridRowSnapshot),blockCount=0;
    auto add=[&](size_t count,size_t width){if(estimate>tokenCacheBudget_||count>(tokenCacheBudget_-estimate)/width)return false;estimate+=count*width;return true;};
    if(!add(states_.size(),sizeof(std::array<std::vector<float>,2>)+sizeof(std::array<HybridRowSnapshot::Blocks,2>))||!add(rowFinalWidth_,sizeof(float)))return false;
    for(const auto& state:states_)for(size_t part=0;part<2;++part) {
        size_t count=state.columns[part];
        if(state.tokenCompact[part]) {
            const auto& shape=state.shape[part];if(shape.size()!=4||tokens>size_t(shape[2]))throw std::invalid_argument("Captured token boundary exceeds native KV capacity");
            count=PrefixWordCount(size_t(shape[1]),tokens,size_t(shape[3]));
            if(count>128*1024*1024/sizeof(float))return false;
            size_t pages=(tokens+kPrefixBlockTokens-1)/kPrefixBlockTokens;
            if(!add(pages,sizeof(DensePrefixBlock)+2*sizeof(void*)+sizeof(std::shared_ptr<const DensePrefixBlock>)+sizeof(PrefixSharedResource)))return false;
            blockCount+=pages;
        }
        if(!add(count,sizeof(float)))return false;
    }
    const auto memory=MetalContext::MemoryStats();
    if(memory.taskStatus!=0||!TokenPrefixHostMemoryFits(memory.allocated,memory.physical,memory.recommended,estimate,1))return false;
    auto hint=tokenCache_->FindCommonPrefix(tokenIdentity_,rowTokens_[size_t(row)]);
    auto parent=std::dynamic_pointer_cast<const HybridRowSnapshot>(hint.snapshot);
    if(parent&&parent->dense.size()!=states_.size())throw std::logic_error("Shared prefix layer layout mismatch");
    auto snapshot=std::make_shared<HybridRowSnapshot>(tokenIdentity_,tokens,sizeof(HybridRowSnapshot));
#ifdef GARNET_METAL_PREFIX_PREFILL_ONLY
    snapshot->prefillTokens=tokenPrecisionHistory_.at(size_t(row)).PrefillTokens();
#endif
    snapshot->states.resize(states_.size());snapshot->dense.resize(states_.size());snapshot->resources.reserve(blockCount);
    for(size_t layer=0;layer<states_.size();++layer)for(size_t part=0;part<2;++part) {
        const auto& state=states_[layer];const auto& shape=state.shape[part];
        if(!state.tokenCompact[part]){snapshot->states[layer][part]=gpu_->ReadRow(state.matrix[part],size_t(row));continue;}
        size_t groups=size_t(shape[1]),width=size_t(shape[3]);
#ifdef GARNET_METAL_PREFIX_DIRECT_CAPTURE
        const HybridRowSnapshot::Blocks empty;
        auto captured=gpu_->ReadRowPrefixBlocks(state.matrix[part],size_t(row),groups,size_t(shape[2]),width,tokens,
            hint.commonTokens,parent?parent->dense[layer][part]:empty);
        tokenPrefixComparedBytes_+=captured.comparedBytes;tokenPrefixCopiedBytes_+=captured.copiedBytes;
        tokenPrefixReusedBlocks_+=captured.reusedBlocks;tokenPrefixNewBlocks_+=captured.newBlocks;
        auto& blocks=snapshot->dense[layer][part];blocks=std::move(captured.blocks);
        for(const auto& block:blocks)snapshot->resources.push_back({block.get(),block->Charge()});
#else
        auto current=gpu_->ReadRowPrefixGroups(state.matrix[part],size_t(row),groups,size_t(shape[2]),width,tokens);
        auto& blocks=snapshot->dense[layer][part];blocks.reserve((tokens+kPrefixBlockTokens-1)/kPrefixBlockTokens);
        for(size_t first=0,index=0;first<tokens;first+=kPrefixBlockTokens,++index) {
            size_t count=std::min(kPrefixBlockTokens,tokens-first);std::shared_ptr<const DensePrefixBlock> block;
            if(parent&&first+count<=hint.commonTokens&&index<parent->dense[layer][part].size()) {
                auto candidate=parent->dense[layer][part][index];
                if(candidate&&PrefixBlockMatches(*candidate,current,groups,tokens,width,first,count))block=std::move(candidate);
            }
            if(!block)block=MakePrefixBlock(current,groups,tokens,width,first,count);
            snapshot->resources.push_back({block.get(),block->Charge()});blocks.push_back(std::move(block));
        }
#endif
    }
    snapshot->hidden=gpu_->ReadRow(rowFinalHidden_,size_t(row));snapshot->FinalizePrivate();
    return tokenCache_->Insert(rowTokens_[size_t(row)],std::move(snapshot));
#else
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
#if GARNET_METAL_TOKEN_PREFIX_HOST_MIB > 512
    const auto memory=MetalContext::MemoryStats();
    if(memory.taskStatus!=0||!TokenPrefixHostMemoryFits(memory.allocated,memory.physical,memory.recommended,bytes,1))return false;
#endif
    auto snapshot=std::make_shared<HybridRowSnapshot>(tokenIdentity_,tokens,bytes);
#ifdef GARNET_METAL_PREFIX_PREFILL_ONLY
    snapshot->prefillTokens=tokenPrecisionHistory_.at(size_t(row)).PrefillTokens();
#endif
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
#endif
}
bool MetalDecoderSession::TouchTokenPrefix(int64_t row,int64_t epoch,X::Value prompt) {
    std::lock_guard<std::mutex> lock(mutex_);
    if(!gpu_||poisoned_||!tokenCache_||!requestRows_||row<0||epoch<=0||size_t(row)>=rowTokens_.size()||
       !requestRows_->Matches(size_t(row),uint64_t(epoch))||!rowTokensValid_[size_t(row)])
        throw std::invalid_argument("Prefix recency requires a current tracked request lease");
    const auto ids=Tokens(prompt,Host());const auto& history=rowTokens_[size_t(row)];
    if(ids.empty()||ids.size()>history.size()||!std::equal(ids.begin(),ids.end(),history.begin()))
        throw std::invalid_argument("Prefix recency key differs from the tracked request prompt");
    return tokenCache_->Touch(tokenIdentity_,ids);
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
#ifdef GARNET_METAL_PREFIX_PREFILL_ONLY
    if(snapshot->prefillTokens!=hit.tokens)throw std::logic_error("Restored snapshot producer domain differs from pure prefill boundary");
#endif
    const size_t width=size_t(head_->inputShapes_[0].back());
    if(snapshot->hidden.size()!=width)throw std::logic_error("Hybrid snapshot hidden width mismatch");
#ifdef GARNET_METAL_PREFIX_SHARED_BLOCKS
    if(snapshot->dense.size()!=states_.size())throw std::logic_error("Shared restored layer count differs");
    size_t scratchCount=0;
#endif
    for(size_t layer=0;layer<states_.size();++layer)for(size_t part=0;part<2;++part) {
        const auto& state=states_[layer];const auto& shape=state.shape[part];
        const size_t count=state.tokenCompact[part]?size_t(shape[1])*hit.tokens*size_t(shape[3]):state.columns[part];
#ifdef GARNET_METAL_PREFIX_SHARED_BLOCKS
        if(state.tokenCompact[part]) {
            const auto& blocks=snapshot->dense[layer][part];
            if(count>128*1024*1024/sizeof(float)||blocks.size()!=(hit.tokens+kPrefixBlockTokens-1)/kPrefixBlockTokens)throw std::logic_error("Shared prefix scratch/page bound differs");
            for(size_t i=0;i<blocks.size();++i) {
                size_t expected=std::min(kPrefixBlockTokens,hit.tokens-i*kPrefixBlockTokens);const auto& b=blocks[i];
                if(!b||b->tokens!=expected||b->groups!=size_t(shape[1])||b->width!=size_t(shape[3])||b->words.size()!=PrefixWordCount(b->groups,b->tokens,b->width))throw std::logic_error("Shared prefix block geometry differs");
            }
            scratchCount=std::max(scratchCount,count);
        }else
#endif
        if(snapshot->states[layer][part].size()!=count)throw std::logic_error("Hybrid snapshot state layout mismatch");
        gpu_->ValidateCompletedRow(state.matrix[part],size_t(row));
    }
#ifdef GARNET_METAL_PREFIX_SHARED_BLOCKS
    std::vector<float> assembled(scratchCount);
#endif
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
#ifdef GARNET_METAL_PREFIX_SHARED_BLOCKS
            if(state.tokenCompact[part]) {
                const auto& blocks=snapshot->dense[layer][part];assembled.resize(PrefixWordCount(size_t(shape[1]),hit.tokens,size_t(shape[3])));
                for(size_t i=0;i<blocks.size();++i)WritePrefixBlock(*blocks[i],assembled,size_t(shape[1]),hit.tokens,size_t(shape[3]),i*kPrefixBlockTokens,blocks[i]->tokens);
                gpu_->WriteRowPrefixGroups(state.matrix[part],size_t(row),size_t(shape[1]),size_t(shape[2]),size_t(shape[3]),hit.tokens,assembled);
            }else gpu_->WriteRow(state.matrix[part],size_t(row),data);
#else
            if(state.tokenCompact[part])gpu_->WriteRowPrefixGroups(state.matrix[part],size_t(row),size_t(shape[1]),size_t(shape[2]),size_t(shape[3]),hit.tokens,data);
            else gpu_->WriteRow(state.matrix[part],size_t(row),data);
#endif
        }
        gpu_->WriteRow(rowFinalHidden_,size_t(row),snapshot->hidden);
        rowTokens_[size_t(row)]=std::move(history);rowTokensValid_[size_t(row)]=1;finalHiddenReady_[size_t(row)]=1;
#ifdef GARNET_METAL_LOOKAHEAD_RUNTIME
        ResetLookaheadRow(size_t(row));
#endif
#ifdef GARNET_METAL_PREFIX_PREFILL_ONLY
        tokenPrecisionHistory_.at(size_t(row)).RestorePure(hit.tokens);
#endif
        poisoned_=false;return int64_t(hit.tokens);
    } catch(...) {throw;}
}
int64_t MetalDecoderSession::ReconcileDenseTokens(int64_t row,int64_t epoch,X::Value tokens) {
    std::lock_guard<std::mutex> lock(mutex_);
#ifdef GARNET_METAL_MTP_BRIDGE
    if(mtp_)throw std::invalid_argument("Dense reconciliation does not support MTP auxiliary state");
#endif
    if(!gpu_||poisoned_||!tokenCache_||!requestRows_||row<0||epoch<=0||
       size_t(row)>=rowTokens_.size()||!requestRows_->Matches(size_t(row),uint64_t(epoch))||
       !rowTokensValid_[size_t(row)])
        throw std::invalid_argument("Dense reconciliation requires a current tracked request row");
    auto ids=Tokens(tokens,Host());
    if(ids.empty()||ids.size()>tokenCapacity_)throw std::invalid_argument("Dense reconciliation query exceeds the nonempty context profile");
    std::vector<std::unique_lock<std::mutex>> locks;ValidateTokenOwners(locks);
    const size_t vocabulary=tokenVocabulary_;
    const auto& old=rowTokens_[size_t(row)];const size_t previous=old.size();
    if(!vocabulary||previous>tokenCapacity_||
       std::any_of(ids.begin(),ids.end(),[&](int64_t value){return uint64_t(value)>=vocabulary;})||
       std::any_of(old.begin(),old.end(),[&](int64_t value){return uint64_t(value)>=vocabulary;}))
        throw std::invalid_argument("Dense reconciliation token/history bounds mismatch");
    size_t common=0;
    while(common<std::min(previous,ids.size())&&old[common]==ids[common])++common;
    size_t kept=common;
    const bool hiddenReady=size_t(row)<finalHiddenReady_.size()&&finalHiddenReady_[size_t(row)];
    // If the new input ends inside discarded history, replay its last token:
    // KV is retained at every token, but historical final hidden is not.
    if(kept==ids.size()&&kept&&(!hiddenReady||kept<previous))--kept;
    std::vector<MetalRowClearRange> ranges;
    for(const auto& state:states_)for(size_t part=0;part<2;++part) {
        const auto& shape=state.shape[part];
        if(!state.tokenCompact[part]||shape.size()!=4||shape[0]!=int64_t(states_[0].rows)||
           shape[1]<=0||shape[2]<=0||shape[3]<=0||previous>size_t(shape[2]))
            throw std::invalid_argument("Dense reconciliation excludes recurrent or non-KV state");
        gpu_->ValidateCompletedRow(state.matrix[part],size_t(row));
        if(kept<previous)ranges.push_back({state.matrix[part],size_t(row),size_t(shape[1]),size_t(shape[2]),size_t(shape[3]),kept,previous});
    }
    if(kept<previous) {
        if(!rowFinalHidden_.Valid()||size_t(row)>=finalHiddenReady_.size()||!rowFinalWidth_)
            throw std::logic_error("Dense reconciliation hidden bank is unavailable");
        gpu_->ValidateCompletedRow(rowFinalHidden_,size_t(row));
        ranges.push_back({rowFinalHidden_,size_t(row),1,1,rowFinalWidth_,0,1});
    }
    if(reconcileCalls_==SIZE_MAX)throw std::overflow_error("Dense reconciliation call counter exhausted");
    size_t bytes=0;
    if(!ranges.empty()) {
        poisoned_=true;
        // The context validates every physical range before encoding writes.
        // It issues one synchronous GPU blit, never snapshots the prefix.
        bytes=gpu_->ZeroRowRanges(ranges);
        rowTokens_[size_t(row)].resize(kept);finalHiddenReady_[size_t(row)]=0;
#ifdef GARNET_METAL_PREFIX_PREFILL_ONLY
        tokenPrecisionHistory_.at(size_t(row)).Truncate(kept);
#endif
        reconcileHiddenPending_=true;poisoned_=false;
    }
    ++reconcileCalls_;reconcileOld_=previous;reconcileLCP_=common;reconcileKept_=kept;
    reconcileZeroBytes_=bytes;reconcileReplay_=kept<common;reconcileGpuMs_=bytes?gpu_->LastGPUMs():0;
#ifdef GARNET_METAL_TARGET_JACOBI
    if(targetJacobiEnabled_)targetJacobi_.ResetRow(size_t(row));
#endif
#ifdef GARNET_METAL_LOOKAHEAD_RUNTIME
    ResetLookaheadRow(size_t(row));
#endif
    return int64_t(kept);
}
std::string MetalDecoderSession::DenseReconcileStatsJson() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream out;
    out<<"{\"calls\":"<<reconcileCalls_<<",\"previous_tokens\":"<<reconcileOld_
       <<",\"common_tokens\":"<<reconcileLCP_<<",\"retained_tokens\":"<<reconcileKept_
       <<",\"zeroed_bytes\":"<<reconcileZeroBytes_<<",\"replay_last_token\":"<<(reconcileReplay_?"true":"false")
       <<",\"gpu_ms\":"<<reconcileGpuMs_<<",\"host_snapshot_bytes\":0}";
    return out.str();
}
std::string MetalDecoderSession::TokenPrefixStatsJson() {
    std::lock_guard<std::mutex> lock(mutex_);
    if(!tokenCache_)throw std::logic_error("Token prefix cache is not configured");
    const auto stats=tokenCache_->Stats();std::ostringstream out;
    out<<"{\"retention_enabled\":"<<(tokenCacheBudget_?"true":"false")<<",\"entries\":"<<stats.entries<<",\"accounted_host_bytes\":"<<stats.accountedBytes
       <<",\"hits\":"<<stats.hits<<",\"misses\":"<<stats.misses<<",\"evictions\":"<<stats.evictions
       <<",\"replacements\":"<<stats.replacements;
#ifdef GARNET_METAL_PREFIX_DIRECT_CAPTURE
    out<<",\"dense_capture_compared_bytes\":"<<tokenPrefixComparedBytes_
       <<",\"dense_capture_copied_bytes\":"<<tokenPrefixCopiedBytes_
       <<",\"dense_capture_reused_blocks\":"<<tokenPrefixReusedBlocks_
       <<",\"dense_capture_new_blocks\":"<<tokenPrefixNewBlocks_;
#endif
#ifdef GARNET_METAL_PREFIX_SHARED_BLOCKS
    out<<",\"private_host_bytes\":"<<stats.privateBytes<<",\"shared_resource_bytes\":"<<stats.resourceBytes
       <<",\"shared_resource_count\":"<<stats.resourceCount;
#endif
#ifdef GARNET_METAL_PREFIX_PREFILL_ONLY
    out<<",\"capture_policy\":\"prefill_only_v1\",\"prefill_capture_skipped_mixed\":"<<tokenPrefillSkippedMixed_;
#endif
    out<<"}";return out.str();
}
bool MetalDecoderSession::ClearTokenPrefixes() {
    std::lock_guard<std::mutex> lock(mutex_);
    if(!tokenCache_)throw std::logic_error("Token prefix cache is not configured");
    tokenCache_->Clear();return true;
}
}
#endif
