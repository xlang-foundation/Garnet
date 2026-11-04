// SPDX-License-Identifier: Apache-2.0
#include "decoder_session.h"
#include "lowering_context.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <atomic>
#include <limits>
namespace Garnet {
namespace {
MetalCompiledGraph* Graph(const X::Value& value,X3PackageHost* host) {
    if(value.runtime()!=host->runtime)throw std::invalid_argument("Metal session graph runtime mismatch");
    auto graph=value.NativeData<MetalCompiledGraph>();
    if(!graph)throw std::invalid_argument("Metal session requires compiled_graph instances");
    return graph;
}
X::Value TensorValue(X3PackageHost* host,MetalContext& gpu,const MetalMatrix& matrix,const std::vector<int64_t>& shape) {
    auto data=gpu.Download(matrix);
    return X::Tensor::Create(host,X3_TENSOR_FLOAT32,shape,data.data(),data.size()*sizeof(float));
}
}
void MetalDecoderSession::Clear() {
#ifdef GARNET_METAL_MTP_BRIDGE
    mtp_.reset();
#endif
#ifdef GARNET_METAL_VERIFY_ATTENTION
    verification_.reset();verificationStats_="{}";
#endif
#ifdef GARNET_METAL_TOKEN_PREFIX_CACHE
    tokenCache_.reset();tokenIdentity_={};tokenCacheBudget_=tokenCapacity_=0;rowTokens_.clear();rowTokensValid_.clear();
    tokenGraphs_.clear();tokenGraphOwners_.clear();
#endif
#ifdef GARNET_METAL_ROW_FINAL_HIDDEN
    rowFinalHidden_=MetalMatrix();rowFinalWidth_=0;finalHiddenReady_.clear();
#endif
#ifdef GARNET_METAL_CONTINUOUS_ROWS
    requestRows_.reset();managedRows_=false;
#endif
#ifdef GARNET_METAL_SHARED_PREFILL_SCRATCH
    prefillScratch_.reset();
#endif
#ifdef GARNET_METAL_REQUEST_ROWS
    predictionInitialized_=false;rowResetBytes_=0;
#endif
#ifdef GARNET_METAL_DENSE_PREFIX_CACHE
    prefixCache_.clear();prefixBytes_=prefixRestoreBytes_=prefixCaptureBytes_=0;prefixClock_=0;prefixFreshReset_=false;
#endif
    screenedHeadDispatches_=0;screenedHeadSeedRows_=0;interleavedScreenDispatches_=0;
    reusedSourceBytes_=reusedCopyBytes_=reusedCopyStates_=0;
#ifdef GARNET_METAL_REUSED_ZERO_SOURCE
    std::vector<float>().swap(resetZeroSource_);
#endif
    prefillPartialSubmissions_=0;fusedHeadArgmaxDispatches_=0;bodyShapeDispatches_=bodyShapeMisses_=0;deferredZeroBytes_=deferredZeroStates_=0;
    dirtyPrefixResetBytes_=dirtyPrefixResetStates_=dirtyPrefixResetFallbackStates_=0;
    joinedPrefillPairs_=0;
    constantHeadDispatches_=0;singleCoderHeadDispatches_=0;residualNormDispatches_=0;packedPrefillNormDispatches_=0;globalHalfAlphaDispatches_=0;wideBatchPrefillDispatches_=0;interleavedBodyDispatches_=0;coalescedBlockDispatches_=0;mlpGraphDispatches_=0;joinedGlobalMlpBlocks_=0;
    hidden_=MetalMatrix();prediction_=MetalMatrix();hiddenShape_.clear();states_.clear();owners_.clear();head_=nullptr;
    gpu_.reset();kvInplaceUpdates_=0;poisoned_=false;inferenceMs_=0;prepareMs_=encodeMs_=gpuMs_=waitMs_=0;
}
void MetalDecoderSession::Close() {std::lock_guard<std::mutex> lock(mutex_);Clear();}
void MetalDecoderSession::ResetUnlocked() {
#ifdef GARNET_METAL_TOKEN_PREFIX_CACHE
    rowTokens_.assign(states_.empty()?0:states_[0].rows,{});rowTokensValid_.assign(rowTokens_.size(),1);
#endif
#ifdef GARNET_METAL_ROW_FINAL_HIDDEN
    rowFinalHidden_=MetalMatrix();rowFinalWidth_=0;finalHiddenReady_.clear();
#endif
#ifdef GARNET_METAL_CONTINUOUS_ROWS
    if(requestRows_)requestRows_->Clear();managedRows_=false;
#endif
#ifdef GARNET_METAL_REQUEST_ROWS
    predictionInitialized_=false;rowResetBytes_=0;
#endif
#ifdef GARNET_METAL_DENSE_PREFIX_CACHE
    prefixFreshReset_=false;prefixRestoreBytes_=prefixCaptureBytes_=0;
#endif
    screenedHeadDispatches_=0;screenedHeadSeedRows_=0;interleavedScreenDispatches_=0;
    reusedCopyBytes_=reusedCopyStates_=0;
    prefillPartialSubmissions_=0;fusedHeadArgmaxDispatches_=0;bodyShapeDispatches_=bodyShapeMisses_=0;deferredZeroBytes_=deferredZeroStates_=0;
    joinedPrefillPairs_=0;
    constantHeadDispatches_=0;singleCoderHeadDispatches_=0;residualNormDispatches_=0;packedPrefillNormDispatches_=0;globalHalfAlphaDispatches_=0;wideBatchPrefillDispatches_=0;interleavedBodyDispatches_=0;coalescedBlockDispatches_=0;mlpGraphDispatches_=0;joinedGlobalMlpBlocks_=0;
    if(!gpu_)throw std::logic_error("Metal session is closed");
#if defined(GARNET_METAL_CONTINUOUS_ROWS) && !defined(GARNET_METAL_DIRTY_PREFIX_RESET)
    poisoned_=true;
#endif
    dirtyPrefixResetBytes_=dirtyPrefixResetStates_=dirtyPrefixResetFallbackStates_=0;
#ifdef GARNET_METAL_DIRTY_PREFIX_RESET
    const bool forceFull=poisoned_;poisoned_=true;
#endif
#ifdef GARNET_METAL_REUSED_ZERO_SOURCE
    poisoned_=true;
#endif
    for(auto& state:states_)for(size_t i=0;i<2;++i) {
#ifdef GARNET_METAL_REUSED_ZERO_SOURCE
        const size_t count=state.rows*state.columns[i];
        if(count>resetZeroSource_.size())throw std::logic_error("Retained reset source does not cover private state");
        gpu_->Update(state.matrix[i],resetZeroSource_.data(),count);
        reusedCopyBytes_+=count*sizeof(float);++reusedCopyStates_;
#else
#ifdef GARNET_METAL_DEFERRED_STATE_ZERO
        gpu_->DeferZero(state.matrix[i]);
#else
#ifdef GARNET_METAL_DIRTY_PREFIX_RESET
        if(!forceFull&&state.prefixLayout[i]&&state.zeroOutsideDirty[i]) {
            const auto& shape=state.shape[i];
            gpu_->ZeroPrefixGroups(state.matrix[i],size_t(shape[1]),size_t(shape[2]),size_t(shape[3]),state.dirtyExtent[i]);
            dirtyPrefixResetBytes_+=state.rows*size_t(shape[1])*state.dirtyExtent[i]*size_t(shape[3])*sizeof(float);
            ++dirtyPrefixResetStates_;
        } else {
            gpu_->Update(state.matrix[i],std::vector<float>(state.rows*state.columns[i],0));
            dirtyPrefixResetBytes_+=state.rows*state.columns[i]*sizeof(float);++dirtyPrefixResetFallbackStates_;
        }
        state.dirtyExtent[i]=0;state.zeroOutsideDirty[i]=state.prefixLayout[i];
#else
#ifdef GARNET_METAL_DIRECT_STATE_ZERO
        gpu_->Zero(state.matrix[i]);
#else
        gpu_->Update(state.matrix[i],std::vector<float>(state.rows*state.columns[i],0));
#endif
#endif
#endif
#endif
    }
#ifdef GARNET_METAL_MTP_BRIDGE
    ResetMTP();
#endif
    hidden_=MetalMatrix();hiddenShape_.clear();kvInplaceUpdates_=0;poisoned_=false;inferenceMs_=0;prepareMs_=encodeMs_=gpuMs_=waitMs_=0;
#ifdef GARNET_METAL_DENSE_PREFIX_CACHE
    prefixFreshReset_=true;
#endif
}
void MetalDecoderSession::Reset() {std::lock_guard<std::mutex> lock(mutex_);ResetUnlocked();}
#ifdef GARNET_METAL_REQUEST_ROWS
void MetalDecoderSession::ResetRow(int64_t row) {
    std::lock_guard<std::mutex> lock(mutex_);
    ResetRowUnlocked(row);
}
void MetalDecoderSession::ResetRowUnlocked(int64_t row) {
    if(!gpu_||poisoned_||states_.empty())throw std::logic_error("Request-row reset requires a ready session; use full reset after failure");
    if(row<0||uint64_t(row)>=states_[0].rows)throw std::invalid_argument("Request-row reset index out of bounds");
    std::vector<MetalMatrix> matrices;
    for(const auto& state:states_)for(size_t part=0;part<2;++part) {
        if(state.rows!=states_[0].rows)throw std::logic_error("Inconsistent native request row count");
        matrices.push_back(state.matrix[part]);
    }
#ifdef GARNET_METAL_ROW_FINAL_HIDDEN
    if(rowFinalHidden_.Valid())matrices.push_back(rowFinalHidden_);
#endif
    if(hidden_.Valid())matrices.push_back(hidden_);
    if(predictionInitialized_)matrices.push_back(prediction_);
    size_t bytes=0;
    // Validate every owner and range before touching any request's state.
    for(const auto& matrix:matrices)gpu_->ValidateCompletedRow(matrix,size_t(row));
    auto charge=[&](size_t columns){
        if(columns>SIZE_MAX/sizeof(float)||bytes>SIZE_MAX-columns*sizeof(float))
            throw std::overflow_error("Request-row state byte count overflow");
        bytes+=columns*sizeof(float);
    };
    for(const auto& state:states_)for(auto columns:state.columns)charge(columns);
    if(hidden_.Valid()) {
        size_t columns=1;
        for(size_t i=1;i<hiddenShape_.size();++i) {
            if(hiddenShape_[i]<=0||uint64_t(hiddenShape_[i])>SIZE_MAX/columns)
                throw std::overflow_error("Request-row hidden width overflow");
            columns*=size_t(hiddenShape_[i]);
        }
        charge(columns);
    }
#ifdef GARNET_METAL_ROW_FINAL_HIDDEN
    if(rowFinalHidden_.Valid())charge(rowFinalWidth_);
#endif
    if(predictionInitialized_)charge(2);
    poisoned_=true;
    for(const auto& matrix:matrices)gpu_->ZeroRow(matrix,size_t(row));
#ifdef GARNET_METAL_ROW_FINAL_HIDDEN
    if(!finalHiddenReady_.empty())finalHiddenReady_[size_t(row)]=0;
#endif
#ifdef GARNET_METAL_TOKEN_PREFIX_CACHE
    rowTokens_[size_t(row)].clear();rowTokensValid_[size_t(row)]=1;
#endif
#ifdef GARNET_METAL_MTP_BRIDGE
    ResetMTPRow(size_t(row));
#endif
    rowResetBytes_=bytes;poisoned_=false;
}
#endif
#ifdef GARNET_METAL_CONTINUOUS_ROWS
X::Value MetalDecoderSession::ClaimRequestRow(int64_t request) {
    std::lock_guard<std::mutex> lock(mutex_);
    if(!gpu_||poisoned_||!requestRows_||request<=0)
        throw std::invalid_argument("Native row claim requires a ready session and positive request identity");
    const auto lease=requestRows_->Claim(uint64_t(request));
    try {
        auto result=X::Value::List(Host());
        result.Append(X::Value(int64_t(lease.row)));result.Append(X::Value(int64_t(lease.epoch)));
        ResetRowUnlocked(int64_t(lease.row));managedRows_=true;return result;
    }
    catch(...) {requestRows_->Release(lease.row,lease.epoch);poisoned_=true;throw;}
}
bool MetalDecoderSession::ReleaseRequestRow(int64_t row,int64_t epoch) {
    std::lock_guard<std::mutex> lock(mutex_);
    if(!gpu_||poisoned_||!requestRows_)throw std::logic_error("Native row release requires a ready session");
    if(row<0||epoch<=0||!requestRows_->Matches(size_t(row),uint64_t(epoch)))return false;
    ResetRowUnlocked(row);
    return requestRows_->Release(size_t(row),uint64_t(epoch));
}
bool MetalDecoderSession::RequestRowMatches(size_t row,uint64_t epoch) {
    std::lock_guard<std::mutex> lock(mutex_);
    return gpu_&&!poisoned_&&requestRows_&&requestRows_->Matches(row,epoch);
}
std::pair<size_t,size_t> MetalDecoderSession::RequestRowLayout() {
    std::lock_guard<std::mutex> lock(mutex_);
    if(!gpu_||poisoned_||!requestRows_||states_.empty())throw std::logic_error("Native scheduler requires a ready Metal session");
    const auto rows=states_[0].rows;size_t capacity=SIZE_MAX;
    for(size_t row=0;row<rows;++row)if(requestRows_->Owned(row))
        throw std::logic_error("Native scheduler requires exclusive unclaimed request rows");
    for(size_t i=0;i<states_.size();++i) {
        auto graph=Graph(owners_[i],Host());std::lock_guard<std::mutex> graphLock(graph->mutex_);
        if(!graph->ready_)throw std::logic_error("Native scheduler graph owner is closed");
        for(size_t part=0;part<2;++part)if(graph->CanResetKVPrefix(part+1)) {
            const auto& shape=states_[i].shape[part];
            if(shape.size()!=4||shape[0]!=int64_t(rows)||shape[2]<=0)throw std::logic_error("Native scheduler KV row layout is invalid");
            capacity=std::min(capacity,size_t(shape[2]));
        }
    }
    if(capacity==SIZE_MAX)throw std::logic_error("Native scheduler requires a proved bounded KV layout");
    return {rows,capacity};
}
#endif
bool MetalDecoderSession::Build(X::Value graphs,X::Value head) {
    std::lock_guard<std::mutex> lock(mutex_);Clear();
    if(graphs.runtime()!=Host()->runtime||!graphs.IsList()||graphs.Size()==0)
        throw std::invalid_argument("Metal session requires a nonempty graph list");
    try {
        gpu_=std::make_unique<MetalContext>();
#ifdef GARNET_METAL_SHARED_PREFILL_SCRATCH
        prefillScratch_=std::make_shared<MetalPrefillScratch>();
#endif
        size_t width=0,batch=0;
        for(uint64_t i=0;i<graphs.Size();++i) {
            auto owner=graphs.Get(i);auto graph=Graph(owner,Host());
            std::lock_guard<std::mutex> graphLock(graph->mutex_);
            if(!graph->ready_||graph->inputs_.size()<3||graph->outputParts_.size()!=3)
                throw std::invalid_argument("Metal session requires packed hidden and two-state layers");
            auto& input=graph->slots_[graph->inputs_[0]];
            if(input.shape.size()!=2&&input.shape.size()!=3)throw std::invalid_argument("Metal session hidden rank invalid");
            const auto layerWidth=size_t(input.shape.back());
            if(i&&(input.rows!=batch||layerWidth!=width))throw std::invalid_argument("Metal session layer dimensions mismatch");
            batch=input.rows;width=layerWidth;State state;state.rows=batch;
            for(size_t k=0;k<2;++k) {
                auto& source=graph->slots_[graph->inputs_[k+1]];
                auto& output=graph->slots_[graph->outputParts_[k+1]];
                if(source.rows!=batch||output.rows!=source.rows||output.columns!=source.columns)
                    throw std::invalid_argument("Metal session state layout mismatch");
                state.shape[k]=source.shape;state.columns[k]=source.columns;
                state.matrix[k]=gpu_->Allocate(source.rows,source.columns);
#ifdef GARNET_METAL_DIRTY_PREFIX_RESET
                state.prefixLayout[k]=graph->CanResetKVPrefix(k+1);
#endif
            }
#ifdef GARNET_METAL_DENSE_PREFIX_CACHE
            state.prefixEligible=batch==1&&graph->CanResetKVPrefix(1)&&graph->CanResetKVPrefix(2);
            state.prefixOwnerGeneration=graph->generation_;
#endif
#ifdef GARNET_METAL_TOKEN_PREFIX_CACHE
            state.tokenOwnerGeneration=graph->generation_;
            for(size_t part=0;part<2;++part)state.tokenCompact[part]=graph->CanResetKVPrefix(part+1);
            tokenGraphs_[graph]={size_t(i),graph->generation_};
#endif
            states_.push_back(std::move(state));owners_.push_back(std::move(owner));
        }
        if(head.IsValid()&&!head.IsNone()) {
            head_=Graph(head,Host());std::lock_guard<std::mutex> graphLock(head_->mutex_);
            if(!head_->ready_||head_->inputs_.size()!=1||head_->inputShapes_[0]!=std::vector<int64_t>{int64_t(batch),int64_t(width)})
                throw std::invalid_argument("Metal session vocabulary head dimensions mismatch");
            headGeneration_=head_->generation_;owners_.push_back(head);prediction_=gpu_->Allocate(batch,2);
        }
#ifdef GARNET_METAL_REUSED_ZERO_SOURCE
        size_t maximum=0;
        for(const auto& state:states_)for(size_t k=0;k<2;++k)maximum=std::max(maximum,state.rows*state.columns[k]);
        resetZeroSource_.assign(maximum,0.0f);reusedSourceBytes_=maximum*sizeof(float);
#endif
#ifdef GARNET_METAL_CONTINUOUS_ROWS
        requestRows_=std::make_unique<RequestRowLeases>(states_[0].rows);
#endif
        ResetUnlocked();
#ifdef GARNET_METAL_QWEN38_WIDE_PREFILL
        // Private resident state is initialized. Retire only the defaults
        // proved unused by resident execution before compiling wider graphs.
        for(size_t i=0;i<states_.size();++i) {
            auto graph=Graph(owners_[i],Host());
            std::lock_guard<std::mutex> graphLock(graph->mutex_);
            graph->RetireResidentDefaults();
        }
#endif
        return true;
    }catch(...){Clear();throw;}
}
#ifdef GARNET_METAL_QWEN38_WIDE_PREFILL
bool MetalDecoderSession::PreparePrefill(X::Value value,int64_t layer) {
    std::lock_guard<std::mutex> lock(mutex_);
    if(!gpu_||poisoned_||layer<0||uint64_t(layer)>=states_.size())
        throw std::invalid_argument("Prefill preparation requires a ready resident layer");
    auto graph=Graph(value,Host()),original=Graph(owners_[size_t(layer)],Host());
    std::vector<MetalCompiledGraph*> ordered={graph,original};
    std::sort(ordered.begin(),ordered.end(),std::less<MetalCompiledGraph*>());
    ordered.erase(std::unique(ordered.begin(),ordered.end()),ordered.end());
    std::vector<std::unique_lock<std::mutex>> locks;
    for(auto owner:ordered)locks.emplace_back(owner->mutex_);
    const auto& state=states_[size_t(layer)];
    if(!original->ready_||original->inputShapes_.empty()||!graph->ready_||graph->outputParts_.size()!=3||
       graph->inputShapes_.size()<3||graph->inputShapes_[0].size()!=3||
       graph->inputShapes_[0][0]!=int64_t(state.rows)||
       graph->inputShapes_[0].back()!=original->inputShapes_[0].back()||
       graph->inputShapes_[0][1]<1||graph->inputShapes_[0][1]>int64_t(kMetalPrefillChunkLimit)||
       graph->inputShapes_[1]!=state.shape[0]||graph->inputShapes_[2]!=state.shape[1])
        throw std::invalid_argument("Prefill preparation does not match resident layer state");
#ifdef GARNET_METAL_TOKEN_PREFIX_CACHE
    // Prefix histories may only execute graph families sharing the base weights.
    if(original->generation_!=state.tokenOwnerGeneration||graph->weightRecords_.size()!=original->weightRecords_.size())
        throw std::invalid_argument("Token-prefix prefill source changed");
    auto buffers=[](const MetalCompiledGraph::Slot& slot) {
        std::vector<uintptr_t> ids;
        auto add=[&](const MetalBufferInfo& info){if(info.identity)ids.push_back(info.identity);};
        add(MetalContext::MatrixBufferInfo(slot.matrix));
        for(const auto& info:MetalContext::WeightBufferInfo(slot.bf16))add(info);
#ifdef GARNET_METAL_MLX_AFFINE_INT4
        for(const auto& info:MetalContext::AffineInt4BufferInfo(slot.affineInt4))add(info);
#endif
        return ids;
    };
    for(const auto& item:graph->weightRecords_) {
        auto found=original->weightRecords_.find(item.first);
        if(found==original->weightRecords_.end()||buffers(item.second.slot)!=buffers(found->second.slot))
            throw std::invalid_argument("Token-prefix prefill must share original native weights");
    }
    if(!tokenGraphs_.count(graph))tokenGraphOwners_.push_back(value);
    tokenGraphs_[graph]={size_t(layer),graph->generation_};
#endif
    graph->RetireResidentDefaults();
#ifdef GARNET_METAL_SHARED_PREFILL_SCRATCH
    graph->SharePrefillScratch(prefillScratch_);
#endif
    return true;
}
#endif
X::Value MetalDecoderSession::Run(X::Value graphs,X::Value hidden,X::Value positions,
    X::Value slots,X::Value active,int64_t predictionToken) {
    std::lock_guard<std::mutex> lock(mutex_);
    return RunUnlocked(graphs,hidden,positions,slots,active,predictionToken);
}
#ifdef GARNET_METAL_CONTINUOUS_ROWS
X::Value MetalDecoderSession::RunLeased(X::Value graphs,X::Value hidden,X::Value positions,
    X::Value slots,X::Value active,int64_t predictionToken,X::Value epochs) {
    std::lock_guard<std::mutex> lock(mutex_);
    if(!gpu_||poisoned_||!requestRows_||states_.empty()||epochs.runtime()!=Host()->runtime||
       !epochs.IsList()||epochs.Size()!=states_[0].rows)
        throw std::invalid_argument("Leased Metal execution requires one epoch per physical request row");
    std::vector<uint64_t> expected;
    for(uint64_t row=0;row<epochs.Size();++row) {
        auto value=epochs.Get(row);
        if(!value.IsInt64()||value.ToInt64()<0)throw std::invalid_argument("Native request epochs must be nonnegative integers");
        auto epoch=uint64_t(value.ToInt64());
        if(epoch&&!requestRows_->Matches(size_t(row),epoch))throw std::invalid_argument("Stale native request row execution");
        expected.push_back(epoch);
    }
    // Keep the session lock through validation, encoding and GPU completion.
    return RunUnlocked(graphs,hidden,positions,slots,active,predictionToken,&expected);
}
#endif
X::Value MetalDecoderSession::RunUnlocked(X::Value graphs,X::Value hidden,X::Value positions,
    X::Value slots,X::Value active,int64_t predictionToken,const std::vector<uint64_t>* epochs,const std::vector<int64_t>* tokenIds,
    VerificationRun* verification) {
#ifndef GARNET_METAL_VERIFY_ATTENTION
    (void)verification;
#endif
#ifndef GARNET_METAL_TOKEN_PREFIX_CACHE
    (void)tokenIds;
#endif
#ifndef GARNET_METAL_CONTINUOUS_ROWS
    (void)epochs;
#endif
    if(!gpu_||poisoned_)throw std::logic_error("Reset a closed/failed Metal session before running");
    if(graphs.runtime()!=Host()->runtime||!graphs.IsList()||graphs.Size()!=states_.size())
        throw std::invalid_argument("Metal session layer count mismatch");
    std::vector<MetalCompiledGraph*> layers,locked;
    for(uint64_t i=0;i<graphs.Size();++i)layers.push_back(Graph(graphs.Get(i),Host()));
    locked=layers;if(head_)locked.push_back(head_);
#ifdef GARNET_METAL_VERIFY_ATTENTION
    if(verification)locked.push_back(verification->plan->head);
#endif
    std::sort(locked.begin(),locked.end(),std::less<MetalCompiledGraph*>());
    if(std::adjacent_find(locked.begin(),locked.end())!=locked.end())throw std::invalid_argument("Metal session graphs must be distinct");
#ifdef GARNET_METAL_VERIFY_ATTENTION
    // Verification may only advance state belonging to the exact source layer.
    // Lock those owners too so close/rebuild cannot race the family check.
    for(size_t i=0;i<states_.size();++i)locked.push_back(Graph(owners_[i],Host()));
    std::sort(locked.begin(),locked.end(),std::less<MetalCompiledGraph*>());
    locked.erase(std::unique(locked.begin(),locked.end()),locked.end());
#endif
    std::vector<std::unique_lock<std::mutex>> graphLocks;
    for(auto graph:locked)graphLocks.emplace_back(graph->mutex_);
#ifdef GARNET_METAL_TOKEN_PREFIX_CACHE
    if(tokenIds)for(size_t layer=0;layer<layers.size();++layer) {
        auto found=tokenGraphs_.find(layers[layer]);
        if(found==tokenGraphs_.end()||found->second.first!=layer||found->second.second!=layers[layer]->generation_)
            throw std::invalid_argument("Tracked execution requires unchanged native graph families");
    }
#endif
#ifdef GARNET_METAL_SHARED_PREFILL_SCRATCH
    // Lock all owners after graph locks in one stable order. Shared temporaries
    // are safe across sessions and standalone graph calls only under this lease.
    std::vector<MetalPrefillScratch*> scratchOwners;
    for(auto graph:locked)if(graph->prefillScratch_)scratchOwners.push_back(graph->prefillScratch_.get());
    std::sort(scratchOwners.begin(),scratchOwners.end(),std::less<MetalPrefillScratch*>());
    scratchOwners.erase(std::unique(scratchOwners.begin(),scratchOwners.end()),scratchOwners.end());
    std::vector<std::unique_lock<std::mutex>> scratchLocks;
    for(auto owner:scratchOwners)scratchLocks.emplace_back(owner->mutex);
#endif
    if(head_&&(!head_->ready_||head_->generation_!=headGeneration_))throw std::logic_error("Metal session head was closed/rebuilt");
    const auto start=std::chrono::steady_clock::now();
    const auto& inputShape=layers[0]->inputShapes_;
    if(inputShape.empty()||(inputShape[0].size()!=2&&inputShape[0].size()!=3))throw std::invalid_argument("Metal session hidden profile missing");
    const size_t chunk=inputShape[0].size()==3?size_t(inputShape[0][1]):1;
#ifdef GARNET_METAL_MTP_BRIDGE
    if(mtp_)ValidateMTPTarget(chunk);
    std::vector<float> mtpRunFlags;
#endif
    if(predictionToken<-1||(predictionToken>=0&&(!head_||uint64_t(predictionToken)>=chunk)))
        throw std::invalid_argument("Metal session prediction token/head invalid");
#ifdef GARNET_METAL_VERIFY_ATTENTION
    if(verification)ValidateVerificationRun(*verification,layers);
#endif
    // Validate and upload every host input before encoding or changing state.
    for(size_t i=0;i<layers.size();++i) {
        auto graph=layers[i];const auto& state=states_[i];
#ifdef GARNET_METAL_VERIFY_ATTENTION
        if(graph->verificationTokens_!=layers[0]->verificationTokens_)
            throw std::invalid_argument("Verification cannot mix ordinary and verification layers");
        if(graph->verificationTokens_) {
            auto original=Graph(owners_[i],Host());
            if(!original->ready_||!graph->SharesVerificationWeights(*original))
                throw std::invalid_argument("Verification source layer was closed, rebuilt or replaced");
        }
#endif
        if(!graph->ready_||graph->outputParts_.size()!=3||graph->inputShapes_.size()<3||graph->inputShapes_[0]!=inputShape[0]||
            graph->inputShapes_[1]!=state.shape[0]||graph->inputShapes_[2]!=state.shape[1])
            throw std::invalid_argument("Metal session layer/state profile mismatch");
        auto inputs=X::Value::List(Host());inputs.Append(i?X::Value():hidden);inputs.Append(X::Value());inputs.Append(X::Value());
        if(graph->inputs_.size()==4)inputs.Append(active);
        else if(graph->inputs_.size()==6){inputs.Append(positions);inputs.Append(slots);inputs.Append(active);}
        else throw std::invalid_argument("Metal session unsupported controls");
        graph->PrepareInputs(inputs,i?std::unordered_set<size_t>{0,1,2}:std::unordered_set<size_t>{1,2});
    }
#ifdef GARNET_METAL_MTP_BRIDGE
    if(mtp_) {
        X::Tensor flags(active);auto lease=flags.Acquire();const auto* data=static_cast<const float*>(flags.Info().data);
        mtpRunFlags.assign(data,data+mtp_->batch*chunk);
    }
#endif
#ifdef GARNET_METAL_ROW_FINAL_HIDDEN
    std::vector<uint8_t> rowWillUpdate(states_[0].rows,0);
#endif
#if defined(GARNET_METAL_INDEXED_TOKEN_PROJECTIONS)||defined(GARNET_METAL_ACTIVE_DECODE_PROJECTIONS)
    std::vector<size_t> selectedTokenRows;
    bool selectProjectionRows=false;
#ifdef GARNET_METAL_INDEXED_TOKEN_PROJECTIONS
    selectProjectionRows=chunk>1;
#endif
#ifdef GARNET_METAL_ACTIVE_DECODE_PROJECTIONS
    selectProjectionRows=selectProjectionRows||chunk==1;
#endif
#endif
    {
        X::Tensor flags(active);auto info=flags.Info();auto lease=flags.Acquire();
        const auto* data=static_cast<const float*>(info.data);
        const size_t count=states_[0].rows*chunk;
        for(size_t i=0;i<count;++i)if(!std::isfinite(data[i])||(data[i]!=0&&data[i]!=1))
            throw std::invalid_argument("Invalid Metal session active flags");
#if defined(GARNET_METAL_INDEXED_TOKEN_PROJECTIONS)||defined(GARNET_METAL_ACTIVE_DECODE_PROJECTIONS)
        if(selectProjectionRows)for(size_t i=0;i<count;++i)if(data[i]!=0)selectedTokenRows.push_back(i);
#endif
#ifdef GARNET_METAL_ROW_FINAL_HIDDEN
        for(size_t i=0;i<count;++i)if(data[i]!=0)rowWillUpdate[i/chunk]=1;
#endif
#ifdef GARNET_METAL_TOKEN_PREFIX_CACHE
        if(tokenIds) {
            if(!tokenCache_||tokenIds->size()!=count)throw std::invalid_argument("Tracked input token profile mismatch");
            X::Tensor slotTensor(slots),positionTensor(positions);auto sl=slotTensor.Acquire();auto pl=positionTensor.Acquire();
            auto slotInfo=slotTensor.Info(),positionInfo=positionTensor.Info();
            if(slotInfo.dtype!=X3_TENSOR_FLOAT32||positionInfo.dtype!=X3_TENSOR_FLOAT32||
               slotInfo.device_type||positionInfo.device_type||slotInfo.symbolic||positionInfo.symbolic||
               !slotInfo.data||!positionInfo.data||!IsContiguousTensor(slotTensor)||!IsContiguousTensor(positionTensor)||
               slotInfo.byte_size!=count*sizeof(float)||positionInfo.byte_size!=count*3*sizeof(float))
                throw std::invalid_argument("Token-prefix controls require text slots and three position axes");
            const auto* slotData=static_cast<const float*>(slotInfo.data);
            const auto* positionData=static_cast<const float*>(positionInfo.data);
            for(size_t row=0;row<states_[0].rows;++row) {
                size_t next=rowTokens_[row].size();
                for(size_t token=0;token<chunk;++token)if(data[row*chunk+token]!=0) {
                    if(!rowTokensValid_[row]||slotData[row*chunk+token]!=float(next))
                        throw std::invalid_argument("Tracked tokens require a contiguous native history");
                    for(size_t axis=0;axis<3;++axis)if(positionData[(row*chunk+token)*3+axis]!=float(next))
                        throw std::invalid_argument("Token-prefix positions do not match text policy");
                    ++next;
                }
                // Allocate before any GPU/state mutation so commit cannot fail.
                if(next>tokenCapacity_)throw std::invalid_argument("Tracked history exceeds the native context profile");
                rowTokens_[row].reserve(next);
            }
        }
#endif
#ifdef GARNET_METAL_CONTINUOUS_ROWS
        if(managedRows_)for(size_t i=0;i<count;++i)if(data[i]!=0&&!requestRows_->Owned(i/chunk))
            throw std::invalid_argument("Active Metal row has no native request lease");
        if(epochs)for(size_t i=0;i<count;++i)if(data[i]!=0&&
            (!(*epochs)[i/chunk]||!requestRows_->Matches(i/chunk,(*epochs)[i/chunk])))
            throw std::invalid_argument("Active Metal row does not match the scheduled request epoch");
#endif
    }
#ifdef GARNET_METAL_DIRTY_PREFIX_RESET
    // Host controls were validated above. Mark before submission so a failed
    // command always takes the existing full-reset recovery path.
    size_t writeExtent=0;bool needExtent=false;
    for(size_t i=0;i<states_.size();++i)for(size_t k=0;k<2;++k)
        needExtent=needExtent||(states_[i].prefixLayout[k]&&states_[i].zeroOutsideDirty[k]&&layers[i]->CanResetKVPrefix(k+1));
    if(needExtent) {
        X::Tensor slotTensor(slots),flagTensor(active);auto sl=slotTensor.Acquire();auto fl=flagTensor.Acquire();
        const auto* slotData=static_cast<const float*>(slotTensor.Info().data);
        const auto* flagData=static_cast<const float*>(flagTensor.Info().data);
        for(size_t j=0;j<states_[0].rows*chunk;++j)if(flagData[j]!=0)writeExtent=std::max(writeExtent,size_t(slotData[j])+1);
    }
    for(size_t i=0;i<layers.size();++i)for(size_t k=0;k<2;++k) {
        auto& state=states_[i];
        if(!state.prefixLayout[k]||!state.zeroOutsideDirty[k])continue;
        if(!layers[i]->CanResetKVPrefix(k+1)){state.zeroOutsideDirty[k]=false;continue;}
        state.dirtyExtent[k]=std::max(state.dirtyExtent[k],writeExtent);
    }
#endif
    for(size_t i=0;i<layers.size();++i)layers[i]->RetireResidentDefaults(i==0);
#ifdef GARNET_METAL_VERIFY_ATTENTION
    if(verification)hidden_=verification->plan->allHidden;
    else
#endif
    if(hiddenShape_!=inputShape[0]) {
        const auto& slot=layers[0]->slots_[layers[0]->inputs_[0]];
        hidden_=gpu_->Allocate(slot.rows,slot.columns);
    }
    prefillPartialSubmissions_=0;fusedHeadArgmaxDispatches_=0;bodyShapeDispatches_=bodyShapeMisses_=0;deferredZeroBytes_=deferredZeroStates_=0;
    screenedHeadDispatches_=0;screenedHeadSeedRows_=0;interleavedScreenDispatches_=0;
#ifdef GARNET_METAL_ROW_FINAL_HIDDEN
    if(!rowFinalHidden_.Valid()) {
        const size_t width=size_t(inputShape[0].back());
        rowFinalHidden_=gpu_->Upload(std::vector<float>(states_[0].rows*width,0),states_[0].rows,width);
        rowFinalWidth_=width;finalHiddenReady_.assign(states_[0].rows,0);
    } else if(rowFinalWidth_!=size_t(inputShape[0].back()))throw std::invalid_argument("Last-active hidden width changed");
#endif
    const auto prepared=std::chrono::steady_clock::now();
    prepareMs_=std::chrono::duration<double,std::milli>(prepared-start).count();
#ifdef GARNET_METAL_DENSE_PREFIX_CACHE
    prefixFreshReset_=false;
#endif
#ifdef GARNET_METAL_MTP_BRIDGE
    if(mtp_) {
        X::Tensor tensor(slots);auto lease=tensor.Acquire();const auto* values=static_cast<const float*>(tensor.Info().data);
        for(size_t row=0;row<mtp_->batch;++row) {
            size_t next=mtp_->targetLengths[row];bool ended=false;
            for(size_t j=0;j<chunk;++j) {
                if(mtpRunFlags[row*chunk+j]==0)ended=true;
                else if(ended||values[row*chunk+j]!=float(next++))throw std::invalid_argument("MTP target requires contiguous native history");
            }
        }
        gpu_->Update(mtp_->captures.at(chunk).active,mtpRunFlags);
    }
#endif
    kvInplaceUpdates_=0;constantHeadDispatches_=0;singleCoderHeadDispatches_=0;residualNormDispatches_=0;packedPrefillNormDispatches_=0;globalHalfAlphaDispatches_=0;wideBatchPrefillDispatches_=0;interleavedBodyDispatches_=0;coalescedBlockDispatches_=0;mlpGraphDispatches_=0;joinedGlobalMlpBlocks_=0;poisoned_=true;gpu_->Begin();
    try {
#ifdef GARNET_METAL_VERIFY_ATTENTION
        if(verification)EncodeVerificationBackups(*verification,layers);
#endif
        auto current=layers[0]->slots_[layers[0]->inputs_[0]].matrix;
        for(size_t i=0;i<layers.size();++i) {
            auto graph=layers[i];std::array<MetalMatrix,3> saved;
            std::array<bool,2> inplace{false,false};
            std::vector<std::pair<size_t,MetalMatrix>> savedOutputs;
            for(size_t k=0;k<2;++k)if(graph->CanUpdateKVInPlace(k+1)) {
                const auto& group=graph->outputStorageGroups_[k+1];
                const auto outputId=graph->outputParts_[k+1];
                bool compatible=states_[i].matrix[k].CompatibleLayout(graph->slots_[outputId].matrix);
#ifdef GARNET_METAL_RETIRE_INPLACE_DEFAULTS
                if(!graph->slots_[outputId].matrix.Valid()&&
                   std::find(graph->lazyDefaultSlots_.begin(),graph->lazyDefaultSlots_.end(),outputId)!=graph->lazyDefaultSlots_.end())
                    compatible=gpu_->IsDefaultLayout(states_[i].matrix[k],graph->slots_[outputId].rows,graph->slots_[outputId].columns);
#endif
                if(!compatible)
                    throw std::logic_error("Private KV cache layout mismatch");
                for(auto id:group){savedOutputs.emplace_back(id,graph->slots_[id].matrix);}
                inplace[k]=true;
            }
            for(size_t k=0;k<2;++k)if(inplace[k])for(auto id:graph->outputStorageGroups_[k+1])
                graph->slots_[id].matrix=states_[i].matrix[k];
            for(size_t k=0;k<3;++k)saved[k]=graph->slots_[graph->inputs_[k]].matrix;
            graph->slots_[graph->inputs_[0]].matrix=current;
            graph->slots_[graph->inputs_[1]].matrix=states_[i].matrix[0];
            graph->slots_[graph->inputs_[2]].matrix=states_[i].matrix[1];
            auto restore=[&](){
                for(size_t k=0;k<3;++k)graph->slots_[graph->inputs_[k]].matrix=saved[k];
                for(auto& entry:savedOutputs)graph->slots_[entry.first].matrix=entry.second;
            };
            try {
#if defined(GARNET_METAL_INDEXED_TOKEN_PROJECTIONS)||defined(GARNET_METAL_ACTIVE_DECODE_PROJECTIONS)
                graph->Encode(*gpu_,false,nullptr,selectProjectionRows?&selectedTokenRows:nullptr
#ifdef GARNET_METAL_VERIFY_ATTENTION
                    ,verification?&verification->bindings[i]:nullptr
#endif
                    );
#else
                graph->Encode(*gpu_,false);
#endif
            }
            catch(...){restore();throw;}
            restore();
            kvInplaceUpdates_+=size_t(inplace[0])+size_t(inplace[1]);
            current=graph->slots_[graph->outputParts_[0]].matrix;
            // Keep the new state private to this session without copying it
            // back. The previous state becomes this graph's next output
            // storage; graph locks cover the exchange and GPU completion.
            for(size_t k=0;k<2;++k)if(!inplace[k]&&!graph->SwapOutputStorage(k+1,states_[i].matrix[k]))
                gpu_->EncodeSliceGroups(graph->slots_[graph->outputParts_[k+1]].matrix,
                    states_[i].matrix[k],1,states_[i].columns[k],0,states_[i].columns[k]);
#ifdef GARNET_METAL_PREFILL_STREAM
            if(chunk>1&&(i+1)%4==0&&i+1<layers.size())gpu_->Flush();
#endif
        }
#ifdef GARNET_METAL_ROW_FINAL_HIDDEN
        if(!verification)gpu_->EncodeLastActiveHidden(current,layers[0]->slots_[layers[0]->inputs_.back()].matrix,rowFinalHidden_,chunk);
#endif
        const size_t hiddenWidth=size_t(inputShape[0].back())*chunk;
#ifdef GARNET_METAL_MTP_BRIDGE
        if(mtp_)EncodeMTPTarget(current,chunk,verification!=nullptr);
#endif
        gpu_->EncodeSliceGroups(current,hidden_,1,hiddenWidth,0,hiddenWidth);
#ifdef GARNET_METAL_VERIFY_ATTENTION
        if(verification)EncodeVerificationHead(*verification,current);
#endif
        if(predictionToken>=0) {
            auto& headInput=head_->slots_[head_->inputs_[0]];
            gpu_->EncodeSliceGroups(current,headInput.matrix,1,chunk*headInput.columns,
                size_t(predictionToken)*headInput.columns,headInput.columns);
#ifdef GARNET_METAL_FUSED_HEAD_DECODE_ONLY
            if(chunk>1) {
                head_->Encode(*gpu_);
                gpu_->EncodeArgmax(head_->slots_[head_->output_].matrix,prediction_);
            } else
#endif
#ifdef GARNET_METAL_ACTIVE_DECODE_PROJECTIONS
            head_->EncodeGreedy(*gpu_,prediction_,chunk==1?&selectedTokenRows:nullptr);
#else
            head_->EncodeGreedy(*gpu_,prediction_);
#endif
        }
        const auto encoded=std::chrono::steady_clock::now();
        encodeMs_=std::chrono::duration<double,std::milli>(encoded-prepared).count();
        gpu_->SubmitAndWait();gpuMs_=gpu_->LastGPUMs();joinedPrefillPairs_=gpu_->LastJoinedPrefillPairs();constantHeadDispatches_=gpu_->LastConstantHeadDispatches();singleCoderHeadDispatches_=gpu_->LastSingleCoderHeadDispatches();residualNormDispatches_=gpu_->LastResidualNormDispatches();packedPrefillNormDispatches_=gpu_->LastPackedPrefillNormDispatches();globalHalfAlphaDispatches_=gpu_->LastGlobalHalfAlphaDispatches();wideBatchPrefillDispatches_=gpu_->LastWideBatchPrefillDispatches();interleavedBodyDispatches_=gpu_->LastInterleavedBodyDispatches();coalescedBlockDispatches_=gpu_->LastCoalescedBlockDispatches();mlpGraphDispatches_=gpu_->LastMlpGraphDispatches();joinedGlobalMlpBlocks_=gpu_->LastJoinedGlobalMlpBlocks();
        prefillPartialSubmissions_=gpu_->LastPartialSubmissions();
        screenedHeadDispatches_=gpu_->LastScreenedHeadDispatches();screenedHeadSeedRows_=gpu_->LastScreenedHeadSeedRows();interleavedScreenDispatches_=gpu_->LastInterleavedScreenDispatches();
        fusedHeadArgmaxDispatches_=gpu_->LastFusedHeadArgmaxDispatches();bodyShapeDispatches_=gpu_->LastBodyShapeDispatches();bodyShapeMisses_=gpu_->LastBodyShapeMisses();
        deferredZeroBytes_=gpu_->LastDeferredZeroBytes();deferredZeroStates_=gpu_->LastDeferredZeroStates();
#ifdef GARNET_METAL_VERIFY_ATTENTION
        if(verification)CommitVerification(*verification,layers);
#endif
        waitMs_=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-encoded).count();
    }catch(...){screenedHeadDispatches_=0;screenedHeadSeedRows_=0;interleavedScreenDispatches_=0;prefillPartialSubmissions_=0;fusedHeadArgmaxDispatches_=0;bodyShapeDispatches_=bodyShapeMisses_=0;deferredZeroBytes_=deferredZeroStates_=0;kvInplaceUpdates_=0;joinedPrefillPairs_=0;constantHeadDispatches_=0;singleCoderHeadDispatches_=0;residualNormDispatches_=0;packedPrefillNormDispatches_=0;globalHalfAlphaDispatches_=0;wideBatchPrefillDispatches_=0;interleavedBodyDispatches_=0;coalescedBlockDispatches_=0;mlpGraphDispatches_=0;joinedGlobalMlpBlocks_=0;gpu_->Abort();throw;}
    hiddenShape_=inputShape[0];
#ifdef GARNET_METAL_VERIFY_ATTENTION
    if(verification) {
        hidden_=verification->plan->finalHidden;
        hiddenShape_={int64_t(verification->plan->batch),int64_t(verification->plan->width)};
    }
#endif
    auto result=X::Value::List(Host());
    if(predictionToken>=0) {
#ifdef GARNET_METAL_REQUEST_ROWS
        predictionInitialized_=true;
#endif
        auto prediction=gpu_->Download(prediction_);
        for(size_t row=0;row<states_[0].rows;++row) {
            if(prediction[row*2+1]!=0)throw std::runtime_error("Nonfinite Metal session logits");
            result.Append(X::Value(int64_t(prediction[row*2])));
        }
    }
#ifdef GARNET_METAL_ROW_FINAL_HIDDEN
    for(size_t row=0;row<rowWillUpdate.size();++row)if(rowWillUpdate[row])finalHiddenReady_[row]=1;
#endif
#ifdef GARNET_METAL_TOKEN_PREFIX_CACHE
    if(tokenCache_) {
        X::Tensor flags(active);auto fl=flags.Acquire();const auto* data=static_cast<const float*>(flags.Info().data);
        for(size_t row=0;row<states_[0].rows;++row)for(size_t token=0;token<chunk;++token)if(
#ifdef GARNET_METAL_VERIFY_ATTENTION
            verification?verification->accepted[row*chunk+token]!=0:
#endif
            data[row*chunk+token]!=0) {
            if(tokenIds)rowTokens_[row].push_back((*tokenIds)[row*chunk+token]);
            else rowTokensValid_[row]=0;
        }
    }
#endif
    inferenceMs_=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
#ifdef GARNET_METAL_MTP_BRIDGE
    if(mtp_) {
        for(size_t row=0;row<mtp_->batch;++row) {
            size_t count=0;for(size_t j=0;j<chunk;++j)count+=size_t(mtpRunFlags[row*chunk+j]!=0);
            if(verification) {
                mtp_->targetLengths[row]+=size_t(verification->consumed[row]);
                if(verification->consumed[row])mtp_->targetPredictionReady[row]=1;
            }else {
                mtp_->targetLengths[row]+=count;
                if(predictionToken>=0)mtp_->targetPredictionReady[row]=uint8_t(count&&size_t(predictionToken)==count-1);
                else if(count)mtp_->targetPredictionReady[row]=0;
            }
        }
        if(!verification)mtp_->captures.at(chunk).flags=std::move(mtpRunFlags);
        mtp_->currentCapture=chunk;
    }
#endif
    poisoned_=false;return result;
}
#ifdef GARNET_METAL_ROW_FINAL_HIDDEN
X::Value MetalDecoderSession::PredictRows(X::Value epochs) {
    std::lock_guard<std::mutex> lock(mutex_);
    if(!gpu_||poisoned_||!head_||!requestRows_||!rowFinalHidden_.Valid()||
       epochs.runtime()!=Host()->runtime||!epochs.IsList()||epochs.Size()!=states_[0].rows)
        throw std::invalid_argument("Row prediction requires completed hidden states and request epochs");
    std::vector<bool> selected(states_[0].rows,false);bool any=false;
    for(size_t row=0;row<selected.size();++row) {
        auto value=epochs.Get(row);
        if(!value.IsInt64()||value.ToInt64()<0)throw std::invalid_argument("Invalid prediction epoch");
        const auto epoch=uint64_t(value.ToInt64());
        if(epoch&&(!requestRows_->Matches(row,epoch)||!finalHiddenReady_[row]))
            throw std::invalid_argument("Stale or uninitialized row prediction");
        selected[row]=epoch!=0;any=any||selected[row];
    }
    if(!any)throw std::invalid_argument("Row prediction requires an active request");
    std::lock_guard<std::mutex> headLock(head_->mutex_);
#ifdef GARNET_METAL_SHARED_PREFILL_SCRATCH
    std::unique_lock<std::mutex> scratchLock;
    if(head_->prefillScratch_)scratchLock=std::unique_lock<std::mutex>(head_->prefillScratch_->mutex);
#endif
    if(!head_->ready_||head_->generation_!=headGeneration_||head_->inputShapes_[0]!=
       std::vector<int64_t>{int64_t(states_[0].rows),int64_t(rowFinalWidth_)})
        throw std::logic_error("Row prediction head was closed, rebuilt or changed shape");
    poisoned_=true;
    try {
        gpu_->Begin();
        gpu_->EncodeSliceGroups(rowFinalHidden_,head_->slots_[head_->inputs_[0]].matrix,1,rowFinalWidth_,0,rowFinalWidth_);
#ifdef GARNET_METAL_ACTIVE_DECODE_PROJECTIONS
        std::vector<size_t> selectedRows;
        for(size_t row=0;row<selected.size();++row)if(selected[row])selectedRows.push_back(row);
        head_->EncodeGreedy(*gpu_,prediction_,&selectedRows);
#else
        head_->EncodeGreedy(*gpu_,prediction_);
#endif
        gpu_->SubmitAndWait();
        auto values=gpu_->Download(prediction_);auto result=X::Value::List(Host());
        predictionInitialized_=true;
        for(size_t row=0;row<selected.size();++row) {
            if(selected[row]&&values[row*2+1]!=0)throw std::runtime_error("Nonfinite row prediction logits");
            result.Append(X::Value(selected[row]?int64_t(values[row*2]):-1));
        }
#ifdef GARNET_METAL_MTP_BRIDGE
        if(mtp_)for(size_t row=0;row<selected.size();++row)mtp_->targetPredictionReady[row]=uint8_t(selected[row]);
#endif
        poisoned_=false;return result;
    } catch(...) {gpu_->Abort();throw;}
}
#endif
X::Value MetalDecoderSession::Snapshot(int64_t layer) {
    std::lock_guard<std::mutex> lock(mutex_);
    if(!gpu_||poisoned_||layer<0||uint64_t(layer)>=states_.size())throw std::invalid_argument("Metal session state snapshot invalid");
    auto result=X::Value::List(Host());const auto& state=states_[layer];
    for(size_t k=0;k<2;++k)result.Append(TensorValue(Host(),*gpu_,state.matrix[k],state.shape[k]));
    return result;
}
X::Value MetalDecoderSession::Hidden() {
    std::lock_guard<std::mutex> lock(mutex_);
    if(!gpu_||poisoned_||hiddenShape_.empty())throw std::logic_error("Metal session hidden unavailable");
    return TensorValue(Host(),*gpu_,hidden_,hiddenShape_);
}
}

namespace Garnet {
#ifdef GARNET_METAL_DENSE_PREFIX_CACHE
namespace {
constexpr size_t kPrefixCacheBytes=64*1024*1024;
std::atomic<int64_t> nextPrefixId{1};
int64_t NewPrefixId() {
    auto id=nextPrefixId.load();
    for(;;){if(id<=0||id==std::numeric_limits<int64_t>::max())throw std::overflow_error("Prefix cache IDs exhausted");
        if(nextPrefixId.compare_exchange_weak(id,id+1))return id;}
}
}
void MetalDecoderSession::ValidatePrefixOwners(std::vector<std::unique_lock<std::mutex>>& locks) {
    std::vector<MetalCompiledGraph*> graphs;
    for(size_t i=0;i<states_.size();++i)graphs.push_back(Graph(owners_.at(i),Host()));
    auto ordered=graphs;if(head_)ordered.push_back(head_);std::sort(ordered.begin(),ordered.end(),std::less<MetalCompiledGraph*>());
    if(std::adjacent_find(ordered.begin(),ordered.end())!=ordered.end())throw std::invalid_argument("Prefix owners must be distinct");
    for(auto graph:ordered)locks.emplace_back(graph->mutex_);
    for(size_t i=0;i<states_.size();++i){const auto& state=states_[i];auto graph=graphs[i];
        if(!state.prefixEligible||!graph->ready_||graph->generation_!=state.prefixOwnerGeneration)
            throw std::logic_error("Prefix cache requires unchanged dense-attention graph owners");
        for(size_t k=0;k<2;++k){const auto& shape=state.shape[k];
            if(shape.size()!=4||shape[0]!=1||shape[1]<=0||shape[2]<64||shape[3]<=0||state.rows!=1)
                throw std::invalid_argument("Prefix cache requires batch-one dense KV layout");}}
    if(head_&&(!head_->ready_||head_->generation_!=headGeneration_))throw std::logic_error("Prefix head owner changed");
}
#endif
int64_t MetalDecoderSession::CapturePrefix(int64_t tokens) {
    std::lock_guard<std::mutex> lock(mutex_);
#ifdef GARNET_METAL_MTP_BRIDGE
    if(mtp_)throw std::invalid_argument("MTP auxiliary prefix snapshots are not prepared");
#endif
#ifndef GARNET_METAL_DENSE_PREFIX_CACHE
    (void)tokens;throw std::logic_error("Dense prefix cache is not enabled");
#else
    if(!gpu_||poisoned_||prefixFreshReset_||states_.empty())throw std::logic_error("Capture requires a completed nonempty inference prefix");
    if(tokens!=64)throw std::invalid_argument("Only the validated64-token prefix is admitted");
    std::vector<std::unique_lock<std::mutex>> locks;ValidatePrefixOwners(locks);
    PrefixSnapshot snapshot;snapshot.tokens=64;
    for(const auto& state:states_)for(size_t k=0;k<2;++k){const auto& shape=state.shape[k];
        const auto count=size_t(shape[1])*64*size_t(shape[3]);
        if(count>kPrefixCacheBytes/sizeof(float)||snapshot.bytes>kPrefixCacheBytes-count*sizeof(float))
            throw std::invalid_argument("Prefix cache byte budget exceeded");
        snapshot.bytes+=count*sizeof(float);}
    for(const auto& state:states_){std::array<std::vector<float>,2> layer;
        for(size_t k=0;k<2;++k){const auto& shape=state.shape[k];layer[k]=gpu_->ReadPrefixGroups(state.matrix[k],size_t(shape[1]),size_t(shape[2]),size_t(shape[3]),64);
            if(!std::all_of(layer[k].begin(),layer[k].end(),[](float v){return std::isfinite(v);}))throw std::runtime_error("Cannot cache nonfinite KV state");}
        snapshot.layers.push_back(std::move(layer));}
    const auto id=NewPrefixId();snapshot.lastUse=++prefixClock_;const auto bytes=snapshot.bytes;
    prefixCache_.emplace(id,std::move(snapshot));prefixBytes_+=bytes;
    while(prefixCache_.size()>2||prefixBytes_>kPrefixCacheBytes){auto oldest=std::min_element(prefixCache_.begin(),prefixCache_.end(),[](const auto& a,const auto& b){return a.second.lastUse<b.second.lastUse;});prefixBytes_-=oldest->second.bytes;prefixCache_.erase(oldest);}
    prefixCaptureBytes_+=bytes;return id;
#endif
}
int64_t MetalDecoderSession::RestorePrefix(int64_t id) {
    std::lock_guard<std::mutex> lock(mutex_);
#ifdef GARNET_METAL_MTP_BRIDGE
    if(mtp_)throw std::invalid_argument("MTP auxiliary prefix snapshots are not prepared");
#endif
#ifndef GARNET_METAL_DENSE_PREFIX_CACHE
    (void)id;throw std::logic_error("Dense prefix cache is not enabled");
#else
    if(!gpu_||poisoned_||!prefixFreshReset_)throw std::logic_error("Restore requires a fresh successful full reset");
    auto found=prefixCache_.find(id);if(found==prefixCache_.end())throw std::invalid_argument("Unknown, evicted or foreign prefix handle");
    std::vector<std::unique_lock<std::mutex>> locks;ValidatePrefixOwners(locks);auto& snapshot=found->second;
    if(snapshot.layers.size()!=states_.size()||snapshot.tokens!=64)throw std::logic_error("Prefix metadata changed");
    for(size_t i=0;i<states_.size();++i)for(size_t k=0;k<2;++k){const auto& shape=states_[i].shape[k];
        if(snapshot.layers[i][k].size()!=size_t(shape[1])*64*size_t(shape[3]))throw std::logic_error("Prefix shape changed");}
    // All owners/layouts/counts are proved before any session state is written.
    prefixFreshReset_=false;poisoned_=true;
    try {for(size_t i=0;i<states_.size();++i)for(size_t k=0;k<2;++k){const auto& shape=states_[i].shape[k];
        gpu_->WritePrefixGroups(states_[i].matrix[k],size_t(shape[1]),size_t(shape[2]),size_t(shape[3]),64,snapshot.layers[i][k]);}
        poisoned_=false;snapshot.lastUse=++prefixClock_;prefixRestoreBytes_+=snapshot.bytes;return 64;
    }catch(...){throw;}
#endif
}
bool MetalDecoderSession::ClearPrefixCache() {std::lock_guard<std::mutex> lock(mutex_);
#ifdef GARNET_METAL_DENSE_PREFIX_CACHE
    prefixCache_.clear();prefixBytes_=prefixRestoreBytes_=prefixCaptureBytes_=0;
#endif
    return true;
}
size_t MetalDecoderSession::PrefixCacheBytes(){std::lock_guard<std::mutex> lock(mutex_);
#ifdef GARNET_METAL_DENSE_PREFIX_CACHE
    return prefixBytes_;
#else
    return 0;
#endif
}
size_t MetalDecoderSession::PrefixCacheEntries(){std::lock_guard<std::mutex> lock(mutex_);
#ifdef GARNET_METAL_DENSE_PREFIX_CACHE
    return prefixCache_.size();
#else
    return 0;
#endif
}
size_t MetalDecoderSession::PrefixRestoreBytes(){std::lock_guard<std::mutex> lock(mutex_);
#ifdef GARNET_METAL_DENSE_PREFIX_CACHE
    return prefixRestoreBytes_;
#else
    return 0;
#endif
}
size_t MetalDecoderSession::PrefixCaptureBytes(){std::lock_guard<std::mutex> lock(mutex_);
#ifdef GARNET_METAL_DENSE_PREFIX_CACHE
    return prefixCaptureBytes_;
#else
    return 0;
#endif
}
}

namespace Garnet {
std::vector<MetalResourceReference> MetalDecoderSession::MemoryReferences() {
    std::lock_guard<std::mutex> lock(mutex_);if(!gpu_)throw std::logic_error("Inventory requires an open session");std::vector<MetalResourceReference> result;
    auto append=[&](const MetalMatrix& matrix,size_t slot,const std::string& role,const std::vector<int64_t>& shape){auto info=MetalContext::MatrixBufferInfo(matrix);if(!info.identity)return;MetalResourceReference r;r.buffer=info;r.slot=slot;r.role=role;r.shape=shape;r.residentOperand=true;result.push_back(std::move(r));};
    for(size_t i=0;i<states_.size();++i)for(size_t k=0;k<2;++k)append(states_[i].matrix[k],i,"private_state"+std::to_string(k+1),states_[i].shape[k]);
#ifdef GARNET_METAL_ROW_FINAL_HIDDEN
    if(rowFinalHidden_.Valid())append(rowFinalHidden_,SIZE_MAX,"last_active_hidden",{int64_t(states_[0].rows),int64_t(rowFinalWidth_)});
#endif
    append(hidden_,SIZE_MAX,"hidden",hiddenShape_);append(prediction_,SIZE_MAX,"prediction",{});
#ifdef GARNET_METAL_VERIFY_ATTENTION
    if(verification_) {
        const auto& p=*verification_;
        for(size_t i=0;i<p.layers.size();++i){const auto& b=p.layers[i];
            append(b.qkv,i,"verification_retained_qkv",{});append(b.prepared,i,"verification_retained_prepared",{});
            append(b.replayConv,i,"verification_replay_conv_scratch",{});append(b.replayDelta,i,"verification_replay_delta_scratch",{});
            for(size_t k=0;k<2;++k)append(b.kvBackup[k],i,"verification_kv_backup"+std::to_string(k),{});}
        append(p.allHidden,SIZE_MAX,"verification_all_hidden",{});append(p.flatHidden,SIZE_MAX,"verification_flat_hidden",{});
        append(p.predictions,SIZE_MAX,"verification_predictions",{});append(p.packedPredictions,SIZE_MAX,"verification_packed_predictions",{});
        append(p.acceptedMask,SIZE_MAX,"verification_accepted_mask",{});append(p.replayMask,SIZE_MAX,"verification_replay_mask",{});
        append(p.finalHidden,SIZE_MAX,"verification_visible_hidden",{});
    }
#endif
#ifdef GARNET_METAL_MTP_BRIDGE
    if(mtp_) {
        const auto& m=*mtp_;
        for(size_t i=0;i<7;++i)append(m.inputs[i],i,"mtp_input",{});
        for(size_t i=0;i<2;++i){append(m.cache[i],i,"mtp_private_cache",{});append(m.backup[i],i,"mtp_draft_backup",{});}
        append(m.lastTarget,SIZE_MAX,"mtp_normalized_target_last",{});append(m.lastHidden,SIZE_MAX,"mtp_last_hidden",{});
        append(m.prediction,SIZE_MAX,"mtp_prediction",{});append(m.baseHidden,SIZE_MAX,"mtp_base_hidden",{});append(m.basePrediction,SIZE_MAX,"mtp_base_prediction",{});
        append(m.validation,SIZE_MAX,"mtp_validation_status",{});
        append(m.backupSlots,SIZE_MAX,"mtp_backup_slots",{});append(m.backupActive,SIZE_MAX,"mtp_backup_active",{});
        for(const auto& c:m.captures){append(c.second.flat,c.first,"mtp_target_flat",{});append(c.second.normalized,c.first,"mtp_target_normalized",{});append(c.second.packed,c.first,"mtp_target_packed",{});append(c.second.active,c.first,"mtp_target_active",{});}
        for(const auto& s:m.slices)append(s.second,s.first,"mtp_slice",{});
        for(const auto& s:m.zeros)append(s.second,s.first,"mtp_zero_tail",{});
    }
#endif
    return result;
}
}
