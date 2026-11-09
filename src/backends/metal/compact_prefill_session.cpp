// SPDX-License-Identifier: Apache-2.0
#include "decoder_session.h"
#ifdef GARNET_METAL_COMPACT_OWNER_PREFILL
#include "lowering_context.h"
#ifdef GARNET_METAL_WIDE_OWNER_PREFILL
#include "prefill_profile_policy.h"
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
namespace Garnet {
namespace {
MetalCompiledGraph* CGraph(X::Value value,X3PackageHost* host) {
    if(value.runtime()!=host->runtime)throw std::invalid_argument("Compact graph runtime mismatch");
    auto graph=value.NativeData<MetalCompiledGraph>();
    if(!graph)throw std::invalid_argument("Compact prefill requires native graph owners");return graph;
}
X::Value Rows(X::Value value,X3PackageHost* host,const std::vector<int64_t>& shape,size_t first,size_t count) {
    if(value.runtime()!=host->runtime)throw std::invalid_argument("Compact tensor runtime mismatch");
    X::Tensor tensor(value);auto lease=tensor.Acquire();auto info=tensor.Info();size_t width=1;
    for(size_t i=1;i<shape.size();++i)width*=size_t(shape[i]);
    if(info.symbolic||info.device_type||info.dtype!=X3_TENSOR_FLOAT32||info.rank!=shape.size()||
       !std::equal(shape.begin(),shape.end(),info.shape)||!info.data||!IsContiguousTensor(tensor)||
       info.byte_size!=size_t(shape[0])*width*4||first+count>size_t(shape[0]))
        throw std::invalid_argument("Compact physical tensor shape/storage mismatch");
    auto outputShape=shape;outputShape[0]=int64_t(count);
    return X::Tensor::Create(host,X3_TENSOR_FLOAT32,outputShape,static_cast<const float*>(info.data)+first*width,count*width*4);
}
}
#ifdef GARNET_METAL_COMPACT_VERIFY_COEXISTENCE
bool MetalDecoderSession::CompactVerificationPreparationSupported() const {
    // Session lock is held by the preparation caller. Inspect only private
    // session metadata here; graph owners are separately locked by each path.
#ifdef GARNET_METAL_WIDE_DENSE_VERIFICATION
    if(wideVerificationPreparationClosed_)return false;
#endif
    if(!gpu_||poisoned_||managedRows_||compactSealed_||compactRunning_||states_.size()!=32)return false;
    const std::vector<int64_t> dense{4,8,4096,128};
    for(const auto& state:states_)if(state.rows!=4||state.shape[0]!=dense||state.shape[1]!=dense)return false;
    return !verification_||(verification_->batch==4&&(verification_->chunk==1
#ifdef GARNET_METAL_WIDE_DENSE_VERIFICATION
        ||verification_->chunk==4
#endif
        )&&verification_->width==4096&&verification_->vocab==32768);
}
#endif
bool MetalDecoderSession::PrepareCompactPrefill(X::Value values,X::Value fallbackValues) {
    std::lock_guard<std::mutex> sessionLock(mutex_);
    if(!gpu_||poisoned_||states_.empty()||states_[0].rows!=4||managedRows_||compactSealed_||
#ifdef GARNET_METAL_WIDE_DENSE_VERIFICATION
       wideVerificationPreparationClosed_||
#endif
       values.runtime()!=Host()->runtime||fallbackValues.runtime()!=Host()->runtime||!values.IsList()||!fallbackValues.IsList()||
       values.Size()!=states_.size()||fallbackValues.Size()!=states_.size())
        throw std::invalid_argument("Prepare full compact families before any execution or physical row claim");
#ifdef GARNET_METAL_VERIFY_ATTENTION
    if(verification_) {
#ifdef GARNET_METAL_COMPACT_VERIFY_COEXISTENCE
        if(!CompactVerificationPreparationSupported())throw std::invalid_argument("Compact verification preparation exceeds the closed B4/T1 dense profile");
#else
        throw std::invalid_argument("Compact prefill with speculative target workspaces is not qualified");
#endif
    }
#endif
    std::vector<MetalCompiledGraph*> graphs,fallback,base,ordered;
    for(size_t i=0;i<states_.size();++i) {
        graphs.push_back(CGraph(values.Get(i),Host()));fallback.push_back(CGraph(fallbackValues.Get(i),Host()));base.push_back(CGraph(owners_[i],Host()));
        ordered.insert(ordered.end(),{graphs.back(),fallback.back(),base.back()});
    }
    std::sort(ordered.begin(),ordered.end(),std::less<MetalCompiledGraph*>());ordered.erase(std::unique(ordered.begin(),ordered.end()),ordered.end());
    std::vector<std::unique_lock<std::mutex>> locks;for(auto graph:ordered)locks.emplace_back(graph->mutex_);
    const auto& shape=graphs[0]->inputShapes_;
    constexpr size_t maximumChunk=
#ifdef GARNET_METAL_WIDE_OWNER_PREFILL
        512;
#else
        128;
#endif
    if(shape.empty()||shape[0].size()!=3||(shape[0][0]!=1&&shape[0][0]!=2)||shape[0][1]<2||
       size_t(shape[0][1])>maximumChunk||(size_t(shape[0][1])&(size_t(shape[0][1])-1))||
       size_t(shape[0][0])*size_t(shape[0][1])>512||shape[0][2]!=4096)
        throw std::invalid_argument("Compact prefill family exceeds its B1/B2 dense profile envelope");
    CompactFamily family;family.graphs=X::Value::List(Host());family.fallback=X::Value::List(Host());
    for(size_t i=0;i<graphs.size();++i){family.graphs.Append(values.Get(i));family.fallback.Append(fallbackValues.Get(i));}
    family.shape=shape[0];family.physicalShape=shape[0];family.physicalShape[0]=4;
    family.rows=size_t(shape[0][0]);family.chunk=size_t(shape[0][1]);family.width=size_t(shape[0][2]);
    family.fallbackChunk=std::min(family.chunk,size_t(128));
    auto fallbackShape=family.physicalShape;fallbackShape[1]=int64_t(family.fallbackChunk);
    if(compactFamilies_.count({family.rows,family.chunk}))throw std::invalid_argument("Compact family already prepared");
    for(size_t i=0;i<graphs.size();++i) {
        auto graph=graphs[i],source=base[i],full=fallback[i];const auto& state=states_[i];
        if(graph==source||graph==full||!graph->ready_||!source->ready_||!full->ready_||
           graph->inputShapes_.size()!=6||full->inputShapes_.size()!=6||source->inputShapes_.size()<3||
           graph->outputParts_.size()!=3||graph->inputShapes_[0]!=family.shape||
           full->inputShapes_[0]!=fallbackShape||source->inputShapes_[0]!=std::vector<int64_t>{4,4096}||
           !graph->SharesVerificationWeights(*source)||!full->SharesVerificationWeights(*source)||
           !graph->CanUpdateKVInPlace(1)||!graph->CanUpdateKVInPlace(2)||
           !full->CanUpdateKVInPlace(1)||!full->CanUpdateKVInPlace(2)||
           !source->CanUpdateKVInPlace(1)||!source->CanUpdateKVInPlace(2))
            throw std::invalid_argument("Compact family must retain exact original dense in-place weights and fallback shapes");
        for(size_t part=0;part<2;++part) {
            auto expected=state.shape[part];if(expected.size()!=4||expected[0]!=4||expected[2]<1||expected[2]>4096)
                throw std::invalid_argument("Compact families require full dense native KV geometry");
            expected[0]=int64_t(family.rows);
            if(graph->inputShapes_[part+1]!=expected||!source->CanResetKVPrefix(part+1)||!graph->CanResetKVPrefix(part+1))
                throw std::invalid_argument("Compact cache geometry differs from original private state");
            if(!family.capacity)family.capacity=size_t(expected[2]);
            if(family.capacity!=size_t(expected[2]))throw std::invalid_argument("Compact layers have different capacities");
        }
        family.generations.push_back(graph->generation_);family.fallbackGenerations.push_back(full->generation_);family.baseGenerations.push_back(source->generation_);
    }
    if(family.chunk>family.capacity)throw std::invalid_argument("Compact chunk exceeds its prepared context capacity");
    if(!rowFinalHidden_.Valid()) {
        rowFinalHidden_=gpu_->Upload(std::vector<float>(4*family.width,0),4,family.width);
        rowFinalWidth_=family.width;finalHiddenReady_.assign(4,0);
    }
    if(!compactDecodeHidden_.Valid())compactDecodeHidden_=gpu_->Allocate(4,family.width);
    const size_t packed=family.chunk*family.width;
    family.zero=gpu_->Upload(std::vector<float>(4*packed,0),4,packed);
    family.hidden=gpu_->Upload(std::vector<float>(4*packed,0),4,packed);
    for(size_t first=0;first+family.rows<=4;++first) {
        CompactSpan span;
        span.finalHidden=gpu_->PrepareKVRowSpan(rowFinalHidden_,first,family.rows,1,family.width,1);
        span.output=gpu_->PrepareKVRowSpan(family.hidden,first,family.rows,1,packed,1);
        for(size_t i=0;i<states_.size();++i) {
            State state=states_[i];state.rows=family.rows;
            for(size_t part=0;part<2;++part) {
                const auto& shape=states_[i].shape[part];state.shape[part][0]=int64_t(family.rows);
                state.matrix[part]=gpu_->PrepareKVRowSpan(states_[i].matrix[part],first,family.rows,size_t(shape[1]),size_t(shape[3]),size_t(shape[2]));
            }
            span.states.push_back(std::move(state));
        }
        family.spans.push_back(std::move(span));
    }
    for(size_t i=0;i<graphs.size();++i) {
        graphs[i]->RetireResidentDefaults();graphs[i]->SharePrefillScratch(prefillScratch_);
        if(i==0){graphs[i]->MaterializeDefault(graphs[i]->inputs_[0]);fallback[i]->MaterializeDefault(fallback[i]->inputs_[0]);base[i]->MaterializeDefault(base[i]->inputs_[0]);}
#ifdef GARNET_METAL_TOKEN_PREFIX_CACHE
        if(!tokenGraphs_.count(graphs[i]))tokenGraphOwners_.push_back(values.Get(i));
        tokenGraphs_[graphs[i]]={i,graphs[i]->generation_};
#endif
    }
    compactFamilies_.emplace(std::make_pair(family.rows,family.chunk),std::move(family));return true;
}
bool MetalDecoderSession::TryCompactPrefill(X::Value graphs,X::Value hidden,X::Value positions,X::Value slots,X::Value active,
    int64_t predictionToken,const std::vector<uint64_t>* epochs,const std::vector<int64_t>* tokenIds,VerificationRun* verification,X::Value& result) {
    if(compactFamilies_.empty()||!epochs||!managedRows_||predictionToken!=-1||verification||!gpu_||poisoned_)return false;
    X::Tensor mask(active);auto lease=mask.Acquire();auto info=mask.Info();
    constexpr size_t maximumChunk=
#ifdef GARNET_METAL_WIDE_OWNER_PREFILL
        512;
#else
        128;
#endif
    if(active.runtime()!=Host()->runtime||info.symbolic||info.device_type||info.dtype!=X3_TENSOR_FLOAT32||info.rank!=2||
       info.shape[0]!=4||info.shape[1]<2||size_t(info.shape[1])>maximumChunk||!info.data||!IsContiguousTensor(mask)||info.byte_size!=size_t(info.shape[1])*16)
        return false;
    const size_t chunk=size_t(info.shape[1]);const auto* flags=static_cast<const float*>(info.data);std::vector<size_t> rows;
    for(size_t row=0;row<4;++row) {
        bool live=false;for(size_t t=0;t<chunk;++t){float f=flags[row*chunk+t];if(f!=0&&f!=1)throw std::invalid_argument("Invalid compact physical active mask");live=live||f!=0;}
        if(live)rows.push_back(row);
    }
    if((rows.size()!=1&&rows.size()!=2)||rows.size()*chunk<=4)return false;
    if(rows.size()==2&&rows[1]!=rows[0]+1)return false;
    auto found=compactFamilies_.find({rows.size(),chunk});if(found==compactFamilies_.end())return false;
    auto& family=found->second;const size_t first=rows[0];
    if(epochs->size()!=4)throw std::invalid_argument("Compact execution requires all physical epochs");
    if(graphs.runtime()!=Host()->runtime||!graphs.IsList()||graphs.Size()!=family.fallback.Size())throw std::invalid_argument("Compact fallback list changed");
    for(size_t i=0;i<graphs.Size();++i)if(CGraph(graphs.Get(i),Host())!=CGraph(family.fallback.Get(i),Host()))
        throw std::invalid_argument("Compact execution requires its registered full-batch fallback family");
    // Validate every physical control, including inactive rows, before dropping
    // them from the compact tensors. Only native leased owners may advance.
    auto compactHidden=Rows(hidden,Host(),family.physicalShape,first,rows.size());
    auto compactPositions=Rows(positions,Host(),{4,int64_t(chunk),3},first,rows.size());
    auto compactSlots=Rows(slots,Host(),{4,int64_t(chunk)},first,rows.size());
    auto compactActive=Rows(active,Host(),{4,int64_t(chunk)},first,rows.size());
    X::Tensor fullPositions(positions),fullSlots(slots);auto pl=fullPositions.Acquire(),sl=fullSlots.Acquire();
    auto ps=static_cast<const float*>(fullPositions.Info().data),ss=static_cast<const float*>(fullSlots.Info().data);
    for(size_t i=0;i<4*chunk;++i) {
        if(!std::isfinite(ss[i])||ss[i]<0||ss[i]!=std::floor(ss[i])||(flags[i]!=0&&ss[i]>=family.capacity))throw std::invalid_argument("Invalid compact physical slot");
        for(size_t axis=0;axis<3;++axis)if(!std::isfinite(ps[i*3+axis]))throw std::invalid_argument("Invalid compact physical position");
        if(flags[i]!=0&&(!(*epochs)[i/chunk]||!requestRows_->Matches(i/chunk,(*epochs)[i/chunk])))throw std::invalid_argument("Stale compact physical row epoch");
    }
    std::vector<int64_t> compactIds;
    if(tokenIds){if(tokenIds->size()!=4*chunk)throw std::invalid_argument("Compact tracked token count differs");compactIds.assign(tokenIds->begin()+first*chunk,tokenIds->begin()+(first+rows.size())*chunk);}
    auto started=std::chrono::steady_clock::now();auto previousHidden=hidden_,previousFinal=rowFinalHidden_;auto previousShape=hiddenShape_;
    auto& span=family.spans.at(first);states_.swap(span.states);compactRunning_=&family;compactFirst_=first;
    hidden_=family.hidden;hiddenShape_=family.shape;rowFinalHidden_=span.finalHidden;
    auto restore=[&]{states_.swap(span.states);rowFinalHidden_=previousFinal;compactRunning_=nullptr;compactFirst_=0;};
    try {
        result=RunUnlocked(family.graphs,compactHidden,compactPositions,compactSlots,compactActive,-1,epochs,tokenIds?&compactIds:nullptr,nullptr);
        restore();hidden_=family.hidden;hiddenShape_=family.physicalShape;
        compactLastRows_=family.rows;compactLastFirst_=first;
        inferenceMs_=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();return true;
    }catch(...){restore();hidden_=previousHidden;hiddenShape_=previousShape;throw;}
}
#ifdef GARNET_METAL_WIDE_OWNER_PREFILL
size_t MetalDecoderSession::MaximumPreparedPrefillChunk() {
    std::lock_guard<std::mutex> lock(mutex_);size_t result=0;
    for(const auto& item:compactFamilies_)result=std::max(result,item.first.second);
    return result;
}
size_t MetalDecoderSession::SelectPreparedPrefillChunk(const std::vector<size_t>& rows,const std::vector<uint64_t>& epochs,
    const std::vector<size_t>& remaining,size_t maximum,size_t budget) {
    std::lock_guard<std::mutex> lock(mutex_);
    if(!gpu_||poisoned_||!requestRows_||rows.size()!=epochs.size()||rows.size()!=remaining.size())
        throw std::invalid_argument("Prepared scheduler profile requires a ready leased session");
    for(size_t i=0;i<rows.size();++i)if(!epochs[i]||!requestRows_->Matches(rows[i],epochs[i]))
        throw std::invalid_argument("Prepared scheduler profile has stale physical row epochs");
    std::vector<DenseOwnerPrefillFamily> families;
    for(const auto& item:compactFamilies_)families.push_back({item.second.rows,item.second.chunk,item.second.fallbackChunk});
    return SelectDenseOwnerPrefillChunk(rows,remaining,maximum,budget,families);
}
#endif
}
#endif
