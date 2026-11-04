// SPDX-License-Identifier: Apache-2.0
#include "decoder_session.h"
#ifdef GARNET_METAL_MTP_BRIDGE
#include "lowering_context.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <stdexcept>
namespace Garnet {
namespace {
MetalCompiledGraph* MGraph(X::Value value,X3PackageHost* host) {
    if(value.runtime()!=host->runtime)throw std::invalid_argument("MTP graph runtime mismatch");
    auto p=value.NativeData<MetalCompiledGraph>();if(!p)throw std::invalid_argument("MTP requires compiled graph owners");return p;
}
X::Value HostCopy(X::Value value,X3PackageHost* host,const std::vector<int64_t>& shape,std::vector<float>* data=nullptr) {
    if(value.runtime()!=host->runtime)throw std::invalid_argument("MTP tensor runtime mismatch");
    X::Tensor tensor(value);auto info=tensor.Info();size_t count=1;for(auto d:shape)count*=size_t(d);
    if(info.symbolic||info.device_type||info.dtype!=X3_TENSOR_FLOAT32||info.rank!=shape.size()||!info.data||
       !std::equal(shape.begin(),shape.end(),info.shape)||info.byte_size<count*4||!IsContiguousTensor(tensor))
        throw std::invalid_argument("MTP host shape/storage mismatch");
    auto lease=tensor.Acquire();const auto* source=static_cast<const float*>(info.data);
    if(data)data->assign(source,source+count);
    return X::Tensor::Create(host,X3_TENSOR_FLOAT32,shape,source,count*4);
}
}
bool MetalDecoderSession::PrepareMTP(X::Value graphValue,X::Value headValue,X::Value chunks) {
    std::lock_guard<std::mutex> sessionLock(mutex_);
    if(!gpu_||poisoned_||!verification_||!hiddenShape_.empty()||mtp_||!head_)
        throw std::invalid_argument("Prepare MTP once on a fresh target with prepared verification");
#ifdef GARNET_METAL_TOKEN_PREFIX_CACHE
    if(tokenCache_)throw std::invalid_argument("MTP auxiliary prefix snapshots are not prepared");
#endif
    auto p=std::make_unique<MTPWorkspace>();p->graph=MGraph(graphValue,Host());p->head=MGraph(headValue,Host());
    p->graphOwner=graphValue;p->headOwner=headValue;
    if(p->graph==p->head||p->head==head_||p->graph==head_)throw std::invalid_argument("MTP graphs must be independent compiled owners");
    std::vector<MetalCompiledGraph*> ordered{p->graph,p->head,head_};std::sort(ordered.begin(),ordered.end(),std::less<MetalCompiledGraph*>());
    std::vector<std::unique_lock<std::mutex>> locks;for(auto g:ordered)locks.emplace_back(g->mutex_);
    auto graph=p->graph,head=p->head;
    if(!graph->ready_||!graph->mtpProgram_||!head->ready_||!head_->ready_||head_->generation_!=headGeneration_)
        throw std::invalid_argument("MTP graph/head owners are not ready");
    p->batch=verification_->batch;p->chunk=verification_->chunk;p->width=verification_->width;
    const auto b=p->batch,t=p->chunk,w=p->width;
    if(graph->inputShapes_[0]!=std::vector<int64_t>{int64_t(b),int64_t(t),int64_t(w)}||
       head->inputShapes_!=std::vector<std::vector<int64_t>>{{int64_t(b),int64_t(w)}}||head->ops_.size()!=1||head->weightRecords_.size()!=1)
        throw std::invalid_argument("MTP graph and projection-only head profiles disagree");
    const auto& projection=head->ops_[0];
    if(projection.kind!=MetalCompiledGraph::Kind::MatMul||!projection.transpose||!projection.indexedDecodeProjection||
       projection.a!=head->inputs_[0]||projection.out!=head->output_||!head->slots_[projection.b].affineInt4.Valid()||
       head->slots_[head->output_].columns!=verification_->vocab)
        throw std::invalid_argument("MTP head must use the target full-vocabulary FP32 affine projection only");
    for(const auto& record:head->weightRecords_) {
        auto original=head_->weightRecords_.find(record.first);
        if(original==head_->weightRecords_.end())throw std::invalid_argument("MTP head weight name differs from target");
        const auto a=MetalContext::AffineInt4BufferInfo(record.second.slot.affineInt4),z=MetalContext::AffineInt4BufferInfo(original->second.slot.affineInt4);
        if(a.size()!=z.size()||a.empty())throw std::invalid_argument("MTP head representation differs from target");
        for(size_t i=0;i<a.size();++i)if(a[i].identity!=z[i].identity)throw std::invalid_argument("MTP must share target head buffers");
    }
    size_t norms=0;
    for(const auto& op:head_->ops_)if(op.kind==MetalCompiledGraph::Kind::Norm) {
        if(op.a!=head_->inputs_[0]||head_->slots_[op.b].columns!=w)throw std::invalid_argument("MTP target normalization identity invalid");
        p->normWeight=head_->slots_[op.b].matrix;p->epsilon=op.epsilon;p->offset=op.offset;++norms;
    }
    if(norms!=1)throw std::invalid_argument("MTP requires exactly one target final normalization");
    if(chunks.runtime()!=Host()->runtime||!chunks.IsList()||!chunks.Size()||chunks.Size()>8)throw std::invalid_argument("Declare up to eight target capture chunk profiles");
    for(uint64_t i=0;i<chunks.Size();++i) {
        auto n=chunks.Get(i);if(!n.IsInt64()||n.ToInt64()<1||uint64_t(n.ToInt64())>kMetalPrefillChunkLimit)
            throw std::invalid_argument("MTP target capture profile exceeds native bounds");
        const auto size=size_t(n.ToInt64());if(p->captures.count(size))throw std::invalid_argument("Duplicate MTP capture profile");
        MTPCapture c;c.flat=gpu_->Allocate(b*size,w);c.normalized=gpu_->Allocate(b*size,w);c.packed=gpu_->Allocate(b,size*w);
        c.active=gpu_->Allocate(b,size);c.flags.assign(b*size,0);p->captures.emplace(size,std::move(c));
    }
    if(!p->captures.count(1)||!p->captures.count(t))throw std::invalid_argument("MTP capture profiles must include decode and verification");
    const auto& cacheShape=graph->inputShapes_[2];p->heads=size_t(cacheShape[1]);p->capacity=size_t(cacheShape[2]);p->dim=size_t(cacheShape[3]);
    bool cacheMatch=false;for(const auto& state:states_)if(state.shape[0]==cacheShape&&state.shape[1]==cacheShape)cacheMatch=true;
    if(!cacheMatch)throw std::invalid_argument("MTP cache profile differs from the target full attention");
    p->generation=graph->generation_;p->headGeneration=head->generation_;
    for(size_t i=0;i<7;++i)p->inputs[i]=gpu_->Allocate(graph->slots_[graph->inputs_[i]].rows,graph->slots_[graph->inputs_[i]].columns);
    for(auto& cache:p->cache)cache=gpu_->Allocate(b,p->heads*p->capacity*p->dim);
    for(auto& backup:p->backup)backup=gpu_->Allocate(b*t,p->heads*p->dim);
    p->lastTarget=gpu_->Allocate(b,w);p->lastHidden=gpu_->Allocate(b,w);p->prediction=gpu_->Allocate(b,2);
    p->validation=gpu_->Allocate(b,2);p->predictionReady.assign(b,0);p->baseReady.assign(b,0);p->nextPositions.assign(b*3,0);
    p->baseHidden=gpu_->Allocate(b,w);p->basePrediction=gpu_->Allocate(b,2);p->backupSlots=gpu_->Allocate(b,t);p->backupActive=gpu_->Allocate(b,t);
    for(size_t n=1;n<t;++n){p->slices[n]=gpu_->Allocate(b,n*w);p->zeros[n]=gpu_->Allocate(b,(t-n)*w);gpu_->Zero(p->zeros[n]);}
    p->committed.assign(b,0);p->tentative.assign(b,0);p->backupFlags.assign(b*t,0);
    p->targetLengths.assign(b,0);p->targetPredictionReady.assign(b,0);
    auto memory=MetalContext::MemoryStats();if(memory.recommended<=256*1024*1024||memory.allocated>memory.recommended-256*1024*1024)
        throw std::runtime_error("MTP bridge exceeds the native working-set guard");
    mtp_=std::move(p);ResetMTP();return true;
}
void MetalDecoderSession::ResetMTP() {
    if(!mtp_)return;auto& p=*mtp_;
    for(auto& m:p.cache)gpu_->Zero(m);
    for(auto& m:p.inputs)gpu_->Zero(m);
    for(auto m:{p.lastTarget,p.lastHidden,p.prediction,p.baseHidden,p.basePrediction})gpu_->Zero(m);
    for(auto& entry:p.captures){gpu_->Zero(entry.second.packed);std::fill(entry.second.flags.begin(),entry.second.flags.end(),0);}
    std::fill(p.committed.begin(),p.committed.end(),0);p.tentative=p.committed;
    std::fill(p.targetLengths.begin(),p.targetLengths.end(),0);std::fill(p.targetPredictionReady.begin(),p.targetPredictionReady.end(),0);
    std::fill(p.predictionReady.begin(),p.predictionReady.end(),0);p.baseReady=p.predictionReady;
    std::fill(p.nextPositions.begin(),p.nextPositions.end(),0);
    p.pending=false;p.currentCapture=0;p.teacherCalls=p.draftCalls=p.restoreCalls=p.headCalls=p.validationCalls=0;p.stats="{}";p.generationStats="{}";
}
void MetalDecoderSession::ResetMTPRow(size_t row) {
    if(!mtp_)return;auto& p=*mtp_;
    for(auto& m:p.cache)gpu_->ZeroRow(m,row);
    for(auto& m:p.inputs)gpu_->ZeroRow(m,row);
    for(auto m:{p.lastTarget,p.lastHidden,p.prediction,p.baseHidden,p.basePrediction})gpu_->ZeroRow(m,row);
    for(auto& entry:p.captures){gpu_->ZeroRow(entry.second.packed,row);std::fill_n(entry.second.flags.begin()+row*entry.first,entry.first,0);}
    p.committed[row]=p.tentative[row]=0;
    p.targetLengths[row]=0;p.targetPredictionReady[row]=0;
    p.predictionReady[row]=p.baseReady[row]=0;std::fill_n(p.nextPositions.begin()+row*3,3,0);
    std::fill_n(p.backupFlags.begin()+row*p.chunk,p.chunk,0);gpu_->Update(p.backupActive,p.backupFlags);
    p.pending=p.pending&&p.tentative!=p.committed;
}
void MetalDecoderSession::ValidateMTPTarget(size_t chunk) {
    if(!mtp_->captures.count(chunk))throw std::invalid_argument("Target chunk has no prepared MTP capture profile");
}
void MetalDecoderSession::EncodeMTPTarget(const MetalMatrix& hidden,size_t chunk,bool verification) {
    auto& p=*mtp_;auto& c=p.captures.at(chunk);
    gpu_->EncodeLayout(hidden,hidden,c.flat);gpu_->EncodeRMSNorm(c.flat,p.normWeight,c.normalized,p.epsilon,p.offset);
    gpu_->EncodeLayout(c.normalized,c.normalized,c.packed);
    if(!verification)gpu_->EncodeLastActiveHidden(c.packed,c.active,p.lastTarget,chunk);
}
void MetalDecoderSession::CommitMTPTarget(const MetalMatrix& active,const std::vector<float>& flags,size_t chunk) {
    auto& c=mtp_->captures.at(chunk);gpu_->EncodeLastActiveHidden(c.packed,active,mtp_->lastTarget,chunk);c.flags=flags;
}
X::Value MetalDecoderSession::MTPTeacher(X::Value embedding,X::Value pos,X::Value slots,X::Value active,int64_t offset,X::Value epochs) {
    return MTPStep(embedding,pos,slots,active,offset,epochs,false);
}
X::Value MetalDecoderSession::MTPDraft(X::Value embedding,X::Value pos,X::Value slots,X::Value active,X::Value epochs) {
    return MTPStep(embedding,pos,slots,active,-1,epochs,true);
}
X::Value MetalDecoderSession::MTPTeacherCache(X::Value embedding,X::Value pos,X::Value slots,X::Value active,int64_t offset,X::Value epochs) {
    return MTPStep(embedding,pos,slots,active,offset,epochs,false,false);
}
X::Value MetalDecoderSession::MTPStep(X::Value embedding,X::Value pos,X::Value slots,X::Value active,int64_t offset,X::Value epochs,bool draft,bool predict) {
    std::lock_guard<std::mutex> sessionLock(mutex_);
    return MTPStepUnlocked(embedding,pos,slots,active,offset,epochs,draft,predict);
}
X::Value MetalDecoderSession::MTPStepUnlocked(X::Value embedding,X::Value pos,X::Value slots,X::Value active,int64_t offset,X::Value epochs,bool draft,bool predict) {
    if(!mtp_||!gpu_||poisoned_||!mtp_->currentCapture)throw std::logic_error("MTP needs completed target hidden states");
    auto& p=*mtp_;auto graph=p.graph,head=p.head;
    std::vector<MetalCompiledGraph*> ordered{graph,head,head_};std::sort(ordered.begin(),ordered.end(),std::less<MetalCompiledGraph*>());
    std::vector<std::unique_lock<std::mutex>> locks;for(auto g:ordered)locks.emplace_back(g->mutex_);
#ifdef GARNET_METAL_SHARED_PREFILL_SCRATCH
    std::vector<MetalPrefillScratch*> pools;for(auto g:ordered)if(g->prefillScratch_)pools.push_back(g->prefillScratch_.get());
    std::sort(pools.begin(),pools.end(),std::less<MetalPrefillScratch*>());pools.erase(std::unique(pools.begin(),pools.end()),pools.end());
    std::vector<std::unique_lock<std::mutex>> poolLocks;for(auto pool:pools)poolLocks.emplace_back(pool->mutex);
#endif
    if(!graph->ready_||graph->generation_!=p.generation||!head->ready_||head->generation_!=p.headGeneration||!head_->ready_||head_->generation_!=headGeneration_)
        throw std::invalid_argument("Prepared MTP/target graph owner changed");
    const auto b=p.batch,t=p.chunk,w=p.width;auto& capture=p.captures.at(p.currentCapture);
    if(!draft&&(offset<0||uint64_t(offset)>=p.currentCapture))throw std::invalid_argument("MTP teacher source offset invalid");
    const size_t count=draft?1:std::min(t,p.currentCapture-size_t(offset));
    std::vector<float> flags,slotValues,positionValues;
    embedding=HostCopy(embedding,Host(),graph->inputShapes_[1]);pos=HostCopy(pos,Host(),graph->inputShapes_[4],&positionValues);
    slots=HostCopy(slots,Host(),graph->inputShapes_[5],&slotValues);active=HostCopy(active,Host(),graph->inputShapes_[6],&flags);
    std::vector<size_t> lengths(b,0);bool any=false;
    for(size_t row=0;row<b;++row) {
        bool ended=false;
        for(size_t j=0;j<t;++j){float flag=flags[row*t+j];if(flag!=0&&flag!=1)throw std::invalid_argument("MTP masks must be binary");
            if(flag==0)ended=true;else {
                if(ended||j>=count||(draft&&j>0))throw std::invalid_argument("MTP requires available contiguous source rows");
                if(!draft&&capture.flags[row*p.currentCapture+size_t(offset)+j]==0)throw std::invalid_argument("MTP teacher cannot read rejected/inactive target rows");
                ++lengths[row];any=true;
            }}
        const auto start=draft?p.tentative[row]:p.committed[row];
        if(!draft&&start+lengths[row]>p.targetLengths[row])throw std::invalid_argument("MTP teacher would advance beyond target history");
        if(draft&&lengths[row]&&(!p.predictionReady[row]||p.committed[row]==0||p.tentative[row]-p.committed[row]+lengths[row]>=t))throw std::invalid_argument("MTP draft needs a current prediction and bounded backup span");
        for(size_t j=0;j<lengths[row];++j)if(slotValues[row*t+j]!=float(start+j)||start+j>=p.capacity)
            throw std::invalid_argument("MTP cache slots disagree with native history");
    }
    if(draft&&!any)throw std::invalid_argument("MTP draft needs an active row");
#ifdef GARNET_METAL_CONTINUOUS_ROWS
    if(epochs.IsValid()&&!epochs.IsNone()) {
        if(epochs.runtime()!=Host()->runtime||!epochs.IsList()||epochs.Size()!=b)throw std::invalid_argument("MTP row epochs invalid");
        for(size_t row=0;row<b;++row){auto epoch=epochs.Get(uint64_t(row));if(!epoch.IsInt64()||epoch.ToInt64()<0||
            (epoch.ToInt64()&&(!requestRows_||!requestRows_->Matches(row,uint64_t(epoch.ToInt64()))))||
            (managedRows_&&lengths[row]&&!epoch.ToInt64()))throw std::invalid_argument("MTP stale/unowned row epoch");}
    }else if(managedRows_)throw std::invalid_argument("Managed MTP requires explicit row epochs");
#endif
    auto args=X::Value::List(Host());args.Append(X::Value());args.Append(embedding);args.Append(X::Value());args.Append(X::Value());args.Append(pos);args.Append(slots);args.Append(active);
    std::array<MetalMatrix,7> saved;for(size_t i=0;i<7;++i){auto& s=graph->slots_[graph->inputs_[i]];saved[i]=s.matrix;s.matrix=(i==2||i==3)?p.cache[i-2]:p.inputs[i];}
    auto restore=[&](){for(size_t i=0;i<7;++i)graph->slots_[graph->inputs_[i]].matrix=saved[i];};
    try{graph->PrepareInputs(args,{0,2,3});}catch(...){restore();throw;}
    const bool beginDraft=draft&&!p.pending;
    if(beginDraft) {
        std::vector<float> backupSlots(b*t);for(size_t row=0;row<b;++row)for(size_t j=0;j<t;++j){backupSlots[row*t+j]=float(p.committed[row]+j);p.backupFlags[row*t+j]=float(p.committed[row]+j<p.capacity);}
        gpu_->Update(p.backupSlots,backupSlots);gpu_->Update(p.backupActive,p.backupFlags);
    }
    poisoned_=true;gpu_->Begin();
    try {
        if(beginDraft) {
            for(size_t i=0;i<2;++i)gpu_->EncodeVerificationKVBackup(p.cache[i],p.backupSlots,p.backupActive,p.backup[i],p.heads,p.dim,p.capacity,t);
            gpu_->EncodeLayout(p.lastHidden,p.lastHidden,p.baseHidden);gpu_->EncodeLayout(p.prediction,p.prediction,p.basePrediction);
        }
        const auto& source=draft?p.lastHidden:capture.packed;const auto sourceTokens=draft?1:p.currentCapture;
        const auto start=draft?0:size_t(offset);
        if(count==t)gpu_->EncodeSliceGroups(source,p.inputs[0],1,sourceTokens*w,start*w,t*w);
        else {
            gpu_->EncodeSliceGroups(source,p.slices.at(count),1,sourceTokens*w,start*w,count*w);
            gpu_->EncodeConcat(p.slices.at(count),p.zeros.at(count),p.zeros.at(count),p.inputs[0],false);
        }
        graph->Encode(*gpu_,false);
        for(size_t i=0;i<2;++i)gpu_->EncodeLayout(graph->slots_[graph->outputParts_[i+1]].matrix,graph->slots_[graph->outputParts_[i+1]].matrix,p.cache[i]);
        if(!draft&&p.pending) {
            for(size_t i=0;i<2;++i)gpu_->EncodeVerificationKVRestore(p.backup[i],p.backupSlots,p.backupActive,p.inputs[6],p.cache[i],p.heads,p.dim,p.capacity,t);
            gpu_->EncodeLastActiveHidden(graph->slots_[graph->outputParts_[0]].matrix,p.inputs[6],p.baseHidden,t);
            gpu_->EncodeLayout(p.baseHidden,p.baseHidden,p.lastHidden);
        } else {
            gpu_->EncodeLastActiveHidden(graph->slots_[graph->outputParts_[0]].matrix,p.inputs[6],p.lastHidden,t);
        }
        if(predict) {
            auto& input=head->slots_[head->inputs_[0]];const auto old=input.matrix;input.matrix=p.lastHidden;
            try{head->EncodeGreedy(*gpu_,p.prediction);}catch(...){input.matrix=old;throw;}input.matrix=old;
        }else {
            if(p.pending)gpu_->EncodeLayout(p.basePrediction,p.basePrediction,p.prediction);
            gpu_->EncodeArgmax(p.lastHidden,p.validation);
        }
        gpu_->SubmitAndWait();restore();
        auto ids=gpu_->Download(predict?p.prediction:p.validation);auto result=X::Value::List(Host());
        for(size_t row=0;row<b;++row){if(ids[row*2+1]!=0||!std::isfinite(ids[row*2])||ids[row*2]<0||ids[row*2]>=(predict?verification_->vocab:w))
            throw std::runtime_error("Nonfinite/invalid MTP result");result.Append(X::Value(predict&&lengths[row]?int64_t(ids[row*2]):-1));}
        if(beginDraft)p.baseReady=p.predictionReady;
        if(!draft&&p.pending)p.predictionReady=p.baseReady;
        for(size_t row=0;row<b;++row)if(lengths[row]) {
            p.predictionReady[row]=uint8_t(predict);
            if(!draft)for(size_t axis=0;axis<3;++axis)p.nextPositions[row*3+axis]=positionValues[(row*t+lengths[row]-1)*3+axis];
        }
        if(predict)++p.headCalls;else ++p.validationCalls;
        if(draft){for(size_t row=0;row<b;++row)p.tentative[row]+=lengths[row];p.pending=true;++p.draftCalls;}
        else{for(size_t row=0;row<b;++row)p.committed[row]+=lengths[row];p.tentative=p.committed;if(p.pending)++p.restoreCalls;p.pending=false;++p.teacherCalls;}
        p.stats=nlohmann::json{{"committed",p.committed},{"tentative",p.tentative},{"pending",p.pending},{"teacher_calls",p.teacherCalls},
            {"draft_calls",p.draftCalls},{"corrections",p.restoreCalls},{"gpu_ms",gpu_->LastGPUMs()},
            {"head_calls",p.headCalls},{"validation_only_calls",p.validationCalls},{"prediction_ready",p.predictionReady},
            {"scope","Native MTP bridge; only argmax IDs/status read to CPU, no serving/speed qualification"}}.dump();
        poisoned_=false;return result;
    }catch(...){restore();gpu_->Abort();throw;}
}
X::Value MetalDecoderSession::MTPSnapshot() {
    std::lock_guard<std::mutex> lock(mutex_);if(!mtp_||poisoned_)throw std::logic_error("MTP snapshot requires a ready session");
    auto result=X::Value::List(Host());auto& p=*mtp_;
    for(auto matrix:{p.cache[0],p.cache[1],p.lastHidden,p.prediction}) {
        auto values=gpu_->Download(matrix);const auto info=MetalContext::MatrixBufferInfo(matrix);
        (void)info;const auto width=values.size()/p.batch;result.Append(X::Tensor::Create(Host(),X3_TENSOR_FLOAT32,{int64_t(p.batch),int64_t(width)},values.data(),values.size()*4));
    }return result;
}
std::string MetalDecoderSession::MTPStatsJson(){std::lock_guard<std::mutex> lock(mutex_);return mtp_?mtp_->stats:"{}";}
}
#endif
