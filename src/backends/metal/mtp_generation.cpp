// SPDX-License-Identifier: Apache-2.0
#include "decoder_session.h"
#ifdef GARNET_METAL_MTP_BRIDGE
#include "nlohmann/json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
namespace Garnet {
namespace {
std::vector<int64_t> ReadInts(X::Value value,X3PackageHost* host,size_t count) {
    if(value.runtime()!=host->runtime||!value.IsList()||value.Size()!=count)throw std::invalid_argument("Native generation integer-list profile mismatch");
    std::vector<int64_t> result;result.reserve(count);
    for(uint64_t i=0;i<count;++i){auto v=value.Get(i);if(!v.IsInt64())throw std::invalid_argument("Native generation requires integer controls");result.push_back(v.ToInt64());}return result;
}
X::Value IntList(X3PackageHost* host,const std::vector<int64_t>& values) {
    auto result=X::Value::List(host);for(auto v:values)result.Append(X::Value(v));return result;
}
X::Value FloatTensor(X3PackageHost* host,const std::vector<float>& values,const std::vector<int64_t>& shape) {
    return X::Tensor::Create(host,X3_TENSOR_FLOAT32,shape,values.data(),values.size()*4);
}
}
bool MetalDecoderSession::PrepareMTPEmbedding(const std::string& path) {
    std::lock_guard<std::mutex> lock(mutex_);
    if(!mtp_||poisoned_||mtp_->currentCapture||mtp_->embedding)throw std::invalid_argument("Bind native MTP embedding once before target execution");
    std::lock_guard<std::mutex> headLock(head_->mutex_);
    if(!head_->ready_||head_->generation_!=headGeneration_)throw std::invalid_argument("Target embedding head owner changed");
    auto embedding=std::make_unique<MTPEmbedding>();std::string error;
    if(!embedding->index.Open(path,error))throw std::invalid_argument(error);
    if(!std::filesystem::equivalent(embedding->index.FilePath(),head_->index_.FilePath()))
        throw std::invalid_argument("Native embedding must use the target checkpoint index");
    embedding->adapter.Open(embedding->index);
    const auto metadata=AffineInt4Metadata::Resolve(embedding->index,embedding->adapter,"model.language_model.embed_tokens.weight");
    const auto original=AffineInt4Metadata::Resolve(head_->index_,head_->mlxCheckpoint_,"model.language_model.embed_tokens.weight");
    if(metadata.layout.rows!=verification_->vocab||metadata.layout.columns!=mtp_->width)
        throw std::invalid_argument("Native embedding vocabulary/width differs from target");
    for(size_t i=0;i<3;++i) {
        const auto& a=*metadata.tensors[i];const auto& b=*original.tensors[i];
        if(!std::filesystem::equivalent(a.filePath,b.filePath)||a.dataOffset!=b.dataOffset||a.dataSize!=b.dataSize||a.dataType!=b.dataType||a.shape!=b.shape)
            throw std::invalid_argument("Native embedding tensor identity differs from target");
        embedding->dependencies.push_back({a.filePath,std::filesystem::last_write_time(a.filePath),std::filesystem::file_size(a.filePath)});
    }
    embedding->mapping.Open(metadata);
    for(const auto& d:embedding->dependencies)if(d.modified!=std::filesystem::last_write_time(d.path)||d.bytes!=std::filesystem::file_size(d.path))
        throw std::invalid_argument("Native embedding changed during mapping");
    mtp_->embedding=std::move(embedding);return true;
}
X::Value MetalDecoderSession::MTPEmbed(const std::vector<int64_t>& ids) {
    auto& p=*mtp_;if(!p.embedding||ids.size()!=p.batch*p.chunk)throw std::logic_error("Prepare the native target embedding map");
    for(const auto& d:p.embedding->dependencies)if(d.modified!=std::filesystem::last_write_time(d.path)||d.bytes!=std::filesystem::file_size(d.path))
        throw std::invalid_argument("Native embedding checkpoint changed");
    std::vector<float> values(ids.size()*p.width);const auto view=p.embedding->mapping.View();
    for(size_t i=0;i<ids.size();++i) {
        if(ids[i]<0||uint64_t(ids[i])>=verification_->vocab)throw std::invalid_argument("Native embedding token outside vocabulary");
        view.DecodeEmbeddingRow(size_t(ids[i]),values.data()+i*p.width,p.width);
    }
    return FloatTensor(Host(),values,{int64_t(p.batch),int64_t(p.chunk),int64_t(p.width)});
}
X::Value MetalDecoderSession::MTPGenerate(X::Value knownValue,X::Value remainingValue,X::Value stopsValue,X::Value epochsValue) {
    // One runtime transaction owns proposals, verification and teacher correction.
    std::lock_guard<std::mutex> lock(mutex_);const auto started=std::chrono::steady_clock::now();
    if(!mtp_||!verification_||!gpu_||poisoned_||!mtp_->embedding||mtp_->pending)
        throw std::logic_error("Native generation requires prepared, committed target and auxiliary engines");
    auto& p=*mtp_;const auto b=p.batch,t=p.chunk;const auto vocab=verification_->vocab;
    auto known=ReadInts(knownValue,Host(),b),remaining=ReadInts(remainingValue,Host(),b);
    if(stopsValue.runtime()!=Host()->runtime||!stopsValue.IsList()||stopsValue.Size()>vocab)throw std::invalid_argument("Native stop IDs invalid");
    auto stops=ReadInts(stopsValue,Host(),stopsValue.Size());stopsValue=IntList(Host(),stops);
    auto isStop=[&](int64_t id){return std::find(stops.begin(),stops.end(),id)!=stops.end();};
    for(auto id:stops)if(id<0||uint64_t(id)>=vocab)throw std::invalid_argument("Native stop token outside vocabulary");
    std::vector<int64_t> epochs;
    if(epochsValue.IsValid()&&!epochsValue.IsNone()){epochs=ReadInts(epochsValue,Host(),b);epochsValue=IntList(Host(),epochs);}
    for(auto epoch:epochs)if(epoch<0)throw std::invalid_argument("Native generation epochs must be nonnegative");
#ifdef GARNET_METAL_CONTINUOUS_ROWS
    if(managedRows_&&epochs.empty())throw std::invalid_argument("Managed native generation requires explicit epochs");
    for(size_t row=0;row<epochs.size();++row)if(epochs[row]<0||(epochs[row]&&(!requestRows_||!requestRows_->Matches(row,uint64_t(epochs[row])))))
        throw std::invalid_argument("Native generation row epoch is stale");
#endif
    std::vector<size_t> lengths(b,0);bool any=false;
    for(size_t row=0;row<b;++row) {
        if(known[row]<0||uint64_t(known[row])>=vocab||remaining[row]<0)throw std::invalid_argument("Native known token/output budget invalid");
        if(!remaining[row])continue;
        if(isStop(known[row])||p.committed[row]==0||p.committed[row]!=p.targetLengths[row]||!p.predictionReady[row]||!p.targetPredictionReady[row])
            throw std::invalid_argument("Native generation needs matching target/teacher history and current predictions");
        if(uint64_t(remaining[row])>p.capacity-p.committed[row])throw std::invalid_argument("Native generation output budget exceeds context capacity");
#ifdef GARNET_METAL_CONTINUOUS_ROWS
        if(managedRows_&&(!epochs[row]||!requestRows_->Owned(row)))throw std::invalid_argument("Native generation active row is unowned");
#endif
        lengths[row]=std::min(t,size_t(remaining[row]));any=true;
    }
    // Check every graph owner before the first state mutation, including shallow
    // requests that need no auxiliary draft invocation.
    {
        std::vector<MetalCompiledGraph*> graphs{p.graph,p.head,head_,verification_->head},layers;
        for(uint64_t i=0;i<verification_->graphs.Size();++i) {
            auto layer=verification_->graphs.Get(i).NativeData<MetalCompiledGraph>();layers.push_back(layer);graphs.push_back(layer);
            graphs.push_back(owners_[i].NativeData<MetalCompiledGraph>());
        }
        std::sort(graphs.begin(),graphs.end(),std::less<MetalCompiledGraph*>());graphs.erase(std::unique(graphs.begin(),graphs.end()),graphs.end());
        std::vector<std::unique_lock<std::mutex>> locks;for(auto graph:graphs)locks.emplace_back(graph->mutex_);
        if(!p.graph->ready_||p.graph->generation_!=p.generation||!p.head->ready_||p.head->generation_!=p.headGeneration||!head_->ready_||head_->generation_!=headGeneration_)
            throw std::invalid_argument("Native generation graph owner changed");
        VerificationRun check;check.plan=verification_.get();ValidateVerificationRun(check,layers);
    }
    // Even a caller-supplied known token must be the authoritative unconsumed
    // target prediction. No emitted token is fed twice or silently replaced.
    if(!any){auto result=X::Value::List(Host());for(size_t row=0;row<b;++row){auto item=X::Value::List(Host());item.Append(X::Value::List(Host()));item.Append(X::Value(int64_t(0)));item.Append(X::Value(false));result.Append(item);}return result;}
    auto targetIds=gpu_->Download(prediction_),draftIds=gpu_->Download(p.prediction);
    for(size_t row=0;row<b;++row)if(lengths[row]&&(targetIds[row*2+1]!=0||targetIds[row*2]!=float(known[row])))
        throw std::invalid_argument("Native known token differs from target prediction");
    const auto bases=p.committed;const auto positions=p.nextPositions;
    auto controls=[&](const std::vector<size_t>& starts,const std::vector<size_t>& sizes,size_t shift) {
        std::vector<float> pos(b*t*3),slots(b*t),active(b*t);
        for(size_t row=0;row<b;++row)for(size_t j=0;j<t;++j){slots[row*t+j]=float(starts[row]+j);active[row*t+j]=float(j<sizes[row]);
            for(size_t axis=0;axis<3;++axis)pos[(row*t+j)*3+axis]=positions[row*3+axis]+float(shift+j);}
        return std::array<X::Value,3>{FloatTensor(Host(),pos,{int64_t(b),int64_t(t),3}),FloatTensor(Host(),slots,{int64_t(b),int64_t(t)}),FloatTensor(Host(),active,{int64_t(b),int64_t(t)})};
    };
    std::vector<int64_t> proposals(b*t,0),nextDraft(b,0);for(size_t row=0;row<b;++row)if(lengths[row]) {
        if(draftIds[row*2+1]!=0||!std::isfinite(draftIds[row*2])||draftIds[row*2]<0||draftIds[row*2]>=vocab)
            throw std::invalid_argument("Native auxiliary prediction is invalid");
        proposals[row*t]=known[row];nextDraft[row]=int64_t(draftIds[row*2]);
    }
    bool mutated=false;size_t proposed=0;
    const auto oldHeads=p.headCalls,oldValidation=p.validationCalls;
    try {
        for(size_t j=1;j<t;++j) {
            std::vector<size_t> advance(b,0);std::vector<int64_t> inputIds(b*t,0);bool run=false;
            for(size_t row=0;row<b;++row)if(j<lengths[row]) {
                proposals[row*t+j]=nextDraft[row];++proposed;
                if(isStop(nextDraft[row]))lengths[row]=j+1;
                if(j+1<lengths[row]){advance[row]=1;inputIds[row*t]=nextDraft[row];run=true;}
            }
            if(run) {
                auto embedding=MTPEmbed(inputIds);auto ctl=controls(p.tentative,advance,j);
                mutated=true;auto result=MTPStepUnlocked(embedding,ctl[0],ctl[1],ctl[2],-1,epochsValue,true,true);
                for(size_t row=0;row<b;++row)if(advance[row])nextDraft[row]=result.Get(uint64_t(row)).ToInt64();
            }
        }
        auto hidden=MTPEmbed(proposals);auto ctl=controls(bases,lengths,0);
        mutated=true;auto result=VerifyGreedyUnlocked(hidden,ctl[0],ctl[1],ctl[2],IntList(Host(),proposals),stopsValue,IntList(Host(),remaining),epochsValue);
        std::vector<size_t> consumed(b,0);std::vector<int64_t> teacherIds(b*t,0);size_t accepted=0;bool predict=false;
        for(size_t row=0;row<b;++row) {
            auto item=result.Get(uint64_t(row)),ids=item.Get(uint64_t(0));consumed[row]=size_t(item.Get(uint64_t(1)).ToInt64());
            for(uint64_t j=0;j<ids.Size();++j)teacherIds[row*t+j]=ids.Get(j).ToInt64();
            if(consumed[row]){accepted+=consumed[row]-1;predict=predict||(remaining[row]>int64_t(consumed[row])&&!isStop(teacherIds[row*t+consumed[row]-1]));}
        }
        auto embedding=MTPEmbed(teacherIds);ctl=controls(bases,consumed,1);
        MTPStepUnlocked(embedding,ctl[0],ctl[1],ctl[2],0,epochsValue,false,predict);
        p.generationStats=nlohmann::json{{"proposals",proposals},{"submitted_lengths",lengths},{"consumed",consumed},{"proposed_drafts",proposed},{"accepted_drafts",accepted},
            {"auxiliary_head_calls",p.headCalls-oldHeads},{"auxiliary_validation_calls",p.validationCalls-oldValidation},
            {"wall_ms",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count()},
            {"scope","One native generation transaction; no benchmark admission"}}.dump();return result;
    }catch(...){if(mutated)poisoned_=true;throw;}
}
std::string MetalDecoderSession::MTPGenerationStatsJson(){std::lock_guard<std::mutex> lock(mutex_);return mtp_?mtp_->generationStats:"{}";}
}
#endif
