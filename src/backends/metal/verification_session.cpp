// SPDX-License-Identifier: Apache-2.0
#include "decoder_session.h"
#ifdef GARNET_METAL_VERIFY_ATTENTION
#include "lowering_context.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace Garnet {
namespace {
MetalCompiledGraph* VGraph(const X::Value& value,X3PackageHost* host) {
    if(value.runtime()!=host->runtime)throw std::invalid_argument("Verification graph runtime mismatch");
    auto graph=value.NativeData<MetalCompiledGraph>();
    if(!graph)throw std::invalid_argument("Verification requires compiled graph owners");return graph;
}
std::vector<int64_t> Integers(X::Value values,X3PackageHost* host,size_t count,const char* name) {
    if(values.runtime()!=host->runtime||!values.IsList()||values.Size()!=count)
        throw std::invalid_argument(std::string("Verification ")+name+" length/runtime mismatch");
    std::vector<int64_t> result;result.reserve(count);
    for(uint64_t i=0;i<count;++i){auto value=values.Get(i);if(!value.IsInt64())throw std::invalid_argument(std::string("Verification ")+name+" must be integers");result.push_back(value.ToInt64());}return result;
}
X::Value ControlCopy(X::Value source,X3PackageHost* host,const std::vector<int64_t>& shape,std::vector<float>* copy=nullptr) {
    if(source.runtime()!=host->runtime)throw std::invalid_argument("Verification controls belong to another runtime");
    X::Tensor tensor(source);auto info=tensor.Info();size_t count=1;for(auto d:shape)count*=size_t(d);
    if(info.symbolic||info.device_type||info.dtype!=X3_TENSOR_FLOAT32||info.rank!=shape.size()||
       !std::equal(shape.begin(),shape.end(),info.shape)||!info.data||info.byte_size<count*4||!IsContiguousTensor(tensor))
        throw std::invalid_argument("Verification control shape/storage mismatch");
    auto lease=tensor.Acquire();const auto* data=static_cast<const float*>(info.data);
    if(copy)copy->assign(data,data+count);
    return X::Tensor::Create(host,X3_TENSOR_FLOAT32,shape,data,count*4);
}
}
bool MetalDecoderSession::PrepareVerification(X::Value values,X::Value headValue) {
#if !defined(GARNET_METAL_ROW_FINAL_HIDDEN)||!defined(GARNET_METAL_ACTIVE_DECODE_PROJECTIONS)||!defined(GARNET_METAL_INDEXED_TOKEN_PROJECTIONS)||!defined(GARNET_METAL_AFFINE_PROFILE)||!defined(GARNET_METAL_INDEXED_DECODE)||!defined(GARNET_METAL_MLX_AFFINE_INT4)
    throw std::logic_error("Native greedy verification requires prepared affine and row-hidden extensions");
#else
    std::lock_guard<std::mutex> sessionLock(mutex_);
#ifdef GARNET_METAL_MTP_BRIDGE
    if(mtp_)throw std::invalid_argument("Prepare verification before attaching MTP");
#endif
    if(!gpu_||poisoned_||states_.empty()||!head_||values.runtime()!=Host()->runtime||!values.IsList()||values.Size()!=states_.size())
        throw std::invalid_argument("Prepare verification on a ready decoder session with its full layer list");
    auto staged=std::make_unique<VerificationWorkspace>();staged->graphs=X::Value::List(Host());
    staged->head=VGraph(headValue,Host());staged->headOwner=headValue;
    if(staged->head==head_)throw std::invalid_argument("Verification needs a separate shared per-position head graph");
    std::vector<MetalCompiledGraph*> layers,ordered{head_,staged->head};
    for(size_t i=0;i<states_.size();++i){auto value=values.Get(uint64_t(i));auto graph=VGraph(value,Host());
        layers.push_back(graph);ordered.push_back(graph);ordered.push_back(VGraph(owners_[i],Host()));staged->graphs.Append(value);}
    auto uniqueLayers=layers;std::sort(uniqueLayers.begin(),uniqueLayers.end(),std::less<MetalCompiledGraph*>());
    if(std::adjacent_find(uniqueLayers.begin(),uniqueLayers.end())!=uniqueLayers.end())throw std::invalid_argument("Verification layers must be distinct");
    std::sort(ordered.begin(),ordered.end(),std::less<MetalCompiledGraph*>());ordered.erase(std::unique(ordered.begin(),ordered.end()),ordered.end());
    std::vector<std::unique_lock<std::mutex>> locks;for(auto graph:ordered)locks.emplace_back(graph->mutex_);
    if(!head_->ready_||head_->generation_!=headGeneration_||!staged->head->ready_||
       !staged->head->SharesVerificationWeights(*head_))throw std::invalid_argument("Verification head must share the current target head weights");
    auto first=layers[0];
    if(!first->ready_||!first->verificationTokens_||first->inputShapes_[0].size()!=3)throw std::invalid_argument("Prepare explicit compiled verification layers first");
    staged->batch=size_t(first->inputShapes_[0][0]);staged->chunk=first->verificationTokens_;staged->width=size_t(first->inputShapes_[0][2]);
    const auto b=staged->batch,t=staged->chunk,w=staged->width,r=b*t;
    if(!b||b>4||!t||r>4||staged->head->inputShapes_!=std::vector<std::vector<int64_t>>{{int64_t(r),int64_t(w)}}||
       head_->inputShapes_!=std::vector<std::vector<int64_t>>{{int64_t(b),int64_t(w)}})
        throw std::invalid_argument("Verification head/profile dimensions disagree");
    const auto& output=staged->head->slots_[staged->head->output_];
    if(output.rows!=r||!output.columns||output.columns>16777216||output.shape.size()!=2)
        throw std::invalid_argument("Verification head output must be bounded full vocabulary rows");
    staged->vocab=output.columns;staged->headGeneration=staged->head->generation_;
    size_t projections=0;
    for(const auto& op:staged->head->ops_)if(op.kind==MetalCompiledGraph::Kind::MatMul||op.kind==MetalCompiledGraph::Kind::BF16MatVec) {
        if(op.kind!=MetalCompiledGraph::Kind::MatMul||!op.transpose||!op.indexedDecodeProjection||
           !staged->head->slots_[op.b].affineInt4.Valid())throw std::invalid_argument("Verification head must retain FP32 affine decode math");++projections;
    }
    if(projections!=1)throw std::invalid_argument("Verification requires one target vocabulary projection");
    std::map<std::pair<size_t,size_t>,MetalMatrix> scratch;
    auto temporary=[&](size_t rows,size_t columns){auto key=std::make_pair(rows,columns);auto found=scratch.find(key);
        if(found==scratch.end())found=scratch.emplace(key,gpu_->Allocate(rows,columns)).first;return found->second;};
    for(size_t i=0;i<layers.size();++i) {
        auto graph=layers[i],source=VGraph(owners_[i],Host());auto& state=states_[i];
        if(!graph->ready_||graph->verificationTokens_!=t||graph->inputShapes_[0]!=first->inputShapes_[0]||
           graph->inputs_.size()!=6||graph->outputParts_.size()!=3||!source->ready_||
           graph->inputShapes_[1]!=state.shape[0]||graph->inputShapes_[2]!=state.shape[1]||!graph->SharesVerificationWeights(*source))
            throw std::invalid_argument("Verification graph/source state family mismatch");
        staged->generations.push_back(graph->generation_);staged->sourceGenerations.push_back(source->generation_);
        MetalCompiledGraph::VerificationBindings binding;
        size_t kvCount=0;
        for(size_t opIndex=0;opIndex<graph->ops_.size();++opIndex) {
            const auto& op=graph->ops_[opIndex];
            auto exactly=[&](size_t& index){if(index!=SIZE_MAX)throw std::invalid_argument("Verification capture has repeated state operations");index=opIndex;};
            if(op.kind==MetalCompiledGraph::Kind::ConvChunk)exactly(binding.conv);
            if(op.kind==MetalCompiledGraph::Kind::Prepare)exactly(binding.prepare);
            if(op.kind==MetalCompiledGraph::Kind::DeltaChunk)exactly(binding.delta);
            if(op.kind==MetalCompiledGraph::Kind::KVChunk){++kvCount;binding.kvDimensions={op.dimensions[0],op.dimensions[1],op.dimensions[2],op.dimensions[3]};}
        }
        if(binding.conv!=SIZE_MAX) {
            if(binding.prepare==SIZE_MAX||binding.delta==SIZE_MAX||kvCount||source->outputStorageGroups_.size()!=3||
               source->outputStorageGroups_[1].empty()||source->outputStorageGroups_[2].empty())
                throw std::invalid_argument("Verification recurrent capture requires independent source state-output banks");
            const auto& conv=graph->ops_[binding.conv];const auto& delta=graph->ops_[binding.delta];
            const auto& qkv=graph->slots_[conv.a];const auto& prepared=graph->slots_[delta.a];
            binding.qkv=gpu_->Allocate(qkv.rows,qkv.columns);binding.prepared=gpu_->Allocate(prepared.rows,prepared.columns);
            binding.replayConv=temporary(r,graph->slots_[conv.out].columns);binding.replayDelta=temporary(r,graph->slots_[delta.out].columns);
        }else {
            if(kvCount!=2||binding.prepare!=SIZE_MAX||binding.delta!=SIZE_MAX)throw std::invalid_argument("Verification requires linear capture or one K/V span pair");
            const auto kw=binding.kvDimensions[0]*binding.kvDimensions[1];
            for(auto& matrix:binding.kvBackup)matrix=gpu_->Allocate(r,kw);
        }
#ifdef GARNET_METAL_SHARED_PREFILL_SCRATCH
        graph->SharePrefillScratch(prefillScratch_);
#endif
#ifdef GARNET_METAL_TOKEN_PREFIX_CACHE
        if(!tokenGraphs_.count(graph))tokenGraphOwners_.push_back(staged->graphs.Get(uint64_t(i)));
        tokenGraphs_[graph]={i,graph->generation_};
#endif
        staged->layers.push_back(std::move(binding));
    }
    staged->allHidden=gpu_->Allocate(b,t*w);staged->flatHidden=gpu_->Allocate(r,w);staged->predictions=gpu_->Allocate(r,2);
    staged->packedPredictions=gpu_->Allocate(b,t*2);staged->acceptedMask=gpu_->Allocate(b,t);staged->replayMask=gpu_->Allocate(b,t);
    staged->finalHidden=gpu_->Allocate(b,w);gpu_->Zero(staged->finalHidden);
#ifdef GARNET_METAL_REQUEST_ROWS
    if(!predictionInitialized_)gpu_->Zero(prediction_);
#else
    gpu_->Zero(prediction_);
#endif
    if(!rowFinalHidden_.Valid()){rowFinalHidden_=gpu_->Allocate(b,w);gpu_->Zero(rowFinalHidden_);rowFinalWidth_=w;finalHiddenReady_.assign(b,0);}
    const auto memory=MetalContext::MemoryStats();
    if(memory.recommended<=256*1024*1024||memory.allocated>memory.recommended-256*1024*1024)
        throw std::runtime_error("Prepared verification workspace exceeds the native working-set guard");
    staged->preparedMetalBytes=memory.allocated;verification_=std::move(staged);verificationStats_="{}";return true;
#endif
}
void MetalDecoderSession::ValidateVerificationRun(VerificationRun& run,const std::vector<MetalCompiledGraph*>& layers) {
    auto& p=*run.plan;
    if(!p.head->ready_||p.head->generation_!=p.headGeneration||!p.head->SharesVerificationWeights(*head_))
        throw std::invalid_argument("Prepared verification head changed");
    run.bindings=p.layers;
    for(size_t i=0;i<layers.size();++i) {
        auto graph=layers[i],source=VGraph(owners_[i],Host());
        if(graph->generation_!=p.generations[i]||source->generation_!=p.sourceGenerations[i]||!source->ready_||
           !graph->ready_||!graph->SharesVerificationWeights(*source))throw std::invalid_argument("Prepared verification graph/source changed");
        auto& binding=run.bindings[i];if(binding.conv==SIZE_MAX)continue;
        // Re-resolve these banks under ALL graph locks on every call. Swapping
        // graph outputs with private session state is a permutation of distinct
        // banks; retaining one across calls could capture another session's state.
        auto& c=source->slots_[source->outputParts_[1]].matrix;auto& d=source->slots_[source->outputParts_[2]].matrix;
        if(!c.Valid()||!d.Valid()||!c.CompatibleLayout(states_[i].matrix[0])||!d.CompatibleLayout(states_[i].matrix[1])||c.SharesStorage(d))
            throw std::invalid_argument("Borrowed verification state bank layout changed");
        for(const auto& state:states_)for(const auto& matrix:state.matrix)if(c.SharesStorage(matrix)||d.SharesStorage(matrix))
            throw std::invalid_argument("Borrowed verification output aliases private state");
        for(const auto& layer:layers)for(auto part:layer->outputParts_)if(c.SharesStorage(layer->slots_[part].matrix)||d.SharesStorage(layer->slots_[part].matrix))
            throw std::invalid_argument("Borrowed verification bank aliases target outputs");
        binding.convS1=c;binding.deltaS1=d;
    }
}
void MetalDecoderSession::EncodeVerificationBackups(VerificationRun& run,const std::vector<MetalCompiledGraph*>& layers) {
    for(size_t i=0;i<layers.size();++i) {
        auto& binding=run.bindings[i];if(binding.conv!=SIZE_MAX)continue;auto graph=layers[i];const auto& dims=binding.kvDimensions;
        for(size_t k=0;k<2;++k)gpu_->EncodeVerificationKVBackup(states_[i].matrix[k],graph->slots_[graph->inputs_[4]].matrix,
            graph->slots_[graph->inputs_[5]].matrix,binding.kvBackup[k],dims[0],dims[1],dims[2],dims[3]);
    }
}
void MetalDecoderSession::EncodeVerificationHead(VerificationRun& run,const MetalMatrix& hidden) {
    auto& p=*run.plan;gpu_->EncodeLayout(hidden,hidden,p.flatHidden,0);auto& slot=p.head->slots_[p.head->inputs_[0]];const auto saved=slot.matrix;slot.matrix=p.flatHidden;
    try{p.head->EncodeGreedy(*gpu_,p.predictions);}catch(...){slot.matrix=saved;throw;}slot.matrix=saved;
}
void MetalDecoderSession::CommitVerification(VerificationRun& run,const std::vector<MetalCompiledGraph*>& layers) {
#if !defined(GARNET_METAL_ROW_FINAL_HIDDEN)||!defined(GARNET_METAL_AFFINE_PROFILE)||!defined(GARNET_METAL_INDEXED_DECODE)
    throw std::logic_error("Row-hidden support is required for verified commit");
#else
    auto& p=*run.plan;const auto predictions=gpu_->Download(p.predictions);
    const double targetGPU=gpu_->LastGPUMs();
    nlohmann::json stats{{"target_affine_dispatches",gpu_->LastAffineProfileDispatches()},{"target_affine_misses",gpu_->LastAffineProfileMisses()},
        {"target_indexed_decode_dispatches",gpu_->LastIndexedDecodeAffineDispatches()},{"target_conv_checkpoints",gpu_->LastVerificationConvCheckpoints()},
        {"target_delta_checkpoints",gpu_->LastVerificationDeltaCheckpoints()},{"target_kv_backups",gpu_->LastVerificationKVBackups()},
        {"target_attention_dispatches",gpu_->LastVerificationAttentionDispatches()},{"prepared_metal_bytes",p.preparedMetalBytes},{"target_gpu_ms",targetGPU},
        {"scope","Native target verification plus exact accepted-prefix commit; no draft model or speed qualification"}};
#ifdef GARNET_METAL_MPP_AFFINE
    stats["target_mpp_affine_dispatches"]=gpu_->LastMPPAffineDispatches();
#endif
#ifdef GARNET_METAL_INDEXED_AFFINE
    stats["target_indexed_prefill_dispatches"]=gpu_->LastIndexedAffineDispatches();
#endif
#ifdef GARNET_METAL_MPP_AFFINE_REUSE64
    stats["target_reuse64_prefill_dispatches"]=gpu_->LastReuse64AffineDispatches();
#endif
    bool rejected=false;
    for(size_t row=0;row<p.batch;++row) {
        size_t length=0;while(length<p.chunk&&run.active[row*p.chunk+length]!=0)++length;
        if(length>size_t(run.remaining[row]))throw std::logic_error("Submitted verification depth exceeds output budget");
        size_t consumed=0;
        for(size_t t=0;t<length;++t) {
            const auto index=(row*p.chunk+t)*2;
            if(predictions[index+1]!=0||!std::isfinite(predictions[index])||predictions[index]<0||predictions[index]>=p.vocab||
               predictions[index]!=std::floor(predictions[index]))throw std::runtime_error("Nonfinite or invalid native verification prediction");
            const auto token=int64_t(predictions[index]);run.emitted[row].push_back(token);consumed=t+1;
            if(std::find(run.stops.begin(),run.stops.end(),token)!=run.stops.end()){run.eos[row]=true;break;}
            if(t+1==length||token!=run.tokens[row*p.chunk+t+1])break;
        }
        run.consumed[row]=int64_t(consumed);rejected=rejected||consumed<length;
        for(size_t t=0;t<p.chunk;++t){run.accepted[row*p.chunk+t]=float(t<consumed);run.replay[row*p.chunk+t]=float(t>0&&t<consumed);}
    }
    gpu_->Update(p.acceptedMask,run.accepted.data(),run.accepted.size());gpu_->Update(p.replayMask,run.replay.data(),run.replay.size());
    gpu_->Begin();size_t replayLayers=0;
    if(rejected)for(size_t i=0;i<layers.size();++i) {
        auto graph=layers[i];auto& binding=run.bindings[i];
        if(binding.conv!=SIZE_MAX) {
            const auto& conv=graph->ops_[binding.conv];const auto& delta=graph->ops_[binding.delta];
            gpu_->EncodeCausalConvChunk(binding.qkv,graph->slots_[conv.b].matrix,binding.convS1,p.replayMask,binding.replayConv,states_[i].matrix[0],p.chunk,conv.dimensions[1]);
            gpu_->EncodeGatedDeltaChunk(binding.prepared,binding.deltaS1,p.replayMask,binding.replayDelta,states_[i].matrix[1],p.chunk,delta.dimensions[1],delta.dimensions[2],delta.dimensions[3]);++replayLayers;
        }else {
            const auto& dims=binding.kvDimensions;
            for(size_t k=0;k<2;++k)gpu_->EncodeVerificationKVRestore(binding.kvBackup[k],graph->slots_[graph->inputs_[4]].matrix,
                graph->slots_[graph->inputs_[5]].matrix,p.acceptedMask,states_[i].matrix[k],dims[0],dims[1],dims[2],dims[3]);
        }
    }
    gpu_->EncodeLastActiveHidden(p.allHidden,p.acceptedMask,rowFinalHidden_,p.chunk);
    gpu_->EncodeLayout(rowFinalHidden_,rowFinalHidden_,p.finalHidden,0);
    gpu_->EncodeLayout(p.predictions,p.predictions,p.packedPredictions,0);
    gpu_->EncodeLastActiveHidden(p.packedPredictions,p.acceptedMask,prediction_,p.chunk);
#ifdef GARNET_METAL_MTP_BRIDGE
    if(mtp_)CommitMTPTarget(p.acceptedMask,run.accepted,p.chunk);
#endif
    gpu_->SubmitAndWait();stats["commit_gpu_ms"]=gpu_->LastGPUMs();stats["sequential_gpu_duration_sum_ms"]=targetGPU+gpu_->LastGPUMs();
    stats["commit_replay_layers"]=replayLayers;stats["commit_kv_restores"]=gpu_->LastVerificationKVRestores();
    stats["consumed_inputs"]=run.consumed;stats["emitted_tokens"]=run.emitted;stats["eos"]=run.eos;
    stats["borrowed_s1_banks"]=size_t(std::count_if(run.bindings.begin(),run.bindings.end(),[](const auto& b){return b.conv!=SIZE_MAX;}))*2;
    verificationStats_=stats.dump();gpuMs_=targetGPU+gpu_->LastGPUMs();
#ifdef GARNET_METAL_REQUEST_ROWS
    predictionInitialized_=predictionInitialized_||std::any_of(run.consumed.begin(),run.consumed.end(),[](auto n){return n>0;});
#endif
#endif
}
X::Value MetalDecoderSession::VerifyGreedy(X::Value hidden,X::Value positions,X::Value slots,X::Value active,
    X::Value tokens,X::Value stops,X::Value remaining,X::Value epochs) {
    std::lock_guard<std::mutex> lock(mutex_);
    if(!verification_||!gpu_||poisoned_)throw std::logic_error("Prepare a ready native verification workspace first");
    auto& p=*verification_;VerificationRun run;run.plan=&p;
    run.tokens=Integers(tokens,Host(),p.batch*p.chunk,"input tokens");run.remaining=Integers(remaining,Host(),p.batch,"remaining budgets");
    if(stops.runtime()!=Host()->runtime||!stops.IsList())throw std::invalid_argument("Verification stop IDs must be a list");
    run.stops=Integers(stops,Host(),stops.Size(),"stop IDs");
    for(auto id:run.tokens)if(id<0||uint64_t(id)>=p.vocab)throw std::invalid_argument("Verification input token exceeds vocabulary");
    for(auto id:run.stops)if(id<0||uint64_t(id)>=p.vocab)throw std::invalid_argument("Verification stop token exceeds vocabulary");
    positions=ControlCopy(positions,Host(),{int64_t(p.batch),int64_t(p.chunk),3});
    slots=ControlCopy(slots,Host(),{int64_t(p.batch),int64_t(p.chunk)});
    active=ControlCopy(active,Host(),{int64_t(p.batch),int64_t(p.chunk)},&run.active);
    for(size_t row=0;row<p.batch;++row){size_t length=0;bool ended=false;
        for(size_t t=0;t<p.chunk;++t){float flag=run.active[row*p.chunk+t];if(flag!=0&&flag!=1)throw std::invalid_argument("Verification active masks must be binary");
            if(flag==0)ended=true;else{if(ended)throw std::invalid_argument("Verification active masks must be prefixes");++length;}}
        if(run.remaining[row]<0||length>uint64_t(run.remaining[row]))throw std::invalid_argument("Verification submitted depth exceeds remaining budget");}
    std::vector<uint64_t> expected;const std::vector<uint64_t>* lease=nullptr;
    if(epochs.IsValid()&&!epochs.IsNone()){auto ids=Integers(epochs,Host(),p.batch,"epochs");for(size_t row=0;row<ids.size();++row){auto id=ids[row];
        if(id<0)throw std::invalid_argument("Verification epochs must be nonnegative");
#ifdef GARNET_METAL_CONTINUOUS_ROWS
        if(id&&(!requestRows_||!requestRows_->Matches(row,uint64_t(id))))throw std::invalid_argument("Stale verification row epoch");
#endif
        expected.push_back(uint64_t(id));}lease=&expected;}
#ifdef GARNET_METAL_CONTINUOUS_ROWS
    if(managedRows_&&!lease)throw std::invalid_argument("Managed verification requires explicit row epochs");
#endif
    run.consumed.assign(p.batch,0);run.accepted.assign(p.batch*p.chunk,0);run.replay=run.accepted;run.emitted.resize(p.batch);run.eos.assign(p.batch,false);
    for(auto& row:run.emitted)row.reserve(p.chunk);
    const std::vector<int64_t>* tracked=nullptr;
#ifdef GARNET_METAL_TOKEN_PREFIX_CACHE
    if(tokenCache_)tracked=&run.tokens;
#endif
    RunUnlocked(p.graphs,hidden,positions,slots,active,-1,lease,tracked,&run);
    auto result=X::Value::List(Host());
    for(size_t row=0;row<p.batch;++row){auto item=X::Value::List(Host()),emitted=X::Value::List(Host());for(auto token:run.emitted[row])emitted.Append(X::Value(token));
        item.Append(emitted);item.Append(X::Value(run.consumed[row]));item.Append(X::Value(run.eos[row]));result.Append(item);}return result;
}
std::string MetalDecoderSession::VerificationStatsJson(){std::lock_guard<std::mutex> lock(mutex_);return verificationStats_;}
}
#endif
