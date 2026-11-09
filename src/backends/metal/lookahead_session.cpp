// SPDX-License-Identifier: Apache-2.0
#include "decoder_session.h"
#ifdef GARNET_METAL_LOOKAHEAD_RUNTIME
#include "nlohmann/json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <set>
#include <stdexcept>
namespace Garnet {
namespace {
std::vector<int64_t> LookaheadInts(X::Value value,X3PackageHost* host,size_t count) {
    if(value.runtime()!=host->runtime||!value.IsList()||value.Size()!=count)
        throw std::invalid_argument("Lookahead integer controls differ from native batch/runtime");
    std::vector<int64_t> result;result.reserve(count);
    for(size_t i=0;i<count;++i){auto item=value.Get(uint64_t(i));if(!item.IsInt64())throw std::invalid_argument("Lookahead controls must be integers");result.push_back(item.ToInt64());}
    return result;
}
}
void MetalDecoderSession::ResetLookaheadRow(size_t row) {
    if(lookahead_&&row<lookahead_->layout.Batch()&&lookahead_->owners[row])lookahead_->owners[row]->Invalidate();
}
MetalDecoderSession::LookaheadLayer MetalDecoderSession::CompileLookaheadLayer(
    MetalCompiledGraph& source,LookaheadWorkspace& w,size_t layer,bool head) {
    using K=MetalCompiledGraph::Kind;const size_t batch=w.layout.Batch(),rows=batch*16;
    if(!source.ready_||source.verificationTokens_||source.inputShapes_.empty()||
       source.inputShapes_[0]!=std::vector<int64_t>{int64_t(batch),4096}||!source.mlxCheckpoint_.DenseMistral())
        throw std::invalid_argument("Lookahead expansion requires a ready ordinary dense Mistral AR graph");
    auto canonical=[&](size_t id){
        size_t steps=0;
        while(source.lookaheadAliases_.count(id)) {
            if(++steps>source.slots_.size())throw std::invalid_argument("Cyclic lookahead source aliases");
            const size_t target=source.lookaheadAliases_.at(id);
            const auto& view=source.slots_.at(id);const auto& backing=source.slots_.at(target);
            if(view.rows!=backing.rows||view.columns!=backing.columns)throw std::invalid_argument("Lookahead source alias changes physical matrix geometry");
            id=target;
        }
        if(id>=source.slots_.size())throw std::invalid_argument("Lookahead source slot out of range");
        return id;
    };
    for(auto input:source.inputs_)if(canonical(input)!=input)throw std::invalid_argument("Lookahead source input unexpectedly aliases another slot");
    auto ops=source.ops_;
    for(auto& op:ops){op.a=canonical(op.a);op.b=canonical(op.b);op.out=canonical(op.out);for(auto& id:op.arguments)id=canonical(id);}
    LookaheadLayer result;result.source=&source;result.generation=source.generation_;
    result.input=source.inputs_[0];result.output=canonical(head?source.output_:source.outputParts_.at(0));
    if(source.slots_[result.output].rows!=batch||source.slots_[result.output].columns!=(head?32768:4096))
        throw std::invalid_argument("Lookahead AR output shape differs from compiled model profile");
    size_t key=SIZE_MAX,value=SIZE_MAX,scores=SIZE_MAX,context=SIZE_MAX;
    if(head) {
        if(source.inputs_.size()!=1)throw std::invalid_argument("Lookahead head requires one hidden input");
    }else {
        const std::vector<int64_t> cache{int64_t(batch),8,int64_t(w.layout.Capacity()),128};
        if(source.inputs_.size()!=6||source.outputParts_.size()!=3||source.inputShapes_[1]!=cache||source.inputShapes_[2]!=cache||
           source.inputShapes_[3]!=std::vector<int64_t>{int64_t(batch),3})
            throw std::invalid_argument("Lookahead source graph cache/control family mismatch");
        for(size_t i=0;i<ops.size();++i) {
            const auto& op=ops[i];
            auto unique=[&](size_t& target){if(target!=SIZE_MAX)throw std::invalid_argument("Repeated attention state op in lookahead source");target=i;};
            if(op.kind==K::KVUpdate) {
                if(op.arguments.size()!=5||op.arguments[2]!=source.inputs_[4]||op.arguments[3]!=source.inputs_[5]||op.arguments[4]!=source.inputs_[3]||
                   op.dimensions!=std::vector<size_t>{8,128,w.layout.Capacity(),1})throw std::invalid_argument("Lookahead source KV update contract mismatch");
                if(op.arguments[1]==source.inputs_[1])unique(key);
                else if(op.arguments[1]==source.inputs_[2])unique(value);
                else throw std::invalid_argument("Lookahead source has an unrelated KV bank");
            }else if(op.kind==K::Scores)unique(scores);
            else if(op.kind==K::AttentionContext)unique(context);
        }
        if(key==SIZE_MAX||value==SIZE_MAX||scores==SIZE_MAX||context==SIZE_MAX)
            throw std::invalid_argument("Lookahead requires one complete AR attention block");
        const auto& score=ops[scores];const auto& attend=ops[context];
        if(score.arguments.size()!=4||attend.arguments.size()!=4||score.arguments[1]!=ops[key].out||
           score.arguments[2]!=source.inputs_[4]||score.arguments[3]!=source.inputs_[5]||
           attend.arguments[0]!=score.out||attend.arguments[1]!=ops[value].out||attend.arguments[3]!=source.inputs_[4]||
           score.dimensions!=std::vector<size_t>{32,8,128,w.layout.Capacity()}||
           attend.dimensions!=std::vector<size_t>{32,8,128,w.layout.Capacity(),0})
            throw std::invalid_argument("Lookahead AR attention connections/geometry mismatch");
        result.key=ops[key].arguments[0];result.value=ops[value].arguments[0];
    }
    // The logical program is one AR-step invocation per DAG node. Only its
    // physical attention/state implementation is fused. Generic linear chunk
    // graphs are deliberately not accepted or silently reinterpreted here.
    std::vector<LookaheadInstruction> expanded;
    for(size_t i=0;i<ops.size();++i) {
        if(!head&&(i==key||i==value||i==scores))continue;
        auto op=ops[i];const bool attention=!head&&i==context;
        if(attention)op.arguments={ops[scores].arguments[0],result.key,result.value,source.inputs_[1],source.inputs_[2]};
        expanded.push_back({std::move(op),attention});
    }
    auto operands=[&](const LookaheadInstruction& instruction) {
        const auto& op=instruction.op;if(instruction.attention)return op.arguments;
        switch(op.kind) {
        case K::MatMul:case K::Norm:case K::Silu:case K::Add:case K::Bias:case K::Mask:case K::Layout:return std::vector<size_t>{op.a,op.b};
        case K::NormRope:return op.arguments;
        case K::Slice:return std::vector<size_t>{op.a};
        default:throw std::invalid_argument("Unsupported numerical op in lookahead AR DAG expansion");
        }
    };
    std::set<size_t> needed{result.output},written;
    std::vector<LookaheadInstruction> reverse;
    for(auto it=expanded.rbegin();it!=expanded.rend();++it)if(needed.count(it->op.out)) {
        if(!written.insert(it->op.out).second)throw std::invalid_argument("Lookahead AR slot has multiple writers");
        auto ids=operands(*it);needed.insert(ids.begin(),ids.end());reverse.push_back(*it);
    }
    result.program.assign(reverse.rbegin(),reverse.rend());result.matrices.resize(source.slots_.size());
    size_t attentions=0,projections=0;
    for(auto id:needed) {
        const auto& slot=source.slots_.at(id);
        if(slot.affineInt4.Valid())continue;
        if(id==result.input)result.matrices[id]=w.input;
        else if(!head&&(id==source.inputs_[1]||id==source.inputs_[2]))result.matrices[id]=MetalMatrix(); // Borrowed only during encoding.
        else if(!head&&id==source.inputs_[3])result.matrices[id]=w.positions;
        else if(!head&&id==source.inputs_[5])result.matrices[id]=w.active;
        else if(written.count(id)) {
            if(slot.rows!=batch||!slot.columns||slot.columns>32768)throw std::invalid_argument("Lookahead AR temporary cannot be row-expanded");
            if(!head&&id==result.output)result.matrices[id]=w.hidden[layer%2];
            else result.matrices[id]=gpu_->Allocate(rows,slot.columns);
        }else {
            if(!slot.matrix.Valid())throw std::invalid_argument("Unbound lookahead AR constant/control slot");
            result.matrices[id]=slot.matrix;
        }
    }
    for(const auto& instruction:result.program) {
        const auto& op=instruction.op;
        if(instruction.attention){++attentions;continue;}
        if(op.kind==K::MatMul) {
            if(!op.transpose||!source.slots_[op.b].affineInt4.Valid())throw std::invalid_argument("Lookahead requires exact native affine target projections");
            gpu_->PrepareLookaheadProjection(result.matrices[op.a],source.slots_[op.b].affineInt4,result.matrices[op.out],w.layout);++projections;
        }else if(op.kind==K::NormRope) {
            if(head||op.arguments.size()!=4||op.arguments[2]!=source.inputs_[3]||op.dimensions.size()!=4||
               (op.dimensions[0]!=32&&op.dimensions[0]!=8)||op.dimensions[1]!=128||op.dimensions[2]!=128)
                throw std::invalid_argument("Lookahead RoPE program differs from source text model");
        }
    }
    if(attentions!=size_t(!head)||projections!=(head?1:7))throw std::invalid_argument("Lookahead source projection/attention count mismatch layer="+std::to_string(layer)+" head="+std::to_string(head)+" projections="+std::to_string(projections)+" attention="+std::to_string(attentions));
    return result;
}
void MetalDecoderSession::EncodeLookaheadLayer(LookaheadLayer& layer,LookaheadWorkspace& w,const MetalMatrix& input,
    const std::array<MetalMatrix,2>* prefix,const std::vector<LookaheadLayout::View>& views) {
    using K=MetalCompiledGraph::Kind;auto& m=layer.matrices;auto* source=layer.source;m[layer.input]=input;
    if(prefix){m[source->inputs_[1]]=(*prefix)[0];m[source->inputs_[2]]=(*prefix)[1];}
    struct PrefixBorrow {
        std::vector<MetalMatrix>& matrices;size_t key,value;
        ~PrefixBorrow(){if(key!=SIZE_MAX){matrices[key]=MetalMatrix();matrices[value]=MetalMatrix();}}
    } borrow{m,prefix?source->inputs_[1]:SIZE_MAX,prefix?source->inputs_[2]:SIZE_MAX};
    for(const auto& instruction:layer.program) {
        const auto& op=instruction.op;
        if(instruction.attention) {
            std::vector<MetalMatrix> args;for(auto id:op.arguments)args.push_back(m.at(id));
            gpu_->EncodeLookaheadAttention(args,w.probabilities,m[op.out],w.layout,views);continue;
        }
        switch(op.kind) {
        case K::MatMul:gpu_->EncodeLookaheadProjection(m[op.a],source->slots_[op.b].affineInt4,m[op.out],w.layout,views);break;
        case K::Norm:gpu_->EncodeRMSNorm(m[op.a],m[op.b],m[op.out],op.epsilon,op.offset);break;
        case K::Silu:gpu_->EncodeSiluMul(m[op.a],m[op.b],m[op.out]);break;
        case K::Add:gpu_->EncodeAdd(m[op.a],m[op.b],m[op.out]);break;
        case K::Bias:gpu_->EncodeBias(m[op.a],m[op.b],m[op.out]);break;
        case K::Mask:gpu_->EncodeMaskRows(m[op.a],m[op.b],m[op.out]);break;
        case K::Layout:gpu_->EncodeLayout(m[op.a],m[op.b],m[op.out],op.dimensions.at(0));break;
        case K::Slice:gpu_->EncodeSliceGroups(m[op.a],m[op.out],op.dimensions.at(0),op.dimensions.at(1),op.dimensions.at(2),op.dimensions.at(3));break;
        case K::NormRope:gpu_->EncodeNormRope(m[op.arguments[0]],m[op.arguments[1]],m[op.arguments[2]],m[op.arguments[3]],
            m[op.out],op.dimensions[0],op.dimensions[1],op.dimensions[2],op.epsilon,op.offset,op.dimensions[3]!=0);break;
        default:throw std::logic_error("Unprepared lookahead DAG instruction");
        }
    }
}
bool MetalDecoderSession::PrepareLookahead(int64_t window,int64_t levels,int64_t guesses) {
    std::lock_guard<std::mutex> sessionLock(mutex_);
    if(!gpu_||poisoned_||lookahead_||!head_||states_.size()!=32||!tokenCache_||!requestRows_||
       tokenVocabulary_!=32768||!tokenCapacity_||tokenCapacity_>4096||window<1||levels<2||guesses<1)
        throw std::invalid_argument("Prepare native lookahead once on a ready tracked dense Mistral session");
    gpu_->RequireDrainedProfileControl();
    for(size_t row=0;row<states_[0].rows;++row)if(requestRows_->Owned(row)||(finalHiddenReady_.size()>row&&finalHiddenReady_[row]))
        throw std::logic_error("Prepare lookahead before claiming or executing requests");
#ifdef GARNET_METAL_WIDE_DENSE_VERIFICATION
    if(wideVerificationPreparationClosed_)throw std::logic_error("Lookahead preparation closed after first native execution");
#endif
#ifdef GARNET_METAL_MTP_BRIDGE
    if(mtp_)throw std::invalid_argument("Lookahead auxiliary state coexistence is not qualified");
#endif
    std::vector<std::unique_lock<std::mutex>> locks;ValidateTokenOwners(locks);
    const size_t batch=states_[0].rows,rows=batch*16;
    auto staged=std::make_unique<LookaheadWorkspace>(LookaheadLayout(batch,tokenCapacity_,size_t(window),size_t(levels),size_t(guesses)));
    auto& w=*staged;const std::vector<int64_t> cache{int64_t(batch),8,int64_t(tokenCapacity_),128};
    for(const auto& state:states_)if(state.shape[0]!=cache||state.shape[1]!=cache)throw std::invalid_argument("Lookahead canonical bank profile mismatch");
    w.input=gpu_->Allocate(rows,4096);w.positions=gpu_->Allocate(rows,3);w.active=gpu_->Allocate(rows,1);
    w.probabilities=gpu_->Allocate(rows,32*tokenCapacity_);w.predictions=gpu_->Allocate(rows,2);
    w.packedHidden=gpu_->Allocate(batch,16*4096);w.packedPredictions=gpu_->Allocate(batch,16*2);
    w.accepted=gpu_->Allocate(batch,16);w.finalHidden=gpu_->Allocate(batch,4096);gpu_->Zero(w.finalHidden);
    for(auto& hidden:w.hidden)hidden=gpu_->Allocate(rows,4096);
    w.hostHidden.resize(rows*4096);w.hostPositions.resize(rows*3);w.hostActive.resize(rows);w.hostAccepted.resize(rows);
    gpu_->PrepareLookaheadAttention(w.layout);
    for(size_t i=0;i<states_.size();++i)w.layers.push_back(CompileLookaheadLayer(*owners_[i].NativeData<MetalCompiledGraph>(),w,i,false));
    w.head=CompileLookaheadLayer(*head_,w,0,true);
    const auto metadata=AffineInt4Metadata::Resolve(head_->index_,head_->mlxCheckpoint_,"model.language_model.embed_tokens.weight");
    if(metadata.layout.rows!=32768||metadata.layout.columns!=4096)throw std::invalid_argument("Lookahead embedding must match target vocabulary/width");
    for(const auto* tensor:metadata.tensors)w.dependencies.push_back({tensor->filePath,std::filesystem::last_write_time(tensor->filePath),std::filesystem::file_size(tensor->filePath)});
    w.embedding.Open(metadata);
    for(const auto& dependency:w.dependencies)if(dependency.modified!=std::filesystem::last_write_time(dependency.path)||dependency.bytes!=std::filesystem::file_size(dependency.path))
        throw std::invalid_argument("Lookahead embedding changed during preparation");
    for(size_t row=0;row<batch;++row)w.owners[row]=std::make_unique<LookaheadRowOwner>(w.layout,32768);
    if(!rowFinalHidden_.Valid()){rowFinalHidden_=gpu_->Allocate(batch,4096);gpu_->Zero(rowFinalHidden_);rowFinalWidth_=4096;finalHiddenReady_.assign(batch,0);}
    else if(rowFinalWidth_!=4096)throw std::invalid_argument("Lookahead final hidden profile mismatch");
    if(!predictionInitialized_)gpu_->Zero(prediction_);
    auto memory=MetalContext::MemoryStats();if(memory.recommended<=256*1024*1024||memory.allocated>memory.recommended-256*1024*1024)
        throw std::runtime_error("Lookahead prepared profile exceeds native working-set guard");
    w.preparedBytes=memory.allocated;
    w.stats=nlohmann::json{{"prepared",true},{"batch",batch},{"local_rows",16},{"window",window},{"levels",levels},{"guesses",guesses},
        {"local_KV_bytes",w.layout.StageBytes(32,8,128)},{"prepared_metal_bytes",w.preparedBytes},{"mode","native_AR_DAG_lookahead_v1"}}.dump();
    lookahead_=std::move(staged);return true;
}
X::Value MetalDecoderSession::LookaheadGenerate(X::Value knownValue,X::Value slotsValue,X::Value remainingValue,X::Value stopsValue,X::Value epochsValue) {
    std::lock_guard<std::mutex> sessionLock(mutex_);
    const auto started=std::chrono::steady_clock::now();
    if(!gpu_||poisoned_||!lookahead_||!tokenCache_||!requestRows_||!managedRows_)throw std::logic_error("Prepare and claim native lookahead request rows first");
    gpu_->RequireDrainedProfileControl();auto& w=*lookahead_;const size_t batch=w.layout.Batch();
    const auto known=LookaheadInts(knownValue,Host(),batch),slots=LookaheadInts(slotsValue,Host(),batch),remaining=LookaheadInts(remainingValue,Host(),batch),epochs=LookaheadInts(epochsValue,Host(),batch);
    if(stopsValue.runtime()!=Host()->runtime||!stopsValue.IsList()||stopsValue.Size()>8)throw std::invalid_argument("Lookahead stop set exceeds profile");
    const auto stops64=LookaheadInts(stopsValue,Host(),stopsValue.Size());std::vector<int32_t> stops;
    for(auto id:stops64){if(id<0||id>=32768)throw std::invalid_argument("Lookahead stop outside vocabulary");stops.push_back(int32_t(id));}
    std::vector<std::unique_lock<std::mutex>> locks;ValidateTokenOwners(locks);
    for(const auto& layer:w.layers)if(!layer.source->ready_||layer.source->generation_!=layer.generation)throw std::logic_error("Lookahead source graph generation changed");
    if(w.head.source!=head_||w.head.generation!=headGeneration_)throw std::logic_error("Lookahead target head generation changed");
    for(const auto& dependency:w.dependencies)if(dependency.modified!=std::filesystem::last_write_time(dependency.path)||dependency.bytes!=std::filesystem::file_size(dependency.path))
        throw std::invalid_argument("Lookahead target embedding identity changed");
    size_t activeRows=0;
    for(size_t row=0;row<batch;++row) {
        if(!epochs[row]){if(known[row]||slots[row]||remaining[row])throw std::invalid_argument("Inactive lookahead controls must be canonical zero");continue;}
        if(epochs[row]<0||!requestRows_->Matches(row,uint64_t(epochs[row]))||known[row]<0||known[row]>=32768||slots[row]<0||
           uint64_t(slots[row])>=tokenCapacity_||remaining[row]<1||uint64_t(remaining[row])>tokenCapacity_||
           !rowTokensValid_[row]||rowTokens_[row].size()!=uint64_t(slots[row]))throw std::invalid_argument("Lookahead controls differ from native lease/token progress");
        for(auto token:rowTokens_[row])if(token<0||token>=32768)throw std::invalid_argument("Invalid native seed ledger token");
        rowTokens_[row].reserve(std::min(tokenCapacity_,rowTokens_[row].size()+w.layout.Levels()));
#ifdef GARNET_METAL_PREFIX_PREFILL_ONLY
        tokenPrecisionHistory_.at(row).Validate(rowTokens_[row].size());
#endif
        ++activeRows;
    }
    auto result=X::Value::List(Host());
    if(!activeRows){for(size_t row=0;row<batch;++row){auto item=X::Value::List(Host());item.Append(X::Value::List(Host()));item.Append(X::Value(int64_t(0)));item.Append(X::Value(false));result.Append(item);}
        gpuMs_=0;inferenceMs_=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
        w.stats=nlohmann::json{{"mode","native_AR_DAG_lookahead_v1"},{"active_rows",0},{"target_token_rows",0},{"consumed_inputs",0},{"no_op",true}}.dump();return result;}
    std::array<std::optional<LookaheadRowOwner::Plan>,4> plans;
    std::array<std::optional<LookaheadRowOwner::Transition>,4> transitions;
    std::vector<LookaheadLayout::View> views;views.reserve(activeRows);
    std::fill(w.hostHidden.begin(),w.hostHidden.end(),0);std::fill(w.hostPositions.begin(),w.hostPositions.end(),0);std::fill(w.hostActive.begin(),w.hostActive.end(),0);
    const auto embedding=w.embedding.View();size_t activeNodes=0;
    for(size_t row=0;row<batch;++row)if(epochs[row]) {
        auto& owner=*w.owners[row];
        if(!owner.Live()||owner.Epoch()!=uint64_t(epochs[row])||owner.Generation()!=w.layers[0].generation||owner.Slot()!=size_t(slots[row])||
           owner.KnownRoot()!=known[row]||owner.Remaining()!=size_t(remaining[row])) {
            owner.Invalidate();std::vector<int32_t> seed(rowTokens_[row].begin(),rowTokens_[row].end());
            owner.Bind(row,uint64_t(epochs[row]),w.layers[0].generation,size_t(slots[row]),size_t(remaining[row]),int32_t(known[row]),seed);
        }
        plans[row]=owner.BuildPlan();const auto& plan=*plans[row];views.push_back(plan.View());
        for(size_t node=0;node<16;++node)if(plan.View().Active()&(uint16_t(1)<<node)) {
            const size_t flat=row*16+node;embedding.DecodeEmbeddingRow(size_t(plan.Inputs()[node]),w.hostHidden.data()+flat*4096,4096);
            const float position=float(w.layout.Position(plan.View(),node));for(size_t axis=0;axis<3;++axis)w.hostPositions[flat*3+axis]=position;
            w.hostActive[flat]=1;++activeNodes;
        }
    }
    gpu_->Update(w.input,w.hostHidden);gpu_->Update(w.positions,w.hostPositions);gpu_->Update(w.active,w.hostActive);
    poisoned_=true;gpu_->Begin();
    try {
        auto hidden=w.input;
        for(size_t i=0;i<w.layers.size();++i){EncodeLookaheadLayer(w.layers[i],w,hidden,&states_[i].matrix,views);hidden=w.layers[i].matrices[w.layers[i].output];}
        EncodeLookaheadLayer(w.head,w,hidden,nullptr,views);gpu_->EncodeArgmax(w.head.matrices[w.head.output],w.predictions);
        gpu_->EncodeLayout(hidden,hidden,w.packedHidden,0);gpu_->SubmitAndWait();const double forwardGPU=gpu_->LastGPUMs();
        const size_t projectionDispatches=gpu_->LastLookaheadProjectionDispatches(),attentionDispatches=gpu_->LastLookaheadAttentionDispatches();
        if(projectionDispatches!=225||attentionDispatches!=32)throw std::logic_error("Lookahead executed a different target DAG");
        const auto prediction=gpu_->Download(w.predictions);std::vector<LookaheadLayout::CommitSelection> paths;paths.reserve(views.size());
        std::fill(w.hostAccepted.begin(),w.hostAccepted.end(),0);size_t consumed=0;
        for(size_t row=0;row<batch;++row)if(plans[row]) {
            std::array<int32_t,16> target;target.fill(-1);const auto& plan=*plans[row];
            for(size_t node=0;node<16;++node)if(plan.View().Active()&(uint16_t(1)<<node)) {
                const size_t at=(row*16+node)*2;const float id=prediction[at];
                if(prediction[at+1]!=0||!std::isfinite(id)||id<0||id>=32768||id!=std::floor(id))throw std::runtime_error("Invalid native lookahead target prediction");
                target[node]=int32_t(id);
            }
            transitions[row]=w.owners[row]->Stage(plan,target,stops);const auto& transition=*transitions[row];paths.push_back(transition.SelectedPath());
            for(size_t at=0;at<transition.Count();++at)w.hostAccepted[row*16+transition.SelectedPath().local[at]]=1;
            consumed+=transition.Count();
        }
        gpu_->Update(w.accepted,w.hostAccepted);gpu_->Begin();
        for(size_t i=0;i<w.layers.size();++i) {
            const auto& layer=w.layers[i];gpu_->EncodeLookaheadKVCommit(layer.matrices[layer.key],states_[i].matrix[0],w.layout,views,paths);
            gpu_->EncodeLookaheadKVCommit(layer.matrices[layer.value],states_[i].matrix[1],w.layout,views,paths);
        }
        gpu_->EncodeLastActiveHidden(w.packedHidden,w.accepted,rowFinalHidden_,16);gpu_->EncodeLayout(rowFinalHidden_,rowFinalHidden_,w.finalHidden,0);
        gpu_->EncodeLayout(w.predictions,w.predictions,w.packedPredictions,0);gpu_->EncodeLastActiveHidden(w.packedPredictions,w.accepted,prediction_,16);
        gpu_->SubmitAndWait();const double commitGPU=gpu_->LastGPUMs();
        for(size_t row=0;row<batch;++row)if(transitions[row]) {
            const auto& plan=*plans[row];const auto& transition=*transitions[row];
#ifdef GARNET_METAL_PREFIX_PREFILL_ONLY
            tokenPrecisionHistory_.at(row).Append(rowTokens_[row].size(),transition.Count(),false);
#endif
            for(size_t at=0;at<transition.Count();++at)rowTokens_[row].push_back(plan.Inputs()[transition.SelectedPath().local[at]]);
            w.owners[row]->Publish(transition);finalHiddenReady_[row]=1;
#ifdef GARNET_METAL_TARGET_JACOBI
            if(targetJacobiEnabled_)targetJacobi_.ResetRow(row);
#endif
        }
        predictionInitialized_=true;hidden_=w.finalHidden;hiddenShape_={int64_t(batch),4096};
        gpuMs_=forwardGPU+commitGPU;inferenceMs_=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
        w.stats=nlohmann::json{{"mode","native_AR_DAG_lookahead_v1"},{"active_rows",activeRows},{"target_token_rows",activeNodes},
            {"consumed_inputs",consumed},{"accepted_proposals",consumed-activeRows},{"forward_gpu_ms",forwardGPU},{"commit_gpu_ms",commitGPU},
            {"projection_dispatches",projectionDispatches},{"attention_dispatches",attentionDispatches},
            {"commit_dispatches",gpu_->LastLookaheadKVCommitDispatches()},{"prepared_metal_bytes",w.preparedBytes}}.dump();
        for(size_t row=0;row<batch;++row) {
            auto item=X::Value::List(Host()),emitted=X::Value::List(Host());size_t count=0;bool eos=false;
            if(transitions[row]){const auto& t=*transitions[row];count=t.Count();eos=t.EndedByEOS();for(size_t i=0;i<count;++i)emitted.Append(X::Value(int64_t(t.Emitted()[i])));}
            item.Append(emitted);item.Append(X::Value(int64_t(count)));item.Append(X::Value(eos));result.Append(item);
        }
        poisoned_=false;return result;
    }catch(...){gpu_->Abort();for(size_t row=0;row<batch;++row)ResetLookaheadRow(row);throw;}
}
}
#endif
