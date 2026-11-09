// SPDX-License-Identifier: Apache-2.0
namespace {
std::array<size_t,5> LookaheadProfileKey(const LookaheadLayout& p) {
    return {p.Batch(),p.Capacity(),p.Window(),p.Levels(),p.Guesses()};
}
}
void MetalContext::PrepareLookaheadAttention(const LookaheadLayout& profile) {
    @autoreleasepool {
    if(impl_->command||impl_->submitted.count)
        throw std::logic_error("Prepare lookahead attention between drained command batches");
    const auto key=LookaheadProfileKey(profile);
    std::unique_lock<std::shared_mutex> lock(impl_->kernels->lookaheadMutex);
    if(impl_->kernels->lookaheadProfiles.count(key))return;
    if(impl_->kernels->lookaheadProfiles.size()>=64)
        throw std::invalid_argument("Lookahead device profile cache is full");
    NSError* error=nil;auto library=impl_->kernels->lookaheadLibrary;
    if(!library) {
        // Match the existing state-kernel library's compiler math mode. Its
        // default compile options are part of the native reference contract.
        auto options=[MTLCompileOptions new];options.languageVersion=MTLLanguageVersion3_1;
        library=[impl_->device newLibraryWithSource:[NSString stringWithUTF8String:kLookaheadAttentionShader] options:options error:&error];
        if(!library)throw std::runtime_error("Cannot compile native lookahead attention");
    }
    auto constants=[MTLFunctionConstantValues new];
    for(size_t i=0;i<key.size();++i){uint32_t value=uint32_t(key[i]);[constants setConstantValue:&value type:MTLDataTypeUInt atIndex:i];}
    auto prepare=[&](NSString* name) {
        auto function=[library newFunctionWithName:name constantValues:constants error:&error];
        if(!function)throw std::runtime_error("Cannot specialize native lookahead attention");
        auto pipeline=[impl_->device newComputePipelineStateWithFunction:function error:&error];
        if(!pipeline||pipeline.threadExecutionWidth!=32||pipeline.maxTotalThreadsPerThreadgroup<128)
            throw std::runtime_error("Lookahead attention requires128-thread/32-lane pipelines");
        return pipeline;
    };
    MetalPipelines::LookaheadProfile staged;
    staged.scores=prepare(@"lookahead_attention_scores");staged.context=prepare(@"lookahead_attention_context");
    const size_t scratch=(profile.Capacity()*sizeof(float)+15)&~size_t(15);
    if(staged.scores.staticThreadgroupMemoryLength>impl_->device.maxThreadgroupMemoryLength||
       scratch>impl_->device.maxThreadgroupMemoryLength-staged.scores.staticThreadgroupMemoryLength)
        throw std::runtime_error("Lookahead softmax scratch exceeds device threadgroup storage");
    // Publish only a fully prepared pair. No model/KV buffers are allocated.
    impl_->kernels->lookaheadProfiles.emplace(key,std::move(staged));impl_->kernels->lookaheadLibrary=library;
    }
}
size_t MetalContext::LookaheadAttentionPipelineCount() const {
    std::shared_lock<std::shared_mutex> lock(impl_->kernels->lookaheadMutex);
    return 2*impl_->kernels->lookaheadProfiles.size();
}
size_t MetalContext::LastLookaheadAttentionDispatches() const {return impl_->lookaheadAttentionDispatches;}
void MetalContext::EncodeLookaheadAttention(const std::vector<MetalMatrix>& inputs,
    const MetalMatrix& probabilities,const MetalMatrix& output,const LookaheadLayout& profile,
    const std::vector<LookaheadLayout::View>& views) {
    @autoreleasepool {
    if(!impl_->command)throw std::logic_error("Begin native lookahead attention first");
    if(inputs.size()!=5||!probabilities.impl_||!output.impl_||views.size()>profile.Batch())
        throw std::invalid_argument("Invalid native lookahead attention operands");
    MetalPipelines::LookaheadProfile prepared;
    {std::shared_lock<std::shared_mutex> lock(impl_->kernels->lookaheadMutex);
     const auto found=impl_->kernels->lookaheadProfiles.find(LookaheadProfileKey(profile));
     if(found==impl_->kernels->lookaheadProfiles.end())throw std::logic_error("Prepare this exact lookahead profile before inference");
     prepared=found->second;}
    for(const auto& input:inputs)if(!input.impl_)throw std::invalid_argument("Missing lookahead attention input");
    const auto q=inputs[0].impl_->matrix,k=inputs[1].impl_->matrix,v=inputs[2].impl_->matrix,
        kc=inputs[3].impl_->matrix,vc=inputs[4].impl_->matrix,p=probabilities.impl_->matrix,o=output.impl_->matrix;
    const size_t batch=profile.Batch(),capacity=profile.Capacity(),rows=batch*LookaheadLayout::LocalRows;
    if(q.rows!=rows||q.columns!=4096||k.rows!=rows||k.columns!=1024||v.rows!=rows||v.columns!=1024||
       kc.rows!=batch||kc.columns!=8*capacity*128||vc.rows!=batch||vc.columns!=kc.columns||
       p.rows!=rows||p.columns!=32*capacity||o.rows!=rows||o.columns!=4096)
        throw std::invalid_argument("Lookahead attention differs from compiled matrix layout");
    auto matrixValid=[&](MPSMatrix* matrix) {
        if(matrix.device!=impl_->device||matrix.dataType!=MPSDataTypeFloat32||matrix.rowBytes%4||matrix.offset%4||
           matrix.rowBytes<matrix.columns*sizeof(float)||matrix.offset>matrix.data.length||
           matrix.rows-1>(SIZE_MAX-matrix.columns*sizeof(float))/matrix.rowBytes||
           (matrix.rows-1)*matrix.rowBytes+matrix.columns*sizeof(float)>matrix.data.length-matrix.offset)
            throw std::invalid_argument("Invalid lookahead attention device/type/storage span");
    };
    matrixValid(p);matrixValid(o);
    if(p.data==o.data||p.offset||o.offset)throw std::invalid_argument("Lookahead outputs alias or have unsupported offsets");
    for(const auto* target:{&probabilities,&output}) {
        if(target->impl_->pending&&!impl_->pendingOutputs.count(target->impl_->PendingKey()))
            throw std::logic_error("Lookahead output belongs to another pending batch");
        target->impl_->ValidateSpanWrite(impl_->pendingOutputs.count(target->impl_->PendingKey()));
    }
    for(const auto& input:inputs) {
        matrixValid(input.impl_->matrix);
        if(input.impl_->matrix.data==p.data||input.impl_->matrix.data==o.data)
            throw std::invalid_argument("Lookahead output aliases a read-only operand");
        if(!input.impl_->ready&&!(input.impl_->pending&&impl_->pendingOutputs.count(input.impl_->PendingKey())))
            throw std::logic_error("Uninitialized lookahead attention input");
    }
    std::array<uint32_t,4> prefix{};std::array<uint16_t,4> active{};std::array<bool,4> assigned{};
    for(const auto& view:views) {
        // Position checks the profile identity of this immutable native view.
        profile.Position(view,0);const size_t row=view.Row();
        if(row>=batch||assigned[row])throw std::invalid_argument("Duplicate/out-of-profile lookahead owner row");
        assigned[row]=true;prefix[row]=uint32_t(view.Slot());active[row]=view.Active();
        for(size_t query=0;query<LookaheadLayout::LocalRows;++query)if(active[row]&(uint16_t(1)<<query)) {
            size_t count=0;for(size_t key=0;key<LookaheadLayout::LocalRows;++key)count+=size_t(profile.Sees(view,query,key));
            if(!count||view.Slot()+count!=profile.Position(view,query)+1||view.Slot()+count>capacity)
                throw std::invalid_argument("Lookahead query lacks a complete causal ancestor path");
        }
    }
    struct Params {uint64_t qStride,kStride,vStride,kcStride,vcStride,pStride,oStride,dim;};
    Params params{q.rowBytes/4,k.rowBytes/4,v.rowBytes/4,kc.rowBytes/4,vc.rowBytes/4,p.rowBytes/4,o.rowBytes/4,128};
    auto publish=[&](const MetalMatrix& matrix) {
        matrix.impl_->ready=false;matrix.impl_->pending=true;
        if(impl_->pendingOutputs.insert(matrix.impl_->PendingKey()).second)impl_->outputs.push_back(matrix.impl_);
    };
    auto bind=[&](id<MTLComputeCommandEncoder> encoder,const std::array<MPSMatrix*,4>& matrices) {
        for(size_t i=0;i<matrices.size();++i)[encoder setBuffer:matrices[i].data offset:matrices[i].offset atIndex:i];
        [encoder setBytes:&params length:sizeof(params) atIndex:4];
        [encoder setBytes:prefix.data() length:sizeof(prefix) atIndex:5];
        [encoder setBytes:active.data() length:sizeof(active) atIndex:6];
    };
    auto encoder=impl_->ComputeEncoder(__func__);
    if(!encoder)throw std::runtime_error("Cannot encode native lookahead scores");
    [encoder setComputePipelineState:prepared.scores];bind(encoder,{q,k,kc,p});
    [encoder setThreadgroupMemoryLength:(capacity*sizeof(float)+15)&~size_t(15) atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake(rows*32,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    [encoder endEncoding];publish(probabilities);
    encoder=impl_->ComputeEncoder(__func__);
    if(!encoder)throw std::runtime_error("Cannot encode native lookahead values");
    [encoder setComputePipelineState:prepared.context];bind(encoder,{p,v,vc,o});
    [encoder dispatchThreads:MTLSizeMake(rows*4096/4,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    [encoder endEncoding];publish(output);++impl_->lookaheadAttentionDispatches;
    }
}
