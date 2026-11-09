// SPDX-License-Identifier: Apache-2.0
namespace {
std::array<size_t,5> LookaheadProjectionKey(MPSMatrix* a,MPSMatrix* o,const LookaheadLayout& layout) {
    return {layout.Batch(),a.columns,o.columns,a.rowBytes/4,o.rowBytes/4};
}
void LookaheadProjectionMatrix(MPSMatrix* matrix,id<MTLDevice> device) {
    if(matrix.device!=device||matrix.dataType!=MPSDataTypeFloat32||!matrix.rows||!matrix.columns||
       matrix.rowBytes%4||matrix.offset%4||matrix.rowBytes<matrix.columns*sizeof(float)||matrix.offset>matrix.data.length||
       matrix.rows-1>(SIZE_MAX-matrix.columns*sizeof(float))/matrix.rowBytes||
       (matrix.rows-1)*matrix.rowBytes+matrix.columns*sizeof(float)>matrix.data.length-matrix.offset)
        throw std::invalid_argument("Invalid lookahead projection matrix device/type/span");
}
bool LookaheadProjectionShape(size_t inner,size_t outputs) {
    return (inner==4096&&(outputs==1024||outputs==4096||outputs==14336||outputs==32768))||
        (inner==14336&&outputs==4096);
}
}
void MetalContext::PrepareLookaheadProjection(const MetalMatrix& input,const MetalAffineInt4Weight& weight,
    const MetalMatrix& output,const LookaheadLayout& layout) {
    @autoreleasepool {
    if(impl_->command||impl_->submitted.count)throw std::logic_error("Prepare lookahead projections between drained batches");
    if(!input.impl_||!weight.impl_||!output.impl_)throw std::invalid_argument("Missing lookahead projection operands");
    auto a=input.impl_->matrix,o=output.impl_->matrix;const auto& w=weight.impl_;
    LookaheadProjectionMatrix(a,impl_->device);LookaheadProjectionMatrix(o,impl_->device);
    const size_t rows=layout.Batch()*LookaheadLayout::LocalRows;
    if(a.rows!=rows||o.rows!=rows||a.columns!=w->columns||o.columns!=w->rows||a.data==o.data||
       w->packed.device!=impl_->device||w->scales.device!=impl_->device||w->biases.device!=impl_->device||
       !LookaheadProjectionShape(a.columns,o.columns)||
       !AffineProfileIndex32Fits(rows,a.columns,o.columns,a.rowBytes/4,o.rowBytes/4))
        throw std::invalid_argument("Lookahead projection differs from bounded dense model profile");
    const auto key=LookaheadProjectionKey(a,o,layout);
    std::unique_lock<std::shared_mutex> lock(impl_->kernels->lookaheadProjectionMutex);
    auto& cache=impl_->kernels->lookaheadProjections;if(cache.count(key))return;
    if(cache.size()>=64)throw std::runtime_error("Lookahead projection profile cache bound exceeded");
    NSError* error=nil;auto library=impl_->kernels->lookaheadProjectionLibrary;
    if(!library) {
        NSString* source=@"#define GARNET_AFFINE_PROFILE 1\n#define GARNET_AFFINE_INDEX32 1\n#define GARNET_AFFINE_FP16_COEFFICIENTS 1\n#define GARNET_LOOKAHEAD_PROJECTION 1\n";
        source=[source stringByAppendingString:[NSString stringWithUTF8String:kAffineInt4ProfilePrelude]];
        source=[source stringByAppendingString:[NSString stringWithUTF8String:kAffineInt4WideDecodeShader]];
        auto options=[MTLCompileOptions new];options.languageVersion=MTLLanguageVersion3_1;options.fastMathEnabled=NO;
        library=[impl_->device newLibraryWithSource:source options:options error:&error];
        if(!library)throw std::runtime_error("Cannot compile lookahead projection: "+std::string(error.localizedDescription.UTF8String));
    }
    auto constants=[MTLFunctionConstantValues new];
    const uint32_t dimensions[5]{uint32_t(a.columns),uint32_t(o.columns),uint32_t(a.rowBytes/4),uint32_t(o.rowBytes/4),uint32_t(rows)};
    for(size_t i=0;i<5;++i)[constants setConstantValue:&dimensions[i] type:MTLDataTypeUInt atIndex:10+i];
    auto function=[library newFunctionWithName:@"affine_q4_lookahead" constantValues:constants error:&error];
    if(!function)throw std::runtime_error("Cannot specialize lookahead projection: "+std::string(error.localizedDescription.UTF8String));
    auto pipeline=[impl_->device newComputePipelineStateWithFunction:function error:&error];
    if(!pipeline||pipeline.threadExecutionWidth!=32||pipeline.maxTotalThreadsPerThreadgroup<128||
       pipeline.staticThreadgroupMemoryLength>impl_->device.maxThreadgroupMemoryLength)
        throw std::runtime_error("Lookahead projection pipeline exceeds execution bounds");
    cache.emplace(key,pipeline);impl_->kernels->lookaheadProjectionLibrary=library;
    }
}
size_t MetalContext::LookaheadProjectionPipelineCount() const {
    std::shared_lock<std::shared_mutex> lock(impl_->kernels->lookaheadProjectionMutex);
    return impl_->kernels->lookaheadProjections.size();
}
size_t MetalContext::LastLookaheadProjectionDispatches() const {return impl_->lookaheadProjectionDispatches;}
void MetalContext::EncodeLookaheadProjection(const MetalMatrix& input,const MetalAffineInt4Weight& weight,
    const MetalMatrix& output,const LookaheadLayout& layout,const std::vector<LookaheadLayout::View>& views) {
    @autoreleasepool {
    if(!impl_->command)throw std::logic_error("Begin lookahead projection first");
    if(!input.impl_||!weight.impl_||!output.impl_||views.size()>layout.Batch())
        throw std::invalid_argument("Missing or excessive lookahead projection operands");
    auto a=input.impl_->matrix,o=output.impl_->matrix;const auto& w=weight.impl_;
    LookaheadProjectionMatrix(a,impl_->device);LookaheadProjectionMatrix(o,impl_->device);
    const size_t rows=layout.Batch()*LookaheadLayout::LocalRows;
    if(a.rows!=rows||o.rows!=rows||a.columns!=w->columns||o.columns!=w->rows||a.data==o.data||
       o.data==w->packed||o.data==w->scales||o.data==w->biases||
       w->packed.device!=impl_->device||w->scales.device!=impl_->device||w->biases.device!=impl_->device||
       !LookaheadProjectionShape(a.columns,o.columns)||
       !AffineProfileIndex32Fits(rows,a.columns,o.columns,a.rowBytes/4,o.rowBytes/4)||
       w->packed.length<o.columns*(a.columns/8)*sizeof(uint32_t)||
       w->scales.length<o.columns*(a.columns/64)*sizeof(uint16_t)||w->biases.length<w->scales.length)
        throw std::invalid_argument("Lookahead projection shape/device/weight span mismatch");
    if(!input.impl_->ready&&!(input.impl_->pending&&impl_->pendingOutputs.count(input.impl_->PendingKey())))
        throw std::logic_error("Uninitialized or foreign pending lookahead projection input");
    if(output.impl_->pending&&!impl_->pendingOutputs.count(output.impl_->PendingKey()))
        throw std::logic_error("Lookahead projection output has a foreign pending writer");
    output.impl_->ValidateSpanWrite(impl_->pendingOutputs.count(output.impl_->PendingKey()));
    id<MTLComputePipelineState> pipeline=nil;
    {std::shared_lock<std::shared_mutex> lock(impl_->kernels->lookaheadProjectionMutex);
     auto found=impl_->kernels->lookaheadProjections.find(LookaheadProjectionKey(a,o,layout));
     if(found==impl_->kernels->lookaheadProjections.end())throw std::logic_error("Prepare exact lookahead projection storage profile first");
     pipeline=found->second;}
    std::array<uint32_t,64> indices{};std::array<bool,4> assigned{};size_t count=0;
    for(const auto& view:views) {
        // This validates immutable profile identity, not actual live leases.
        // Session integration must validate graph generations and row epochs.
        layout.Position(view,0);const size_t row=view.Row();
        if(row>=layout.Batch()||assigned[row])throw std::invalid_argument("Duplicate/out-of-profile lookahead projection row");
        assigned[row]=true;
        for(size_t node=0;node<LookaheadLayout::LocalRows;++node)if(view.Active()&(uint16_t(1)<<node)) {
            size_t ancestors=0;for(size_t key=0;key<LookaheadLayout::LocalRows;++key)ancestors+=size_t(layout.Sees(view,node,key));
            if(!ancestors||view.Slot()+ancestors!=layout.Position(view,node)+1)
                throw std::invalid_argument("Lookahead projection node lacks complete ancestors");
            indices[count++]=uint32_t(row*LookaheadLayout::LocalRows+node);
        }
    }
    // All admission checks precede the first output mutation. The storage
    // layout/weights/PSO are already prepared; no fallback compiler runs here.
    const size_t bytes=(o.rows-1)*o.rowBytes+o.columns*sizeof(float);
    auto publish=[&]{output.impl_->ready=false;output.impl_->pending=true;
        if(impl_->pendingOutputs.insert(output.impl_->PendingKey()).second)impl_->outputs.push_back(output.impl_);};
    if(count<rows) {
        auto blit=[impl_->command blitCommandEncoder];
        if(!blit)throw std::runtime_error("Cannot clear inactive lookahead projection rows");
        [blit fillBuffer:o.data range:NSMakeRange(o.offset,bytes) value:0];[blit endEncoding];publish();
    }
    if(count) {
        struct Params {uint64_t rows,inner,columns,aStride,oStride;};
        Params params{count,a.columns,o.columns,a.rowBytes/4,o.rowBytes/4};const uint32_t selected=uint32_t(count);
        auto encoder=impl_->ComputeEncoder(__func__);if(!encoder)throw std::runtime_error("Cannot encode lookahead projection");
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:a.data offset:a.offset atIndex:0];[encoder setBuffer:w->packed offset:0 atIndex:1];
        [encoder setBuffer:w->scales offset:0 atIndex:2];[encoder setBuffer:w->biases offset:0 atIndex:3];
        [encoder setBuffer:o.data offset:o.offset atIndex:4];[encoder setBytes:&params length:sizeof(params) atIndex:5];
        [encoder setBytes:indices.data() length:count*sizeof(uint32_t) atIndex:6];[encoder setBytes:&selected length:sizeof(selected) atIndex:7];
        [encoder dispatchThreadgroups:MTLSizeMake(((o.columns+15)/16)*((count+3)/4),1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
        [encoder endEncoding];publish();++impl_->lookaheadProjectionDispatches;
    }
    }
}
