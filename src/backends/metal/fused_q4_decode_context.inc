// SPDX-License-Identifier: Apache-2.0
// Included in metal_context.mm after the native bounded affine definitions.
namespace {
bool FusedDecodeShape(MPSMatrix* a,MPSMatrix* o) {
    if(a.columns!=4096||o.columns!=14336||a.rows!=o.rows)return false;
    if(a.rows==4)return true;
#ifdef GARNET_METAL_FUSED_Q4_WIDE_SELECTION
    return a.rows==16&&a.rowBytes==4096*4&&o.rowBytes==14336*4;
#else
    return false;
#endif
}
size_t FusedDecodeExtent(MPSMatrix* matrix) {
    if(!matrix||!matrix.rows||!matrix.columns||matrix.dataType!=MPSDataTypeFloat32||
       matrix.rowBytes<matrix.columns*4||matrix.rowBytes%4||matrix.offset%4||
       matrix.rows-1>(SIZE_MAX-matrix.columns*4)/matrix.rowBytes)
        throw std::invalid_argument("Invalid fused decode matrix extent");
    const size_t bytes=(matrix.rows-1)*matrix.rowBytes+matrix.columns*4;
    if(matrix.offset>matrix.data.length||bytes>matrix.data.length-matrix.offset)
        throw std::invalid_argument("Fused decode matrix exceeds its buffer");
    return bytes;
}
}
bool MetalContext::PrepareFusedMistralDecode(const MetalMatrix& input,const MetalAffineInt4Weight& gate,
 const MetalAffineInt4Weight& up,const MetalMatrix& output) {
 @autoreleasepool {
    if(impl_->command||impl_->submitted.count)throw std::logic_error("Prepare fused decode between drained commands");
    if(!input.impl_||!output.impl_||!gate.impl_||!up.impl_)return false;
    auto a=input.impl_->matrix,o=output.impl_->matrix;auto g=gate.impl_,u=up.impl_;
    if(!FusedDecodeShape(a,o)||
       g->columns!=4096||u->columns!=4096||g->rows!=14336||u->rows!=14336)return false;
    if(a.device!=impl_->device||o.device!=impl_->device||g->packed.device!=impl_->device||u->packed.device!=impl_->device||
       g->scales.device!=impl_->device||g->biases.device!=impl_->device||u->scales.device!=impl_->device||u->biases.device!=impl_->device)
        throw std::invalid_argument("Fused decode operand uses another device");
    FusedDecodeExtent(a);FusedDecodeExtent(o);
    if(!AffineProfileIndex32Fits(a.rows,4096,14336,a.rowBytes/4,o.rowBytes/4))return false;
    if(![impl_->device supportsFamily:MTLGPUFamilyApple10])return false;
    auto library=impl_->fusedDecodeLibrary;
    if(!library) {
        NSString* text=[@"#define GARNET_AFFINE_PROFILE 1\n#define GARNET_AFFINE_INDEX32 1\n#define GARNET_AFFINE_FP16_COEFFICIENTS 1\n#define FUSED_EXP exp\n" stringByAppendingString:[NSString stringWithUTF8String:kAffineInt4ProfilePrelude]];
        text=[text stringByAppendingString:[NSString stringWithUTF8String:kFusedQ4DecodeShader]];
        NSError* error=nil;library=[impl_->device newLibraryWithSource:text options:nil error:&error];
        if(!library)throw std::runtime_error("Cannot prepare default-math FP32 fused decode: "+std::string(error.localizedDescription.UTF8String));
    }
    std::vector<std::pair<std::array<size_t,2>,id<MTLComputePipelineState>>> staged;
    for(size_t stride:{size_t(a.rowBytes/4),size_t(4096)}) {
        const std::array<size_t,2> key{stride,size_t(o.rowBytes/4)};
        if(impl_->fusedDecodePlans.count(key)||std::any_of(staged.begin(),staged.end(),[&](auto& v){return v.first==key;}))continue;
        if(impl_->fusedDecodePlans.size()+staged.size()>=64)throw std::runtime_error("Fused decode profile count exceeds64");
        auto constants=[MTLFunctionConstantValues new];uint32_t count=4,columns=4;
        uint32_t dims[]{4096,14336,uint32_t(stride),uint32_t(o.rowBytes/4)};bool shadow=false;
        [constants setConstantValue:&count type:MTLDataTypeUInt atIndex:0];
        for(NSUInteger i=0;i<4;++i)[constants setConstantValue:dims+i type:MTLDataTypeUInt atIndex:10+i];
        [constants setConstantValue:&shadow type:MTLDataTypeBool atIndex:20];[constants setConstantValue:&columns type:MTLDataTypeUInt atIndex:21];
        NSError* error=nil;auto function=[library newFunctionWithName:@"fused_q4_decode" constantValues:constants error:&error];
        if(!function)throw std::runtime_error("Cannot specialize fused decode");
        auto pipeline=[impl_->device newComputePipelineStateWithFunction:function error:&error];
        if(!pipeline||pipeline.threadExecutionWidth!=32||pipeline.maxTotalThreadsPerThreadgroup<128||pipeline.staticThreadgroupMemoryLength)
            throw std::runtime_error("Fused decode pipeline exceeds qualified resources");
        staged.emplace_back(key,pipeline);
    }
    auto* reserved=&impl_->fusedDecodeInput;
#ifdef GARNET_METAL_FUSED_Q4_WIDE_SELECTION
    if(a.rows==16)reserved=&impl_->fusedWideDecodeInput;
#endif
    auto scratch=*reserved;
    if(!scratch.Valid()) {
        const auto memory=MemoryStats();const size_t required=a.rows*4096*4,margin=256*1024*1024;
        if(memory.recommended<=required+margin||memory.allocated>memory.recommended-required-margin)
            throw std::runtime_error("Fused decode input scratch exceeds working-set budget");
        scratch=Allocate(a.rows,4096);Zero(scratch);
    }
    for(const auto& pair:staged)impl_->fusedDecodePlans.emplace(pair);
    impl_->fusedDecodeLibrary=library;*reserved=std::move(scratch);return true;
 }
}
size_t MetalContext::FusedMistralDecodeProfileCount() const{return impl_->fusedDecodePlans.size();}
size_t MetalContext::LastFusedMistralDecodeBlocks() const{return impl_->fusedDecodeBlocks;}
size_t MetalContext::LastFusedMistralDecodeInputCopies() const{return impl_->fusedDecodeInputCopies;}
MetalMatrix MetalContext::FusedMistralDecodeScratch() const{return impl_->fusedDecodeInput;}
#ifdef GARNET_METAL_FUSED_Q4_WIDE_SELECTION
void MetalContext::SetFusedMistralWideDecodeEnabled(bool enabled) {
    if(impl_->command||impl_->submitted.count)throw std::logic_error("Select wide fusion between drained commands");
    if(enabled&&![impl_->device supportsFamily:MTLGPUFamilyApple10])throw std::logic_error("Wide fusion requires Apple10");
    impl_->fusedWideDecodeEnabled=enabled;
}
bool MetalContext::FusedMistralWideDecodeEnabled() const{return impl_->fusedWideDecodeEnabled;}
size_t MetalContext::LastFusedMistralWideDecodeBlocks() const{return impl_->fusedWideDecodeBlocks;}
MetalMatrix MetalContext::FusedMistralWideDecodeScratch() const{return impl_->fusedWideDecodeInput;}
#endif
bool MetalContext::TryEncodeFusedMistralDecode(const MetalMatrix& input,const MetalAffineInt4Weight& gate,
 const MetalAffineInt4Weight& up,const MetalMatrix& output,const std::vector<size_t>* selectedRows) {
 @autoreleasepool {
    if(!impl_->command||!input.impl_||!output.impl_||!gate.impl_||!up.impl_)
        throw std::invalid_argument("Fused decode requires an open command and complete operands");
    auto a=input.impl_->matrix,o=output.impl_->matrix;auto g=gate.impl_,u=up.impl_;
    if(!FusedDecodeShape(a,o)||
       g->columns!=4096||u->columns!=4096||g->rows!=14336||u->rows!=14336)return false;
    std::array<uint32_t,4> indices{0,1,2,3};
#ifdef GARNET_METAL_FUSED_Q4_WIDE_SELECTION
    if(a.rows==16&&!impl_->fusedWideDecodeEnabled)return false;
#endif
    if(selectedRows) {
        if(selectedRows->size()>a.rows)throw std::invalid_argument("Fused decode selection exceeds capacity");
        std::array<bool,16> seen{};
        for(size_t i=0;i<selectedRows->size();++i){auto row=(*selectedRows)[i];if(row>=a.rows||seen[row])throw std::invalid_argument("Fused decode selection is not unique/in bounds");seen[row]=true;if(i<4)indices[i]=uint32_t(row);}
        if(selectedRows->size()!=4)return false;
    }else if(a.rows!=4)return false;
#ifdef GARNET_METAL_SILU_PROFILE32
    if(impl_->siluProfile32Enabled)return false;
#endif
    if(a.device!=impl_->device||o.device!=impl_->device||g->packed.device!=impl_->device||u->packed.device!=impl_->device||
       g->scales.device!=impl_->device||g->biases.device!=impl_->device||u->scales.device!=impl_->device||u->biases.device!=impl_->device)
        throw std::invalid_argument("Fused decode device mismatch");
    const auto aBytes=FusedDecodeExtent(a),oBytes=FusedDecodeExtent(o);
    if(!AffineProfileIndex32Fits(a.rows,4096,14336,a.rowBytes/4,o.rowBytes/4))throw std::invalid_argument("Fused decode index range changed");
    auto* reserved=&impl_->fusedDecodeInput;
#ifdef GARNET_METAL_FUSED_Q4_WIDE_SELECTION
    if(a.rows==16)reserved=&impl_->fusedWideDecodeInput;
#endif
    if(!reserved->Valid())return false;
    const bool copy=a.data==o.data&&a.offset<o.offset+oBytes&&o.offset<a.offset+aBytes;
    const std::array<size_t,2> key{copy?4096:size_t(a.rowBytes/4),size_t(o.rowBytes/4)};
    auto plan=impl_->fusedDecodePlans.find(key);if(plan==impl_->fusedDecodePlans.end())return false;
    if(!input.impl_->ready&&!(input.impl_->pending&&impl_->pendingOutputs.count(input.impl_->PendingKey())))throw std::logic_error("Fused decode input not ready");
    if(output.impl_->pending&&!impl_->pendingOutputs.count(output.impl_->PendingKey()))throw std::logic_error("Fused decode output belongs to another command");
    output.impl_->ValidateSpanWrite(impl_->pendingOutputs.count(output.impl_->PendingKey()));
    auto scratch=(*reserved).impl_->matrix;
    for(auto m:{a,o}) {
        if(m.data==scratch.data)throw std::invalid_argument("Fused decode operand aliases private input scratch");
#ifdef GARNET_METAL_FUSED_Q4_WIDE_SELECTION
        for(const auto* privateInput:{&impl_->fusedDecodeInput,&impl_->fusedWideDecodeInput})
            if(privateInput->Valid()&&m.data==privateInput->impl_->matrix.data)throw std::invalid_argument("Fused decode operand aliases another private input capacity");
#endif
        for(const auto& weight:{g,u})if(m.data==weight->packed||m.data==weight->scales||m.data==weight->biases)
            throw std::invalid_argument("Fused decode operand aliases checkpoint storage");
    }
    if(copy) {
        if((*reserved).impl_->pending&&!impl_->pendingOutputs.count((*reserved).impl_->PendingKey()))throw std::logic_error("Fused decode private scratch belongs to another command");
        (*reserved).impl_->ValidateSpanWrite(impl_->pendingOutputs.count((*reserved).impl_->PendingKey()));
    }
    // Validate every owner, span and prepared profile before publishing writes.
    output.impl_->ready=false;output.impl_->pending=true;
    if(impl_->pendingOutputs.insert(output.impl_->PendingKey()).second)impl_->outputs.push_back(output.impl_);
    if(copy){EncodeSliceGroups(input,(*reserved),1,4096,0,4096);a=scratch;}
    struct Params{uint64_t rows,inner,columns,aStride,oStride;};Params params{a.rows,4096,14336,a.rowBytes/4,o.rowBytes/4};
    auto encoder=impl_->ComputeEncoder(__func__);if(!encoder)throw std::runtime_error("Cannot encode fused FP32 decode");
    [encoder setComputePipelineState:plan->second];[encoder setBuffer:a.data offset:a.offset atIndex:0];
    [encoder setBuffer:g->packed offset:0 atIndex:1];[encoder setBuffer:g->scales offset:0 atIndex:2];[encoder setBuffer:g->biases offset:0 atIndex:3];
    [encoder setBuffer:u->packed offset:0 atIndex:4];[encoder setBuffer:u->scales offset:0 atIndex:5];[encoder setBuffer:u->biases offset:0 atIndex:6];
    [encoder setBuffer:o.data offset:o.offset atIndex:7];[encoder setBytes:&params length:sizeof(params) atIndex:8];
    [encoder setBytes:indices.data() length:sizeof(indices) atIndex:9];
    [encoder setBuffer:o.data offset:o.offset atIndex:10];[encoder setBuffer:o.data offset:o.offset atIndex:11];
    [encoder dispatchThreadgroups:MTLSizeMake(896,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[encoder endEncoding];
    ++impl_->fusedDecodeBlocks;impl_->fusedDecodeInputCopies+=size_t(copy);
#ifdef GARNET_METAL_FUSED_Q4_WIDE_SELECTION
    if(a.rows==16)++impl_->fusedWideDecodeBlocks;
#endif
    impl_->affineProfileDispatches+=2;impl_->affineDecodeBlockDispatches+=2;
    impl_->affineProfileMaxThreads=std::max(impl_->affineProfileMaxThreads,size_t(plan->second.maxTotalThreadsPerThreadgroup));
    if(selectedRows)impl_->indexedDecodeAffineDispatches+=2;
    return true;
 }
}
