// SPDX-License-Identifier: Apache-2.0
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metal_context.h"
#include "linear_chunk_original_shader.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
using namespace Garnet;
namespace {
void Need(bool value,const std::string& why){if(!value)throw std::runtime_error(why);}
std::vector<float> Values(size_t count,uint32_t seed,float scale) {
    std::vector<float> values(count);
    for(auto& value:values){seed=1664525u*seed+1013904223u;value=float(int((seed>>8)%65521)-32760)*scale;}
    return values;
}
std::vector<float> Mask(size_t batch,size_t chunk,size_t mode,size_t step) {
    std::vector<float> mask(batch*chunk);
    for(size_t row=0;row<batch;++row)for(size_t token=0;token<chunk;++token) {
        bool enabled=false;
        switch((mode+step)%6) {
        case 0:enabled=true;break;
        case 1:break;
        case 2:enabled=row%2==0&&token<std::max(size_t(1),chunk/3);break;
        case 3:enabled=(token+row)%3!=1;break;
        case 4:enabled=token+1==chunk;break;
        case 5:enabled=(token+row)%2==0;break;
        }
        mask[row*chunk+token]=float(enabled);
    }
    return mask;
}
struct Result{std::vector<float> output,state;};
struct Reference {
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue=[device newCommandQueue];
    id<MTLLibrary> library;
    id<MTLComputePipelineState> conv,delta;
    Reference() {
        NSError* error=nil;
        library=[device newLibraryWithSource:[NSString stringWithUTF8String:kOriginalLinearChunkShader] options:nil error:&error];
        if(!library)throw std::runtime_error(error.localizedDescription.UTF8String);
        conv=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"causal_conv_chunk"] error:&error];
        delta=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"gated_delta_chunk"] error:&error];
        Need(conv&&delta&&delta.threadExecutionWidth==32&&delta.maxTotalThreadsPerThreadgroup>=128,"Original pipelines unavailable");
    }
    Result Run(bool recurrent,std::initializer_list<const std::vector<float>*> inputs,
               const void* parameters,size_t bytes,size_t groups,size_t outputWords,size_t stateWords) {
        @autoreleasepool {
            NSMutableArray<id<MTLBuffer>>* buffers=[NSMutableArray new];
            for(auto input:inputs) {
                auto buffer=[device newBufferWithBytes:input->data() length:input->size()*4 options:MTLResourceStorageModeShared];
                Need(buffer!=nil,"Original input allocation failed");[buffers addObject:buffer];
            }
            auto output=[device newBufferWithLength:outputWords*4 options:MTLResourceStorageModeShared];
            auto state=[device newBufferWithLength:stateWords*4 options:MTLResourceStorageModeShared];
            Need(output&&state,"Original output allocation failed");
            std::fill_n(static_cast<float*>(output.contents),outputWords,std::numeric_limits<float>::quiet_NaN());
            std::fill_n(static_cast<float*>(state.contents),stateWords,std::numeric_limits<float>::quiet_NaN());
            [buffers addObject:output];[buffers addObject:state];
            auto command=[queue commandBuffer];auto encoder=[command computeCommandEncoder];
            [encoder setComputePipelineState:recurrent?delta:conv];
            for(size_t i=0;i<buffers.count;++i)[encoder setBuffer:buffers[i] offset:0 atIndex:i];
            [encoder setBytes:parameters length:bytes atIndex:buffers.count];
            [encoder dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
            [encoder endEncoding];[command commit];[command waitUntilCompleted];
            if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(command.error.localizedDescription.UTF8String);
            Result result{std::vector<float>(outputWords),std::vector<float>(stateWords)};
            std::memcpy(result.output.data(),output.contents,outputWords*4);std::memcpy(result.state.data(),state.contents,stateWords*4);
            return result;
        }
    }
};
size_t checks=0,words=0,pendingChecks=0;
void Same(MetalContext& gpu,const MetalMatrix& output,const MetalMatrix& next,const Result& expected,const std::string& label) {
    auto actual=gpu.Download(output),state=gpu.Download(next);
    Need(actual.size()==expected.output.size()&&state.size()==expected.state.size(),label+" sizes differ");
    Need(!std::memcmp(actual.data(),expected.output.data(),actual.size()*4),label+" output bits differ");
    Need(!std::memcmp(state.data(),expected.state.data(),state.size()*4),label+" state bits differ");
    for(float value:actual)Need(std::isfinite(value),label+" output not finite");
    for(float value:state)Need(std::isfinite(value),label+" state not finite");
    ++checks;words+=actual.size()+state.size();
}
void Pending(MetalContext& gpu,const MetalMatrix& output,const MetalMatrix& next) {
    for(const auto& matrix:{output,next}) {
        bool rejected=false;try{gpu.Download(matrix);}catch(const std::logic_error&){rejected=true;}
        Need(rejected,"Pending output/state readable");++pendingChecks;
    }
}
void Delta(MetalContext& gpu,Reference& ref,size_t batch,size_t chunk,size_t heads,size_t width,size_t mode) {
    const size_t qw=heads*128,vw=heads*width,pw=2*qw+vw+2*heads,sw=qw*width;
    auto initial=Values(batch*sw,153,.000001f);auto state=gpu.Upload(initial,batch,sw);
    auto output=gpu.Allocate(batch*chunk,vw),next=gpu.Allocate(batch,sw);
    const auto label="delta B"+std::to_string(batch)+" T"+std::to_string(chunk)+" H"+std::to_string(heads)+" V"+std::to_string(width)+" mode"+std::to_string(mode);
    for(size_t step=0;step<2;++step) {
        @autoreleasepool {
            auto mask=Mask(batch,chunk,mode,step),input=Values(batch*chunk*pw,393+uint32_t(step),.00001f);
            for(size_t row=0;row<batch;++row)for(size_t token=0;token<chunk;++token) {
                const size_t base=(row*chunk+token)*pw;
                for(size_t h=0;h<heads;++h){input[base+2*qw+vw+h]=-.1f-float((h+token)%11)*.03f;input[base+2*qw+vw+heads+h]=.2f+float((h+token)%7)*.1f;}
                if(mask[row*chunk+token]==0)std::fill_n(input.data()+base,pw,std::numeric_limits<float>::quiet_NaN());
            }
            struct Params{uint64_t batch,chunk,heads,keyDim,valueDim,pStride,sStride,aStride,oStride,nextStride;};
            Params p{batch,chunk,heads,128,width,pw,sw,chunk,vw,sw};
            auto expected=ref.Run(true,{&input,&initial,&mask},&p,sizeof(p),batch*heads*width/16,batch*chunk*vw,batch*sw);
            auto prepared=gpu.Upload(input,batch*chunk,pw),source=gpu.Upload(mask,batch,chunk);
            auto active=step?gpu.Allocate(batch,chunk):source;
            auto zero=gpu.Upload(std::vector<float>(batch*chunk,0),batch,chunk);
            gpu.Update(output,std::vector<float>(batch*chunk*vw,std::numeric_limits<float>::quiet_NaN()));
            gpu.Update(next,std::vector<float>(batch*sw,std::numeric_limits<float>::quiet_NaN()));
            gpu.Begin();
            if(step){gpu.EncodeAdd(source,zero,active);if(mode==3)gpu.Flush();}
            gpu.EncodeGatedDeltaChunk(prepared,state,active,output,next,chunk,heads,128,width);
            Pending(gpu,output,next);gpu.SubmitAndWait();Same(gpu,output,next,expected,label);
            initial=expected.state;gpu.Update(state,initial);
        }
    }
}
void Conv(MetalContext& gpu,Reference& ref,size_t batch,size_t chunk,size_t channels,size_t kernel,size_t mode) {
    auto initial=Values(batch*channels*kernel,151,.00001f),weight=Values(channels*kernel,49,.00001f);
    auto state=gpu.Upload(initial,batch,channels*kernel),weights=gpu.Upload(weight,channels,kernel);
    auto output=gpu.Allocate(batch*chunk,channels),next=gpu.Allocate(batch,channels*kernel);
    const auto label="conv B"+std::to_string(batch)+" T"+std::to_string(chunk)+" C"+std::to_string(channels)+" K"+std::to_string(kernel)+" mode"+std::to_string(mode);
    for(size_t step=0;step<2;++step) {
        @autoreleasepool {
            auto mask=Mask(batch,chunk,mode,step),input=Values(batch*chunk*channels,833+uint32_t(step),.00001f);
            for(size_t row=0;row<batch;++row)for(size_t token=0;token<chunk;++token)
                if(mask[row*chunk+token]==0)std::fill_n(input.data()+(row*chunk+token)*channels,channels,std::numeric_limits<float>::quiet_NaN());
            struct Params{uint64_t batch,chunk,channels,kSize,tStride,wStride,sStride,aStride,oStride,nStride;};
            Params p{batch,chunk,channels,kernel,channels,kernel,channels*kernel,chunk,channels,channels*kernel};
            auto expected=ref.Run(false,{&input,&weight,&initial,&mask},&p,sizeof(p),(batch*channels+127)/128,batch*chunk*channels,batch*channels*kernel);
            auto tokens=gpu.Upload(input,batch*chunk,channels),source=gpu.Upload(mask,batch,chunk);
            auto active=step?gpu.Allocate(batch,chunk):source;
            auto zero=gpu.Upload(std::vector<float>(batch*chunk,0),batch,chunk);
            gpu.Update(output,std::vector<float>(batch*chunk*channels,std::numeric_limits<float>::quiet_NaN()));
            gpu.Update(next,std::vector<float>(batch*channels*kernel,std::numeric_limits<float>::quiet_NaN()));
            gpu.Begin();
            if(step){gpu.EncodeAdd(source,zero,active);if(mode==3)gpu.Flush();}
            gpu.EncodeCausalConvChunk(tokens,weights,state,active,output,next,chunk,kernel);
            Pending(gpu,output,next);gpu.SubmitAndWait();Same(gpu,output,next,expected,label);
            initial=expected.state;gpu.Update(state,initial);
        }
    }
}
void Lifecycle(MetalContext& gpu,MetalContext& other,Reference& ref) {
    constexpr size_t batch=2,chunk=4,heads=2,width=16,qw=heads*128,vw=heads*width,pw=2*qw+vw+2*heads,sw=qw*width;
    auto input=Values(batch*chunk*pw,13,.00001f),initial=Values(batch*sw,37,.000001f),mask=Mask(batch,chunk,3,0);
    for(size_t row=0;row<batch;++row)for(size_t t=0;t<chunk;++t)for(size_t h=0;h<heads;++h){input[(row*chunk+t)*pw+2*qw+vw+h]=-.2f;input[(row*chunk+t)*pw+2*qw+vw+heads+h]=.3f;}
    auto prepared=gpu.Upload(input,batch*chunk,pw),state=gpu.Upload(initial,batch,sw),active=gpu.Upload(mask,batch,chunk),output=gpu.Allocate(batch*chunk,vw),next=gpu.Allocate(batch,sw);
    auto encode=[&](MetalContext& owner,const MetalMatrix& data,const MetalMatrix& target){owner.EncodeGatedDeltaChunk(data,state,active,output,target,chunk,heads,128,width);};
    gpu.Begin();encode(gpu,prepared,next);gpu.Abort();Pending(gpu,output,next);
    gpu.Begin();encode(gpu,prepared,next);gpu.SubmitAndWait();
    struct Params{uint64_t batch,chunk,heads,keyDim,valueDim,pStride,sStride,aStride,oStride,nextStride;};
    Params p{batch,chunk,heads,128,width,pw,sw,chunk,vw,sw};
    auto expected=ref.Run(true,{&input,&initial,&mask},&p,sizeof(p),batch*heads*width/16,batch*chunk*vw,batch*sw);
    Same(gpu,output,next,expected,"abort recovery");
    gpu.Begin();bool rejected=false;try{encode(gpu,prepared,state);}catch(const std::invalid_argument&){rejected=true;}gpu.Abort();Need(rejected,"In-place state accepted");
    auto missing=gpu.Allocate(batch*chunk,pw);
    gpu.Begin();rejected=false;try{encode(gpu,missing,next);}catch(const std::logic_error&){rejected=true;}gpu.Abort();Need(rejected,"Uninitialized producer accepted");
    gpu.Begin();encode(gpu,prepared,next);other.Begin();rejected=false;
    try{encode(other,prepared,next);}catch(const std::logic_error&){rejected=true;}
    other.Abort();Need(rejected,"Foreign writer accepted");gpu.SubmitAndWait();Same(gpu,output,next,expected,"foreign writer recovery");
}
}
int main(){try{
#ifndef GARNET_METAL_LINEAR_CHUNK_ACTIVE_ONLY
    throw std::runtime_error("Active-only test requires its explicit candidate build");
#endif
    @autoreleasepool {
        Reference ref;MetalContext gpu,other;
        Need(ref.device.registryID==MetalContext::MemoryStats().registry,"Reference and candidate devices differ");
        for(size_t batch:{1,2,4})for(size_t chunk:{1,4,64,128,256})for(size_t width:{16,32,128})for(size_t mode=0;mode<6;++mode)
            Delta(gpu,ref,batch,chunk,2,width,mode);
        for(size_t chunk:{128,256})for(size_t mode:{0,1,2,3,4,5})Delta(gpu,ref,2,chunk,48,128,mode);
        for(size_t batch:{1,2,4})for(size_t chunk:{1,4,64,128,256})for(size_t channels:{3,129})for(size_t kernel:{1,4,8})for(size_t mode=0;mode<6;++mode)
            Conv(gpu,ref,batch,chunk,channels,kernel,mode);
        for(size_t chunk:{128,256})for(size_t mode:{0,1,2,3,4,5})Conv(gpu,ref,2,chunk,10240,4,mode);
        Lifecycle(gpu,other,ref);
    }
    const auto owners=MetalContext::ProfileOwnerStats();
    Need(!owners.matrices&&!owners.affineWeights&&!owners.bf16Weights&&!owners.pipelineSets,"Native owners remain after close");
    std::cout<<"linear chunk active-only original reference PASS checks="<<checks<<" fp32_words="<<words<<" pending_rejections="<<pendingChecks
             <<" carried_state=2 inactive_nan_poison=true gpu_masks=true partial_submissions=true abort_alias_uninitialized_foreign=true owners=0\n";
    return 0;
}catch(const std::exception& error){std::cerr<<"linear chunk active-only reference FAILED: "<<error.what()<<"\n";return 1;}}
