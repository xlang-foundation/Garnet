// SPDX-License-Identifier: Apache-2.0
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../../src/backends/metal/packed_affine_layout_guard.h"
#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
using namespace Garnet;
void Need(bool value,const char* message){if(!value)throw std::runtime_error(message);}
std::array<uint32_t,32> Run(id<MTLDevice> device,id<MTLCommandQueue> queue,NSString* source,id<MTLLibrary> supplied=nil) {
    NSError* error=nil;auto options=[MTLCompileOptions new];options.languageVersion=MTLLanguageVersion4_0;options.mathMode=MTLMathModeSafe;
    auto library=supplied?supplied:[device newLibraryWithSource:source options:options error:&error];Need(library!=nil,error?error.localizedDescription.UTF8String:"Compile guard");
    auto pipeline=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"packed_affine_layout_guard"] error:&error];
    Need(pipeline&&pipeline.threadExecutionWidth==32&&pipeline.maxTotalThreadsPerThreadgroup>=32,"Guard pipeline limits");
    auto status=[device newBufferWithLength:32*sizeof(uint32_t) options:MTLResourceStorageModeShared];Need(status!=nil,"Guard status allocation");
    std::fill_n(static_cast<uint32_t*>(status.contents),32,UINT32_MAX);
    auto command=[queue commandBuffer];auto encoder=[command computeCommandEncoder];Need(command&&encoder,"Guard encoder");
    [encoder setComputePipelineState:pipeline];[encoder setBuffer:status offset:0 atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[encoder endEncoding];
    [command commit];[command waitUntilCompleted];Need(command.status==MTLCommandBufferStatusCompleted,"Guard GPU completion");
    std::array<uint32_t,32> words;std::copy_n(static_cast<const uint32_t*>(status.contents),32,words.begin());return words;
}
int main(int argc,char** argv){@autoreleasepool{try{
    Need(argc==1||(argc==3&&std::string(argv[1])=="--offline"),"Usage [--offline GUARDED_LIBRARY]");auto device=MTLCreateSystemDefaultDevice();auto queue=[device newCommandQueue];Need(device&&queue,"Metal required");
    if(argc==3){
        NSError* error=nil;auto library=[device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[2]]] error:&error];
        Need(library!=nil,error?error.localizedDescription.UTF8String:"Offline guard library");auto words=Run(device,queue,nil,library);
        Need(!PackedAffineLayoutMatches(words.data(),words.size()),"Expected incompatible offline ABI was accepted");
        std::cout<<"{\"expected_incompatible_layout_rejected\":true,\"status\":[";
        for(size_t i=0;i<words.size();++i){Need(words[i]!=UINT32_MAX,"Unwritten offline lane");if(i)std::cout<<',';std::cout<<words[i];}
        std::cout<<"]}\n";return 0;
    }
    NSString* source=[NSString stringWithUTF8String:kPackedAffineLayoutGuardShader];
    auto good=Run(device,queue,source);Need(PackedAffineLayoutMatches(good.data(),good.size()),"Active compiler layout is incompatible with packed prefill");
    Need(!PackedAffineLayoutMatches(nullptr,32)&&!PackedAffineLayoutMatches(good.data(),0)&&!PackedAffineLayoutMatches(good.data(),31)&&!PackedAffineLayoutMatches(good.data(),33),"Status extent/null guards");
    for(size_t lane=0;lane<32;++lane){auto bad=good;bad[lane]=1;Need(!PackedAffineLayoutMatches(bad.data(),bad.size()),"Ignored failed lane");bad[lane]=UINT32_MAX;Need(!PackedAffineLayoutMatches(bad.data(),bad.size()),"Ignored unwritten lane");}
    // Compile a deliberately incompatible expected capacity; every lane must
    // execute and report the mismatch, rather than retaining poisoned status.
    NSString* fault=[source stringByReplacingOccurrencesOfString:@"left.get_capacity()!=8" withString:@"left.get_capacity()!=9"];
    Need(![fault isEqualToString:source],"Missing fault mutation");auto rejected=Run(device,queue,fault);
    Need(!PackedAffineLayoutMatches(rejected.data(),rejected.size()),"Wrong-layout shader accepted");
    for(auto word:rejected)Need(word!=UINT32_MAX&&(word&1),"Wrong-layout lane was not written/rejected");
    std::cout<<"Packed cooperative layout: live ABI, capacity fault, 64 lane faults and 4 status bounds PASS\n";return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}}
