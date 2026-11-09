// SPDX-License-Identifier: Apache-2.0
// Reproduce the admitted runtime source/options/constants; archive original
// pipelines for static inspection. No model loading or performance measurement.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include "affine_int4_profile_shader.h"
#include "packed_affine_shader.h"
#include "paired_prefill_shader.h"
#include "packed_affine_layout_guard.h"
static void Need(bool value,const char* message){if(!value)throw std::runtime_error(message);}
static void Check(bool value,NSError* error,const char* message){if(!value)throw std::runtime_error(std::string(message)+(error?std::string(": ")+error.localizedDescription.UTF8String:""));}
int main(int argc,const char* argv[]){@autoreleasepool{try{
 Need(argc==2,"Expected new output directory");NSString* root=[NSString stringWithUTF8String:argv[1]];auto fm=[NSFileManager defaultManager];Need(![fm fileExistsAtPath:root],"Refuse existing output");NSError* error=nil;Check([fm createDirectoryAtPath:root withIntermediateDirectories:NO attributes:nil error:&error],error,"Create output");
 id<MTLDevice> device=MTLCreateSystemDefaultDevice();Need(device!=nil,"No Metal device");Need([device supportsFamily:MTLGPUFamilyApple10],"Expected qualified Apple10 device");
 NSString* source=[@"#define GARNET_AFFINE_PROFILE 1\n#define GARNET_AFFINE_INDEX32 1\n#define GARNET_AFFINE_FP16_COEFFICIENTS 1\n" stringByAppendingString:[NSString stringWithUTF8String:Garnet::kAffineInt4ProfilePrelude]];
 source=[source stringByAppendingString:[NSString stringWithUTF8String:Garnet::kPackedAffineShader]];
 source=[source stringByAppendingString:[NSString stringWithUTF8String:Garnet::kPairedPrefillShader]];
 source=[source stringByAppendingString:[NSString stringWithUTF8String:Garnet::kPackedAffineLayoutGuardShader]];
 Check([source writeToFile:[root stringByAppendingPathComponent:@"runtime-source.metal"] atomically:YES encoding:NSUTF8StringEncoding error:&error],error,"Save exact source");
 auto options=[MTLCompileOptions new];options.languageVersion=MTLLanguageVersion4_0;options.mathMode=MTLMathModeSafe;
 id<MTLLibrary> library=[device newLibraryWithSource:source options:options error:&error];Check(library!=nil,error,"Compile runtime source");
 auto guard=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"packed_affine_layout_guard"] error:&error];Check(guard!=nil,error,"Compile native guard");Need(guard.threadExecutionWidth==32&&guard.maxTotalThreadsPerThreadgroup>=32,"Guard limits");
 auto status=[device newBufferWithLength:128 options:MTLResourceStorageModeShared];Need(status!=nil,"Guard status allocation");std::fill_n(static_cast<uint32_t*>(status.contents),32,UINT32_MAX);auto queue=[device newCommandQueue];auto command=[queue commandBuffer];auto encoder=[command computeCommandEncoder];[encoder setComputePipelineState:guard];[encoder setBuffer:status offset:0 atIndex:0];[encoder dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[encoder endEncoding];[command commit];[command waitUntilCompleted];Need(command.status==MTLCommandBufferStatusCompleted,"Guard dispatch failed");Need(Garnet::PackedAffineLayoutMatches(static_cast<uint32_t*>(status.contents),32),"Runtime cooperative ABI differs");
 NSMutableArray* records=[NSMutableArray array];
 const std::array<std::array<uint32_t,2>,3> shapes{{{{4096,14336}},{{14336,4096}},{{4096,4096}}}};
 for(auto shape:shapes)for(size_t tile:{size_t(128),size_t(256)}){@autoreleasepool{
  const uint32_t dimensions[]{shape[0],shape[1],shape[0],shape[1]};auto constants=[MTLFunctionConstantValues new];for(NSUInteger i=0;i<4;++i)[constants setConstantValue:dimensions+i type:MTLDataTypeUInt atIndex:10+i];NSString* name=tile==128?@"half_m128k128":@"half_m256k128";
  auto function=[library newFunctionWithName:name constantValues:constants error:&error];Check(function!=nil,error,"Specialize original kernel");auto original=[device newComputePipelineStateWithFunction:function error:&error];Check(original!=nil,error,"Compile original policy");Need(original.threadExecutionWidth==32&&original.maxTotalThreadsPerThreadgroup>=256&&original.staticThreadgroupMemoryLength==17408,"Original pipeline resources differ");
  auto archive=[device newBinaryArchiveWithDescriptor:[MTLBinaryArchiveDescriptor new] error:&error];Check(archive!=nil,error,"Create binary archive");auto descriptor=[MTLComputePipelineDescriptor new];descriptor.computeFunction=function;Check([archive addComputePipelineFunctionsWithDescriptor:descriptor error:&error],error,"Add original compute pipeline");
  NSString* file=[NSString stringWithFormat:@"%@-K%u-N%u.metalar",name,shape[0],shape[1]];NSURL* url=[NSURL fileURLWithPath:[root stringByAppendingPathComponent:file]];Check([archive serializeToURL:url error:&error],error,"Serialize pipeline archive");
  auto reload=[MTLBinaryArchiveDescriptor new];reload.url=url;auto loaded=[device newBinaryArchiveWithDescriptor:reload error:&error];Check(loaded!=nil,error,"Reload binary archive");descriptor.binaryArchives=@[loaded];auto fromArchive=[device newComputePipelineStateWithDescriptor:descriptor options:MTLPipelineOptionFailOnBinaryArchiveMiss reflection:nil error:&error];Check(fromArchive!=nil,error,"Explicit archive lookup miss");Need(fromArchive.threadExecutionWidth==original.threadExecutionWidth&&fromArchive.maxTotalThreadsPerThreadgroup==original.maxTotalThreadsPerThreadgroup&&fromArchive.staticThreadgroupMemoryLength==original.staticThreadgroupMemoryLength,"Archive resource metadata differs");
  auto empty=[device newBinaryArchiveWithDescriptor:[MTLBinaryArchiveDescriptor new] error:&error];Check(empty!=nil,error,"Create negative empty archive");descriptor.binaryArchives=@[empty];NSError* missError=nil;auto missing=[device newComputePipelineStateWithDescriptor:descriptor options:MTLPipelineOptionFailOnBinaryArchiveMiss reflection:nil error:&missError];Need(missing==nil&&missError!=nil,"Empty archive silently fell back to cached/recompiled code");
  [records addObject:@{@"name":name,@"tile":@(tile),@"K":@(shape[0]),@"N":@(shape[1]),@"function_constant_10":@(dimensions[0]),@"function_constant_11":@(dimensions[1]),@"function_constant_12":@(dimensions[2]),@"function_constant_13":@(dimensions[3]),@"archive":file,@"archive_reload_without_fallback":@YES,@"empty_archive_rejected":@YES,@"empty_archive_error_domain":missError.domain,@"empty_archive_error_code":@(missError.code),@"simd_width":@(original.threadExecutionWidth),@"maximum_threadgroup_threads":@(original.maxTotalThreadsPerThreadgroup),@"static_threadgroup_bytes":@(original.staticThreadgroupMemoryLength)}];
 }}
 NSDictionary* report=@{@"completed":@YES,@"scope":@"Original runtime-packed shader specialization archives, matching source/options/constants and public pipeline resource metadata. Static inspection only, no model, performance, occupancy/stall or new numerical qualification.",@"device":device.name,@"registry_id":@(device.registryID),@"language":@"Metal4.0",@"math_mode":@"Safe",@"maximum_logical_rows":@512,@"cooperative_abi_guard_passed":@YES,@"arithmetic_dispatches":@0,@"guard_dispatches":@1,@"records":records};
 NSData* data=[NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:&error];Check(data!=nil,error,"Encode report");Check([data writeToFile:[root stringByAppendingPathComponent:@"result.json"] options:NSDataWritingAtomic error:&error],error,"Save report");puts("Original packed runtime archive creation/reload PASS (6 pipelines)");return 0;
 }catch(const std::exception& error){fprintf(stderr,"archive inspection failed: %s\n",error.what());return 1;}}}
