// SPDX-License-Identifier: Apache-2.0
// Offline synthetic operator study, independent of the optional native route.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>
#include "../../../src/backends/metal/affine_int4_profile_shader.h"
#include "../../../src/backends/metal/affine_int4_mpp_shader.h"
#include "../../../src/backends/metal/affine_int4_reuse64_shader.h"
#include "../../../src/backends/metal/affine_int4_reuse128_shader.h"
#include "../../../src/model/float16_codec.h"
#include "../../../third_party/nlohmann/json.hpp"
using json=nlohmann::json;
void Need(bool b,const char* why){if(!b)throw std::runtime_error(why);}
struct Params{uint64_t rows,inner,columns,aStride,oStride;};
const char* helpers=R"MSL(
kernel void unpack(device const uint* w [[buffer(0)]],device const half* s [[buffer(1)]],device const half* b [[buffer(2)]],device half* expanded [[buffer(3)]],constant uint2& shape [[buffer(4)]],uint i [[thread_position_in_grid]]){
 if(i>=shape.x*shape.y)return;uint n=i/shape.y,k=i%shape.y,c=n*(shape.y/64)+k/64;expanded[i]=half(fma(float((w[n*(shape.y/8)+k/8]>>(4*(k%8)))&15u),float(s[c]),float(b[c])));
}
kernel void cast_input(device const float* a [[buffer(0)]],device half* h [[buffer(1)]],constant uint& count [[buffer(2)]],uint i [[thread_position_in_grid]]){if(i<count)h[i]=half(a[i]);}
kernel void sweep(device const float* a [[buffer(0)]],device float* out [[buffer(1)]],uint i [[thread_position_in_grid]]){float sum=0;for(uint j=0;j<256;++j)sum+=a[i+j*262144];out[i]=sum;}
)MSL";
int main(int argc,char** argv){@autoreleasepool{try{
 Need(argc==3,"Usage NEW_OUTPUT REPS");const std::filesystem::path root=argv[1];Need(std::filesystem::create_directory(root),"Fresh output required");const int reps=std::stoi(argv[2]);Need(reps>=3&&reps<=31,"Invalid repetitions");
 auto device=MTLCreateSystemDefaultDevice();auto queue=[device newCommandQueue];Need(device&&queue,"Metal unavailable");NSError* error=nil;
 NSString* source=[@"#define GARNET_AFFINE_PROFILE 1\n#define GARNET_AFFINE_INDEX32 1\n#define GARNET_AFFINE_FP16_COEFFICIENTS 1\n#define GARNET_MPP_FP16 1\n#define GARNET_MPP_REGISTERS 1\n#define GARNET_MPP_K 64\n#define GARNET_REUSE64_RELAXED 0\n" stringByAppendingString:[NSString stringWithUTF8String:Garnet::kAffineInt4ProfilePrelude]];
 for(auto text:{Garnet::kAffineInt4MPPShader,Garnet::kAffineInt4Reuse64Shader,Garnet::kAffineInt4Reuse128Shader,helpers})source=[source stringByAppendingString:[NSString stringWithUTF8String:text]];
 auto opts=[MTLCompileOptions new];opts.languageVersion=MTLLanguageVersion4_0;opts.mathMode=MTLMathModeSafe;
 auto lib=[device newLibraryWithSource:source options:opts error:&error];Need(lib!=nil,error?error.localizedDescription.UTF8String:"Metal compile failed");
 auto plain=[&](NSString* name){auto p=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:name] error:&error];Need(p!=nil,"Helper pipeline failed");return p;};
 auto unpack=plain(@"unpack"),cast=plain(@"cast_input"),flush=plain(@"sweep");
 auto buffer=[&](size_t bytes){auto v=[device newBufferWithLength:bytes options:MTLResourceStorageModeShared];Need(v!=nil,"Allocation failed");return v;};
 auto trash=buffer(256ull*1024*1024),sink=buffer(1024*1024);std::fill_n((float*)trash.contents,64*1024*1024,.125f);
 auto finish=[&](id<MTLCommandBuffer> cmd){[cmd commit];[cmd waitUntilCompleted];Need(cmd.status==MTLCommandBufferStatusCompleted,"GPU command failed");};
 auto sweep=[&](){auto cmd=[queue commandBuffer];auto e=[cmd computeCommandEncoder];[e setComputePipelineState:flush];[e setBuffer:trash offset:0 atIndex:0];[e setBuffer:sink offset:0 atIndex:1];[e dispatchThreads:MTLSizeMake(262144,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[e endEncoding];finish(cmd);};
 std::cout<<json{{"kind","metadata"},{"device",device.name.UTF8String},{"warmups",3},{"reps",reps},{"reuse64_control_threads",128},{"scope","FP16 operands/FP32 requested output; serial synthetic operator experiment. Vendor internal accumulation is not separately instrumented. Finite poisoned coverage and sampled FP64 checks in-process; full-output FP64 audit required separately. Sweep is not a proof of cold cache; no model or serving qualification."}}.dump()<<'\n'<<std::flush;
 size_t index=0,checked=0;
 for(auto shape:std::vector<std::array<uint32_t,2>>{{14336,4096},{4096,14336},{4096,4096}})for(uint32_t m:{128u,256u,512u}){@autoreleasepool{
  const uint32_t n=shape[0],k=shape[1];Params p{m,k,n,k,n};
  auto a=buffer(size_t(m)*k*4),ah=buffer(size_t(m)*k*2),w=buffer(size_t(n)*k/2),s=buffer(size_t(n)*k/32),b=buffer(size_t(n)*k/32),wh=buffer(size_t(n)*k*2),out=buffer(size_t(m)*n*4),ids=buffer(m*4);
  uint32_t seed=37;for(size_t i=0;i<size_t(m)*k;++i){seed=1664525u*seed+1013904223u;((float*)a.contents)[i]=float(int((seed>>8)%65521)-32760)*0x1p-18f;}
  for(size_t i=0;i<size_t(n)*k/8;++i)((uint32_t*)w.contents)[i]=uint32_t(i)*2654435761u+131u;
  for(size_t i=0;i<size_t(n)*k/64;++i){((uint16_t*)s.contents)[i]=uint16_t(0x2c00+i%8);((uint16_t*)b.contents)[i]=uint16_t(0xb000+i%7);}
  for(uint32_t i=0;i<m;++i)((uint32_t*)ids.contents)[i]=i;
  auto constants=[MTLFunctionConstantValues new];const uint32_t dims[]{k,n,k,n};for(NSUInteger i=0;i<4;++i)[constants setConstantValue:dims+i type:MTLDataTypeUInt atIndex:10+i];
  std::array<id<MTLComputePipelineState>,3> controls{};std::array<NSString*,3> names{@"affine_q4_mpp",@"affine_q4_reuse64",@"affine_q4_reuse128"};
  for(size_t i=0;i<3;++i){auto fn=[lib newFunctionWithName:names[i] constantValues:constants error:&error];Need(fn!=nil,"Control specialization failed");controls[i]=[device newComputePipelineStateWithFunction:fn error:&error];Need(controls[i]!=nil,"Control pipeline failed");}
  auto av=[[MPSMatrix alloc] initWithBuffer:ah descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:m columns:k rowBytes:k*2 dataType:MPSDataTypeFloat16]];
  auto bv=[[MPSMatrix alloc] initWithBuffer:wh descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:n columns:k rowBytes:k*2 dataType:MPSDataTypeFloat16]];
  auto cv=[[MPSMatrix alloc] initWithBuffer:out descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:m columns:n rowBytes:n*4 dataType:MPSDataTypeFloat32]];
  auto mm=[[MPSMatrixMultiplication alloc] initWithDevice:device transposeLeft:NO transposeRight:YES resultRows:m resultColumns:n interiorColumns:k alpha:1 beta:0];Need(av&&bv&&cv&&mm,"MPS setup failed");
  auto encodeUnpack=[&](id<MTLCommandBuffer> cmd){auto e=[cmd computeCommandEncoder];[e setComputePipelineState:unpack];[e setBuffer:w offset:0 atIndex:0];[e setBuffer:s offset:0 atIndex:1];[e setBuffer:b offset:0 atIndex:2];[e setBuffer:wh offset:0 atIndex:3];uint32_t shape[]{n,k};[e setBytes:shape length:sizeof(shape) atIndex:4];[e dispatchThreads:MTLSizeMake(size_t(n)*k,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding];};
  std::fill_n((uint16_t*)wh.contents,size_t(n)*k,uint16_t(0x7eab));auto prep=[queue commandBuffer];encodeUnpack(prep);finish(prep);double preparation=(prep.GPUEndTime-prep.GPUStartTime)*1000;
  for(size_t i=0;i<size_t(n)*k;++i){uint32_t col=i/k,j=i%k,coefficient=col*(k/64)+j/64;uint32_t code=(((uint32_t*)w.contents)[i/8]>>(4*(j%8)))&15;auto expected=Garnet::Float32ToFloat16RNE(std::fma(float(code),Garnet::Float16ToFloat32(((uint16_t*)s.contents)[coefficient]),Garnet::Float16ToFloat32(((uint16_t*)b.contents)[coefficient])));Need(((uint16_t*)wh.contents)[i]==expected,"Expanded weight differs from exact half dequant");}
  std::vector<double> expected(9),bound(9);size_t z=0;
  for(uint32_t row:{0u,m/2,m-1})for(uint32_t col:{0u,n/2,n-1}){double sum=0,absolute=0;for(uint32_t j=0;j<k;++j){double product=double(Garnet::Float16ToFloat32(Garnet::Float32ToFloat16RNE(((float*)a.contents)[size_t(row)*k+j])))*Garnet::Float16ToFloat32(((uint16_t*)wh.contents)[size_t(col)*k+j]);sum+=product;absolute+=std::abs(product);}expected[z]=sum;bound[z++]=2e-6+4e-6*absolute;}
  std::array<std::string,5> labels{"mpp64","reuse64","reuse128","mps_cached","mps_streamed"};
  for(bool swept:{false,true}){
   std::array<std::vector<double>,5> gpu,wall;std::array<std::vector<float>,5> first;std::array<double,5> errorFraction{};
   for(int rep=-3;rep<reps;++rep)for(size_t order=0;order<5;++order){size_t v=(order+size_t(rep+3))%5;if((rep+3)&1)v=4-v;if(swept)sweep();std::fill_n((float*)out.contents,size_t(m)*n,std::numeric_limits<float>::quiet_NaN());
    const auto begin=std::chrono::steady_clock::now();auto cmd=[queue commandBuffer];
    if(v<3){auto e=[cmd computeCommandEncoder];[e setComputePipelineState:controls[v]];[e setBuffer:a offset:0 atIndex:0];[e setBuffer:w offset:0 atIndex:1];[e setBuffer:s offset:0 atIndex:2];[e setBuffer:b offset:0 atIndex:3];[e setBuffer:out offset:0 atIndex:4];[e setBytes:&p length:sizeof(p) atIndex:5];if(v)[e setBuffer:ids offset:0 atIndex:6];[e dispatchThreadgroups:MTLSizeMake((n+(v?64:32)-1)/(v?64:32),(m+(v==2?128:64)-1)/(v==2?128:64),1) threadsPerThreadgroup:MTLSizeMake(v==2?256:128,1,1)];[e endEncoding];}
    else{if(v==4)encodeUnpack(cmd);auto e=[cmd computeCommandEncoder];[e setComputePipelineState:cast];[e setBuffer:a offset:0 atIndex:0];[e setBuffer:ah offset:0 atIndex:1];uint32_t count=m*k;[e setBytes:&count length:sizeof(count) atIndex:2];[e dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding];[mm encodeToCommandBuffer:cmd leftMatrix:av rightMatrix:bv resultMatrix:cv];}
    finish(cmd);auto elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();const auto values=(float*)out.contents;
    for(size_t i=0;i<size_t(m)*n;++i)Need(std::isfinite(values[i]),"Missing/nonfinite output after poison");
    if(first[v].empty())first[v].assign(values,values+size_t(m)*n);else Need(std::memcmp(first[v].data(),values,size_t(m)*n*4)==0,"Nondeterministic output");
    z=0;for(uint32_t row:{0u,m/2,m-1})for(uint32_t col:{0u,n/2,n-1}){double f=std::abs(double(values[size_t(row)*n+col])-expected[z])/bound[z];Need(f<=1.,"Original sampled FP64 bound exceeded");errorFraction[v]=std::max(errorFraction[v],f);++z;++checked;}
    if(rep>=0){gpu[v].push_back((cmd.GPUEndTime-cmd.GPUStartTime)*1000);wall[v].push_back(elapsed);}
   }
   for(size_t v=0;v<5;++v){std::string file="case-"+std::to_string(index)+"-"+(swept?"swept-":"hot-")+labels[v]+".f32";std::ofstream stream(root/file,std::ios::binary);stream.write((char*)first[v].data(),first[v].size()*4);Need(bool(stream),"Cannot save output");std::cout<<json{{"kind","cell"},{"index",index},{"rows",m},{"outputs",n},{"inner",k},{"variant",labels[v]},{"swept",swept},{"gpu_ms",gpu[v]},{"wall_ms",wall[v]},{"sampled_fp64_fraction",errorFraction[v]},{"weight_prepare_gpu_ms",preparation},{"expanded_weight_bytes",size_t(n)*k*2},{"metal_allocated_bytes",device.currentAllocatedSize},{"output",file}}.dump()<<'\n'<<std::flush;}
  }++index;
 }}
 std::cout<<json{{"kind","closed"},{"cases",index},{"sampled_fp64_checks",checked}}.dump()<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}}
