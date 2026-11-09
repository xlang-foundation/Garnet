// SPDX-License-Identifier: Apache-2.0
// Exact native-captured gate/up/SiLU/down replay; no serving timing.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "nlohmann/json.hpp"
#include "src/model/float16_codec.h"
#include "src/backends/metal/packed_affine_layout_guard.h"
#include <array>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
using J=nlohmann::json;namespace fs=std::filesystem;
void Need(bool b,const char* s){if(!b)throw std::runtime_error(s);}
template<class T>std::vector<T> Read(fs::path p,size_t count){std::ifstream f(p,std::ios::binary|std::ios::ate);Need(bool(f)&&size_t(f.tellg())==count*sizeof(T),"Wrong replay input bytes");f.seekg(0);std::vector<T> a(count);f.read((char*)a.data(),count*sizeof(T));Need(bool(f),"Replay input read failed");return a;}
void Save(fs::path,const void*,size_t){}
void Report(fs::path p,const J& value){std::ofstream f(p);f<<value.dump(2)<<'\n';Need(bool(f),"Replay report failed");}
size_t Diff(const void* a,const void* b,size_t count,size_t bytes){size_t mismatches=0;auto x=(const unsigned char*)a,y=(const unsigned char*)b;for(size_t i=0;i<count;i++)mismatches+=std::memcmp(x+i*bytes,y+i*bytes,bytes)!=0;return mismatches;}
struct Projection{uint64_t rows,inner,columns,aStride,oStride;};
struct Pointwise{uint64_t rows,columns,aStride,bStride,cStride;float epsilon,offset;};
int main(int argc,char**argv){fs::path output;J result={{"completed",false},{"layers",J::array()}};
 @autoreleasepool{try{
  Need(argc==7,"COMBINED ORDINARY ASSETS CAPTURES FRESH_OUTPUT forward|reverse required");bool reverse=std::string(argv[6])=="reverse";Need(reverse||std::string(argv[6])=="forward","Invalid balanced order");result["reverse"]=reverse;result["samples"]=J::array();result["timing_witnesses"]=J::array();output=fs::absolute(argv[5]);Need(fs::create_directory(output),"Fresh replay output required");auto device=MTLCreateSystemDefaultDevice();Need(device&&[device supportsFamily:MTLGPUFamilyApple10],"Apple10/M5 required");auto queue=[device newCommandQueue];Need(queue!=nil,"Missing queue");NSError* error=nil;
  auto compile=[&](const char* file,MTLCompileOptions* options){auto text=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:file] encoding:NSUTF8StringEncoding error:&error];Need(text!=nil,"Read shader failed");auto library=[device newLibraryWithSource:text options:options error:&error];if(!library)throw std::runtime_error(error.localizedDescription.UTF8String);return library;};
  auto safeOptions=[MTLCompileOptions new];safeOptions.languageVersion=MTLLanguageVersion4_0;safeOptions.mathMode=MTLMathModeSafe;
  auto fastOptions=[MTLCompileOptions new];fastOptions.languageVersion=MTLLanguageVersion4_0;fastOptions.mathMode=MTLMathModeFast;
  auto explicit31=[MTLCompileOptions new];explicit31.languageVersion=MTLLanguageVersion3_1;
  auto safe=compile(argv[1],safeOptions),fast=compile(argv[1],fastOptions),ordinary=compile(argv[2],nil),surrogate=compile(argv[2],explicit31);
  auto pipe=[&](id<MTLLibrary> lib,NSString* name,MTLFunctionConstantValues* c=nil){auto f=c?[lib newFunctionWithName:name constantValues:c error:&error]:[lib newFunctionWithName:name];Need(f!=nil,"Missing replay function");auto p=[device newComputePipelineStateWithFunction:f error:&error];if(!p)throw std::runtime_error(error.localizedDescription.UTF8String);Need(p.threadExecutionWidth==32,"SIMD mismatch");return p;};
  auto buffer=[&](size_t bytes){auto b=[device newBufferWithLength:bytes options:MTLResourceStorageModeShared];Need(b!=nil,"Replay allocation failed");return b;};
  auto finish=[&](id<MTLCommandBuffer> cb){[cb commit];[cb waitUntilCompleted];Need(cb.status==MTLCommandBufferStatusCompleted,"Replay GPU command failed");};
  for(auto lib:{safe,fast}){auto guard=pipe(lib,@"packed_affine_layout_guard");auto b=buffer(128);std::fill_n((uint32_t*)b.contents,32,0xffffffffu);auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];[e setComputePipelineState:guard];[e setBuffer:b offset:0 atIndex:0];[e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[e endEncoding];finish(cb);Need(Garnet::PackedAffineLayoutMatches((uint32_t*)b.contents,32),"Replay cooperative layout mismatch");}
  constexpr size_t M=512,K=4096,N=14336;constexpr uint32_t GUARD=0x1937abef;constexpr uint16_t HGUARD=0x35a5;
  auto input=buffer((M*K+32)*4),compact=buffer((M*K+64)*2),half=buffer((M*N+64)*2),counts=buffer((M*N+32)*4);
  std::array<id<MTLBuffer>,4> outs{buffer((M*N+32)*4),buffer((M*N+32)*4),buffer((M*N+32)*4),buffer((M*K+32)*4)};
  std::vector<uint32_t> ids(M);for(size_t i=0;i<M;i++)ids[i]=uint32_t(i);
  auto constants=[&](size_t k,size_t n,bool shadow=false,bool writer=false){auto c=[MTLFunctionConstantValues new];uint32_t dims[]={uint32_t(k),uint32_t(n),uint32_t(k),uint32_t(n)};for(NSUInteger i=0;i<4;i++)[c setConstantValue:dims+i type:MTLDataTypeUInt atIndex:10+i];[c setConstantValue:&shadow type:MTLDataTypeBool atIndex:20];[c setConstantValue:&writer type:MTLDataTypeBool atIndex:21];return c;};
  auto gatePipe=pipe(safe,@"half_m256k128",constants(K,N)),downPipe=pipe(safe,@"half_m256k128",constants(N,K));auto downTensor=pipe(safe,@"shared_tensor_m256",constants(N,K));std::array<std::array<id<MTLComputePipelineState>,3>,2> fused;
  for(size_t method=0;method<2;method++)for(size_t v=0;v<3;v++){fused[method][v]=pipe(fast,method?@"fused_tensor_m256":@"fused_m256k128",constants(K,N,v==0,v==2));Need(fused[method][v].staticThreadgroupMemoryLength==17408&&fused[method][v].maxTotalThreadsPerThreadgroup>=256,"Fused pipeline resources differ");}
  auto gather=pipe(safe,@"packed_affine_gather"),activation=pipe(ordinary,@"silu_mul"),oldActivation=pipe(surrogate,@"silu_mul");
  size_t controlWords=0,halfWords=0,downWords=0;bool candidateExact=true;
  for(size_t layer:(reverse?std::vector<size_t>{31,15,0}:std::vector<size_t>{0,15,31})){@autoreleasepool{
   auto fixture=fs::path(argv[3])/("layer-"+std::to_string(layer)),capture=fs::path(argv[4])/("layer-"+std::to_string(layer));std::array<std::vector<float>,5> expected;
   const std::array<const char*,5> names{"norm-input","gate","up","silu","down"};for(size_t i=0;i<5;i++)expected[i]=Read<float>(capture/(std::string(names[i])+".f32"),M*((i==0||i==4)?K:N));
   std::fill_n((uint32_t*)input.contents,M*K+32,GUARD);std::memcpy((float*)input.contents+16,expected[0].data(),M*K*4);
   std::vector<uint32_t> inputBefore((uint32_t*)input.contents,(uint32_t*)input.contents+M*K+32);
   std::array<id<MTLBuffer>,3> weights,scales,biases;std::array<std::vector<uint32_t>,3> hostWeights;std::array<std::vector<uint16_t>,3> hostScales,hostBiases;
   for(size_t stream=0;stream<3;stream++){size_t n=stream==2?K:N,k=stream==2?N:K;std::string prefix=stream==0?"gate":stream==1?"up":"down";hostWeights[stream]=Read<uint32_t>(fixture/(prefix+"-weight.u32"),n*k/8);hostScales[stream]=Read<uint16_t>(fixture/(prefix+"-scales.f16"),n*k/64);hostBiases[stream]=Read<uint16_t>(fixture/(prefix+"-biases.f16"),n*k/64);auto upload=[&](const auto& a){auto b=[device newBufferWithBytes:a.data() length:a.size()*sizeof(a[0]) options:MTLResourceStorageModeShared];Need(b!=nil,"Replay weight allocation failed");return b;};weights[stream]=upload(hostWeights[stream]);scales[stream]=upload(hostScales[stream]);biases[stream]=upload(hostBiases[stream]);}
   auto reset=[&](){for(size_t i=0;i<4;i++)std::fill_n((uint32_t*)outs[i].contents,M*(i==3?K:N)+32,GUARD);std::fill_n((uint16_t*)half.contents,M*N+64,HGUARD);std::fill_n((uint16_t*)compact.contents,M*K+64,HGUARD);std::fill_n((uint32_t*)counts.contents,M*N+32,GUARD);std::fill_n((uint32_t*)counts.contents+16,M*N,0u);};
   auto guards=[&](){for(size_t stream=0;stream<3;stream++)Need(!std::memcmp(weights[stream].contents,hostWeights[stream].data(),hostWeights[stream].size()*4)&&!std::memcmp(scales[stream].contents,hostScales[stream].data(),hostScales[stream].size()*2)&&!std::memcmp(biases[stream].contents,hostBiases[stream].data(),hostBiases[stream].size()*2),"Read-only Q4 weight parents changed");Need(!std::memcmp(inputBefore.data(),input.contents,inputBefore.size()*4),"Replay changed native input");for(size_t i=0;i<4;i++){size_t size=M*(i==3?K:N);for(size_t j=0;j<16;j++)Need(((uint32_t*)outs[i].contents)[j]==GUARD&&((uint32_t*)outs[i].contents)[16+size+j]==GUARD,"Replay FP32 guard overwritten");}for(size_t j=0;j<32;j++)Need(((uint16_t*)half.contents)[j]==HGUARD&&((uint16_t*)half.contents)[32+M*N+j]==HGUARD,"Replay half guard overwritten");};
   auto encodeGather=[&](id<MTLCommandBuffer> cb,id<MTLBuffer> source,id<MTLBuffer> target,size_t columns){auto e=[cb computeCommandEncoder];uint32_t shape[]={uint32_t(M),uint32_t(columns),uint32_t(columns),0};[e setComputePipelineState:gather];[e setBuffer:source offset:64 atIndex:0];[e setBuffer:target offset:64 atIndex:1];[e setBytes:ids.data() length:M*4 atIndex:2];[e setBytes:shape length:sizeof(shape) atIndex:3];[e dispatchThreads:MTLSizeMake(M*columns,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding];};
   auto projection=[&](id<MTLCommandBuffer> cb,size_t stream,id<MTLComputePipelineState> pso,bool isFused){size_t k=stream==2?N:K,n=stream==2?K:N;Projection p{M,k,n,k,n};auto e=[cb computeCommandEncoder];[e setComputePipelineState:pso];[e setBuffer:(stream==2?half:compact) offset:64 atIndex:0];[e setBuffer:weights[stream] offset:0 atIndex:1];[e setBuffer:scales[stream] offset:0 atIndex:2];[e setBuffer:biases[stream] offset:0 atIndex:3];[e setBuffer:outs[stream==2?3:stream] offset:64 atIndex:4];[e setBytes:&p length:sizeof(p) atIndex:5];[e setBytes:ids.data() length:M*4 atIndex:6];if(isFused){[e setBuffer:weights[1] offset:0 atIndex:8];[e setBuffer:scales[1] offset:0 atIndex:9];[e setBuffer:biases[1] offset:0 atIndex:10];[e setBuffer:outs[1] offset:64 atIndex:11];[e setBuffer:half offset:64 atIndex:13];[e setBuffer:counts offset:64 atIndex:14];}[e dispatchThreadgroups:MTLSizeMake(n/(isFused?32:64),2,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding];};
   auto silu=[&](id<MTLCommandBuffer> cb,id<MTLComputePipelineState> pso){Pointwise p{M,N,N,N,N,0,0};auto e=[cb computeCommandEncoder];[e setComputePipelineState:pso];[e setBuffer:outs[0] offset:64 atIndex:0];[e setBuffer:outs[1] offset:64 atIndex:1];[e setBuffer:outs[2] offset:64 atIndex:2];[e setBytes:&p length:sizeof(p) atIndex:3];[e dispatchThreads:MTLSizeMake(M*N,1,1) threadsPerThreadgroup:MTLSizeMake(pso.maxTotalThreadsPerThreadgroup,1,1)];[e endEncoding];};
   std::vector<uint16_t> goldHalf(M*N);for(size_t i=0;i<M*N;i++)goldHalf[i]=Garnet::Float32ToFloat16RNE(expected[3][i]);
   reset();auto cb=[queue commandBuffer];for(size_t stream=0;stream<2;stream++){encodeGather(cb,input,compact,K);projection(cb,stream,gatePipe,false);}silu(cb,activation);encodeGather(cb,outs[2],half,N);projection(cb,2,downPipe,false);finish(cb);guards();
   J row={{"layer",layer},{"rows",M},{"variants",J::array()},{"fp64",J::array()}};std::string prefix="layer-"+std::to_string(layer);
   for(size_t i=0;i<4;i++){size_t size=M*(i==3?K:N);size_t mismatch=Diff((float*)outs[i].contents+16,expected[i+1].data(),size,4);row["native_control_mismatches"].push_back(mismatch);Need(mismatch==0,"Default-option original MLP replay differs from complete native-captured FP32 values");controlWords+=size;Save(output/(prefix+"-control-"+names[i+1]+".u32"),outs[i].contents,outs[i].length);}
   Need(!std::memcmp((uint16_t*)half.contents+32,goldHalf.data(),M*N*2),"Native down-input cast differs from CPU RNE");Save(output/(prefix+"-control-half.u16"),half.contents,half.length);
   cb=[queue commandBuffer];silu(cb,oldActivation);encodeGather(cb,outs[2],half,N);finish(cb);guards();row["prior_explicit31_silu_word_mismatches"]=Diff((float*)outs[2].contents+16,expected[3].data(),M*N,4);row["prior_explicit31_half_word_mismatches"]=Diff((uint16_t*)half.contents+32,goldHalf.data(),M*N,2);Save(output/(prefix+"-explicit31-silu.u32"),outs[2].contents,outs[2].length);
   for(size_t stream=0;stream<3;stream++)for(size_t rr:{0ul,256ul,511ul})for(size_t col:{0ul,1ul,15ul,16ul,31ul,32ul,(stream==2?K:N)/2,(stream==2?K:N)-2,(stream==2?K:N)-1}){
    const size_t inner=stream==2?N:K;const auto& values=expected[stream==2?3:0];double sum=0,sumabs=0;
    for(size_t k=0;k<inner;k++){size_t at=col*(inner/64)+k/64;uint32_t code=(hostWeights[stream][col*(inner/8)+k/8]>>((k%8)*4))&15u;float scale=Garnet::Float16ToFloat32(hostScales[stream][at]),bias=Garnet::Float16ToFloat32(hostBiases[stream][at]);float w=Garnet::Float16ToFloat32(Garnet::Float32ToFloat16RNE(std::fma(float(code),scale,bias))),x=Garnet::Float16ToFloat32(Garnet::Float32ToFloat16RNE(values[rr*inner+k]));double product=double(w)*double(x);sum+=product;sumabs+=std::abs(product);}
    float actual=expected[stream==2?4:stream+1][rr*(stream==2?K:N)+col];double fraction=std::abs(actual-sum)/(2e-6+4e-6*sumabs);Need(std::isfinite(actual)&&fraction<=1,"Actual native MLP product fails fixed FP64 gate");row["fp64"].push_back({{"stream",stream},{"row",rr},{"column",col},{"sum",sum},{"sum_abs",sumabs},{"actual",actual},{"fraction",fraction}});
   }
   for(size_t method=0;method<4;method++)for(size_t v=0;v<3;v++){
    reset();cb=[queue commandBuffer];encodeGather(cb,input,compact,K);projection(cb,0,fused[method>=2][v],true);projection(cb,2,(method==1||method==2)?downTensor:downPipe,false);finish(cb);guards();J item={{"method",method},{"variant",v},{"guards_exact",true}};
    if(v==0){item["gate_mismatches"]=Diff((float*)outs[0].contents+16,expected[1].data(),M*N,4);item["up_mismatches"]=Diff((float*)outs[1].contents+16,expected[2].data(),M*N,4);for(size_t i=0;i<2;i++)Save(output/(prefix+"-method-"+std::to_string(method)+"-fused-shadow-"+(i?"up":"gate")+".u32"),outs[i].contents,outs[i].length);}
    else for(size_t stream=0;stream<2;stream++)for(size_t i=0;i<M*N+32;i++)Need(((uint32_t*)outs[stream].contents)[i]==GUARD,"Normal fused kernel materialized intermediate");
    item["half_mismatches"]=Diff((uint16_t*)half.contents+32,goldHalf.data(),M*N,2);item["down_mismatches"]=Diff((float*)outs[3].contents+16,expected[4].data(),M*K,4);halfWords+=M*N;downWords+=M*K;
    auto map=(uint32_t*)counts.contents;for(size_t i=0;i<16;i++)Need(map[i]==GUARD&&map[16+M*N+i]==GUARD,"Writer guard overwritten");for(size_t i=0;i<M*N;i++)Need(map[16+i]==(v==2?1u:0u),"Wrong full trained writer map");item["writers_exact"]=true;
    std::string name=prefix+"-method-"+std::to_string(method)+"-fused-"+std::to_string(v);Save(output/(name+"-half.u16"),half.contents,half.length);Save(output/(name+"-down.u32"),outs[3].contents,outs[3].length);if(v==2)Save(output/(name+"-writers.u32"),counts.contents,counts.length);
    bool exact=item["half_mismatches"]==0&&item["down_mismatches"]==0&&(v!=0||(item["gate_mismatches"]==0&&item["up_mismatches"]==0));candidateExact=candidateExact&&exact;row["variants"].push_back(item);
   }
   auto encodeNormal=[&](size_t method){auto command=[queue commandBuffer];encodeGather(command,input,compact,K);projection(command,0,fused[method>=2][1],true);projection(command,2,(method==1||method==2)?downTensor:downPipe,false);finish(command);return command;};
   auto validateNormal=[&](size_t method,const char* phase){reset();encodeNormal(method);guards();auto halfMismatch=Diff((uint16_t*)half.contents+32,goldHalf.data(),M*N,2),downMismatch=Diff((float*)outs[3].contents+16,expected[4].data(),M*K,4);Need(halfMismatch==0&&downMismatch==0,"Timed normal MLP differs from trained capture");for(size_t stream=0;stream<3;stream++)for(size_t i=0;i<M*N+32;i++)Need(((uint32_t*)outs[stream].contents)[i]==GUARD,"Timed normal MLP materialized intermediate");result["timing_witnesses"].push_back({{"layer",layer},{"method",method},{"phase",phase},{"half_mismatches",halfMismatch},{"down_mismatches",downMismatch},{"all_input_and_weight_parents_unchanged",true},{"guards_exact",true},{"no_intermediates_materialized",true}});};
   for(size_t method=0;method<4;method++)validateNormal(method,"pre");
   for(int round=-4;round<12;round++)for(int position=0;position<4;position++){
    int method=(position+round+4)%4;if(reverse)method=3-method;
    auto began=std::chrono::steady_clock::now();auto command=encodeNormal(size_t(method));double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-began).count();double gpu=(command.GPUEndTime-command.GPUStartTime)*1000.0;Need(std::isfinite(gpu)&&gpu>0&&std::isfinite(wall)&&wall>0,"Missing MLP timer");
    result["samples"].push_back({{"layer",layer},{"method",method},{"round",round},{"position",position},{"warmup",round<0},{"gpu_start",command.GPUStartTime},{"gpu_end",command.GPUEndTime},{"gpu_ms",gpu},{"wall_ms",wall}});
   }
   for(size_t method=0;method<4;method++)validateNormal(method,"post");
   result["layers"].push_back(row);Report(output/"result1.json",result);std::cout<<J{{"layer",layer},{"actual_native_controls_exact",true},{"fused_variants",row["variants"]}}.dump()<<'\n'<<std::flush;
  }}
  result.update({{"completed",true},{"candidate_exact",candidateExact},{"original_native_fp32_words",controlWords},{"fused_half_word_comparisons",halfWords},{"fused_down_word_comparisons",downWords},{"original_main_options","nil/default, matching native compiled profile"},{"trained_activations",true},{"timing_measured",true},{"native_integration",false},{"serving_speed_claim",false},{"quality_qualified",false},{"goal_complete",false}});Report(output/"result1.json",result);return 0;
 }catch(const std::exception& e){result["error"]=e.what();if(!output.empty()&&fs::exists(output))Report(output/"result1.json",result);std::cerr<<e.what()<<'\n';return 1;}}
}
