// SPDX-License-Identifier: Apache-2.0
// Isolated fused gate/up + SiLU output screen. No Garnet runtime mutation.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "nlohmann/json.hpp"
#include "src/model/float16_codec.h"
#include "src/backends/metal/packed_affine_layout_guard.h"
namespace fs=std::filesystem;
using J=nlohmann::json;
void Need(bool v,const char* message){if(!v)throw std::runtime_error(message);}
template<class T>std::vector<T> Read(fs::path p,size_t count){std::ifstream f(p,std::ios::binary|std::ios::ate);Need(bool(f)&&size_t(f.tellg())==count*sizeof(T),"Wrong input size");f.seekg(0);std::vector<T> v(count);f.read((char*)v.data(),count*sizeof(T));Need(bool(f),"Cannot read input");return v;}
void Save(fs::path p,const void* data,size_t bytes){std::ofstream f(p,std::ios::binary);f.write((const char*)data,bytes);Need(bool(f),"Cannot save output");}
void Report(fs::path p,const J& v){std::ofstream f(p);f<<v.dump(2)<<'\n';Need(bool(f),"Cannot save report");}
size_t Differences(const void* a,const void* b,size_t count,size_t size){size_t changed=0;auto x=(const unsigned char*)a,y=(const unsigned char*)b;for(size_t i=0;i<count;i++)changed+=std::memcmp(x+i*size,y+i*size,size)!=0;return changed;}
struct Projection {uint64_t rows,inner,columns,aStride,oStride;};
struct Pointwise {uint64_t rows,columns,aStride,bStride,cStride;float epsilon,offset;};
int main(int argc,char**argv){fs::path output;J report={{"completed",false},{"cases",J::array()}};
 @autoreleasepool{try{
  Need(argc==5||argc==8,"COMBINED ORDINARY ASSETS FRESH_OUTPUT [timing forward|reverse FIXTURE] required");bool timed=argc==8;std::string ordering=timed?argv[6]:"forward",selected=timed?argv[7]:"";Need(!timed||(std::string(argv[5])=="timing"&&(ordering=="forward"||ordering=="reverse")&&(selected=="synthetic"||selected=="layer-0")),"Unknown timing order/fixture");output=fs::absolute(argv[4]);Need(fs::create_directory(output),"Fresh output required");
  auto device=MTLCreateSystemDefaultDevice();Need(device&&[device supportsFamily:MTLGPUFamilyApple10],"Apple10/M5 required");auto queue=[device newCommandQueue];Need(queue!=nil,"Queue unavailable");NSError* error=nil;
  auto compile=[&](const char* path,NSUInteger version,MTLMathMode math,bool setMath){
   auto text=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:path] encoding:NSUTF8StringEncoding error:&error];Need(text!=nil,"Shader read failed");auto options=[MTLCompileOptions new];options.languageVersion=(MTLLanguageVersion)version;if(setMath)options.mathMode=math;
   auto lib=[device newLibraryWithSource:text options:options error:&error];if(!lib)throw std::runtime_error(error.localizedDescription.UTF8String);return lib;
  };
  auto safe=compile(argv[1],MTLLanguageVersion4_0,MTLMathModeSafe,true);
  auto fast=compile(argv[1],MTLLanguageVersion4_0,MTLMathModeFast,true);
  auto ordinary=compile(argv[2],MTLLanguageVersion3_1,MTLMathModeFast,false);
  auto pipe=[&](id<MTLLibrary> library,NSString* name,MTLFunctionConstantValues* constants=nil){auto f=constants?[library newFunctionWithName:name constantValues:constants error:&error]:[library newFunctionWithName:name];Need(f!=nil,"Function unavailable");auto p=[device newComputePipelineStateWithFunction:f error:&error];if(!p)throw std::runtime_error(error.localizedDescription.UTF8String);Need(p.threadExecutionWidth==32,"Unexpected SIMD width");return p;};
  auto finish=[&](id<MTLCommandBuffer> cb){[cb commit];[cb waitUntilCompleted];Need(cb.status==MTLCommandBufferStatusCompleted,"GPU command failed");};
  auto alloc=[&](size_t bytes){auto b=[device newBufferWithLength:bytes options:MTLResourceStorageModeShared];Need(b!=nil,"Buffer allocation failed");return b;};
  for(auto lib:{safe,fast}){
   auto p=pipe(lib,@"packed_affine_layout_guard");auto b=alloc(256);std::fill_n((uint32_t*)b.contents,64,0x1937abefu);auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];[e setComputePipelineState:p];[e setBuffer:b offset:64 atIndex:0];[e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[e endEncoding];finish(cb);
   Need(Garnet::PackedAffineLayoutMatches((uint32_t*)b.contents+16,32),"Cooperative ABI mismatch");for(size_t i=0;i<16;i++)Need(((uint32_t*)b.contents)[i]==0x1937abefu&&((uint32_t*)b.contents)[48+i]==0x1937abefu,"ABI probe guard overwritten");
  }
  constexpr size_t N=14336,K=4096,PHYSICAL=514,AS=4104,OS=14344;
  constexpr uint32_t GUARD=0x1937abef;
  constexpr uint16_t HALF_GUARD=0x35a5;
  size_t matrixWords=PHYSICAL*OS,halfWords=512*N,compactWords=512*K;
  auto a=alloc((PHYSICAL*AS+32)*4),compact=alloc((compactWords+64)*2),half=alloc((halfWords+64)*2),writers=alloc((halfWords+32)*4);
  std::array<id<MTLBuffer>,2> weights,scales,biases,out;
  auto silu=alloc((matrixWords+32)*4);for(auto& b:out)b=alloc((matrixWords+32)*4);
  std::fill_n((uint32_t*)a.contents,PHYSICAL*AS+32,GUARD);
  auto input=Read<float>(fs::path(argv[3])/"input-parent.f32",PHYSICAL*K);
  for(size_t row=0;row<PHYSICAL;row++)std::copy_n(input.data()+row*K,K,(float*)a.contents+16+row*AS);
  std::vector<uint32_t> originalInput((uint32_t*)a.contents,(uint32_t*)a.contents+PHYSICAL*AS+32);
  auto gather=pipe(safe,@"packed_affine_gather");auto activation=pipe(ordinary,@"silu_mul");auto activation32=pipe(ordinary,@"silu_profile32");
  std::array<std::array<id<MTLComputePipelineState>,2>,2> controls;
  std::array<std::array<std::array<id<MTLComputePipelineState>,3>,2>,2> candidates;
  for(size_t tileIndex=0;tileIndex<2;tileIndex++){
   auto values=[MTLFunctionConstantValues new];uint32_t dimensions[]={K,N,AS,OS};for(NSUInteger i=0;i<4;i++)[values setConstantValue:dimensions+i type:MTLDataTypeUInt atIndex:10+i];
   auto name=[NSString stringWithFormat:@"half_m%zuk128",tileIndex?256ul:128ul];controls[tileIndex][0]=pipe(safe,name,values);controls[tileIndex][1]=pipe(safe,[NSString stringWithFormat:@"pair_m%zuk128",tileIndex?256ul:128ul],values);
   for(size_t math=0;math<2;math++)for(size_t variant=0;variant<3;variant++){
    bool shadow=variant==0,count=variant==2;[values setConstantValue:&shadow type:MTLDataTypeBool atIndex:20];[values setConstantValue:&count type:MTLDataTypeBool atIndex:21];
    auto p=pipe(math?fast:safe,[NSString stringWithFormat:@"fused_m%zuk128",tileIndex?256ul:128ul],values);Need(p.maxTotalThreadsPerThreadgroup>=256&&p.staticThreadgroupMemoryLength==17408,"Fused resource envelope differs");candidates[math][tileIndex][variant]=p;
   }
  }
  report.update({{"device",device.name.UTF8String},{"rows",{127,128,129,255,256,257,383,511,512}},{"normal_atomics",false},{"timing_admitted",timed},{"native_integration",false},{"quality_qualified",false}});
  auto saveMatrix=[&](std::string name,id<MTLBuffer> b){Save(output/name,b.contents,b.length);};
  auto reset=[&](){
   for(auto b:{out[0],out[1],silu}){std::fill_n((uint32_t*)b.contents,matrixWords+32,GUARD);std::fill_n((uint32_t*)b.contents+16,matrixWords,0u);}
   std::fill_n((uint16_t*)half.contents,halfWords+64,HALF_GUARD);std::fill_n((uint16_t*)compact.contents,compactWords+64,HALF_GUARD);
   std::fill_n((uint32_t*)writers.contents,halfWords+32,GUARD);std::fill_n((uint32_t*)writers.contents+16,halfWords,0u);
  };
  size_t allHalfWords=0,allF32Words=0,allWriterWords=0,fp64Count=0;double maximumFraction=0;
  for(const std::string label:{"synthetic","layer-0","layer-15","layer-31"}){@autoreleasepool{
   if(timed&&label!=selected)continue;
   std::array<std::vector<uint32_t>,2> hostWeights;std::array<std::vector<uint16_t>,2> hostScales,hostBiases;
   for(size_t stream=0;stream<2;stream++){
    std::string prefix=stream?"up":"gate";auto folder=fs::path(argv[3])/label;
    hostWeights[stream]=Read<uint32_t>(folder/(prefix+"-weight.u32"),N*K/8);hostScales[stream]=Read<uint16_t>(folder/(prefix+"-scales.f16"),N*K/64);hostBiases[stream]=Read<uint16_t>(folder/(prefix+"-biases.f16"),N*K/64);
    auto upload=[&](const auto& v){auto b=[device newBufferWithBytes:v.data() length:v.size()*sizeof(v[0]) options:MTLResourceStorageModeShared];Need(b!=nil,"Weight upload failed");return b;};weights[stream]=upload(hostWeights[stream]);scales[stream]=upload(hostScales[stream]);biases[stream]=upload(hostBiases[stream]);
   }
   for(size_t rows:{127ul,128ul,129ul,255ul,256ul,257ul,383ul,511ul,512ul}){@autoreleasepool{
    if(timed&&rows!=128&&rows!=256&&rows!=512)continue;
    std::vector<uint32_t> ids(rows);for(size_t i=0;i<rows;i++)ids[i]=uint32_t(rows-i-(timed?1:0));
    size_t tileIndex=rows>=256;Projection p{rows,K,N,AS,OS};std::string cell=label+"-m"+std::to_string(rows);J row={{"label",label},{"rows",rows},{"tile",tileIndex?256:128},{"variants",J::array()},{"fp64",J::array()}};
    auto encodeGather=[&](id<MTLCommandBuffer> cb,id<MTLBuffer> source,size_t offset,id<MTLBuffer> target,size_t columns,size_t stride){
     if(timed&&source==a)offset+=AS*4;auto e=[cb computeCommandEncoder];uint32_t shape[]={uint32_t(rows),uint32_t(columns),uint32_t(stride),0};[e setComputePipelineState:gather];[e setBuffer:source offset:offset atIndex:0];[e setBuffer:target offset:64 atIndex:1];[e setBytes:ids.data() length:ids.size()*4 atIndex:2];[e setBytes:shape length:sizeof(shape) atIndex:3];[e dispatchThreads:MTLSizeMake(rows*columns,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding];
    };
    auto projection=[&](id<MTLCommandBuffer> cb,size_t stream,id<MTLComputePipelineState> pipeline,bool fused,size_t overrideY=0){
     auto e=[cb computeCommandEncoder];[e setComputePipelineState:pipeline];[e setBuffer:compact offset:64 atIndex:0];[e setBuffer:weights[stream] offset:0 atIndex:1];[e setBuffer:scales[stream] offset:0 atIndex:2];[e setBuffer:biases[stream] offset:0 atIndex:3];[e setBuffer:out[stream] offset:64 atIndex:4];[e setBytes:&p length:sizeof(p) atIndex:5];[e setBytes:ids.data() length:ids.size()*4 atIndex:6];
     if(fused){[e setBuffer:weights[1] offset:0 atIndex:8];[e setBuffer:scales[1] offset:0 atIndex:9];[e setBuffer:biases[1] offset:0 atIndex:10];[e setBuffer:out[1] offset:64 atIndex:11];[e setBuffer:half offset:64 atIndex:13];[e setBuffer:writers offset:64 atIndex:14];}
     size_t tile=tileIndex?256:128;[e dispatchThreadgroups:MTLSizeMake(N/(fused?32:64),overrideY?overrideY:(rows+tile-1)/tile,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding];
    };
    auto verifyGuards=[&](){
     Need(!std::memcmp(originalInput.data(),a.contents,originalInput.size()*4),"Input mutated");
     for(auto b:{out[0],out[1],silu})for(size_t i=0;i<16;i++)Need(((uint32_t*)b.contents)[i]==GUARD&&((uint32_t*)b.contents)[matrixWords+16+i]==GUARD,"FP32 guard overwritten");
     auto h=(uint16_t*)half.contents;for(size_t i=0;i<32;i++)Need(h[i]==HALF_GUARD&&h[halfWords+32+i]==HALF_GUARD,"Half guard overwritten");for(size_t i=rows*N;i<halfWords;i++)Need(h[32+i]==HALF_GUARD,"Inactive half row overwritten");
     auto ah=(uint16_t*)compact.contents;for(size_t i=0;i<32;i++)Need(ah[i]==HALF_GUARD&&ah[compactWords+32+i]==HALF_GUARD,"Compact guard overwritten");for(size_t i=rows*K;i<compactWords;i++)Need(ah[32+i]==HALF_GUARD,"Inactive compact row overwritten");
     for(size_t i=0;i<rows;i++)for(size_t j=0;j<K;j++)Need(ah[32+i*K+j]==Garnet::Float32ToFloat16RNE(input[size_t(ids[i]+(timed?1:0))*K+j]),"Half input gather mismatch");
    };

    if(timed){
     auto gold=Read<uint16_t>(output.parent_path()/"correctness1"/(cell+"-control-half.u16"),halfWords+64);
     std::array<std::vector<double>,5> gpu,wall;
     std::array<std::string,5> labels{"separate_ordinary","paired_ordinary","separate_profile32","paired_profile32","fused_fast_half"};
     for(int repetition=-3;repetition<9;repetition++)for(size_t pos=0;pos<5;pos++){
      size_t v=(pos+size_t(repetition+3))%5;if(ordering=="reverse")v=4-v;
      reset();auto begin=std::chrono::steady_clock::now();auto command=[queue commandBuffer];
      if(v<4){
       if(v%2){encodeGather(command,a,64,compact,K,AS);projection(command,0,controls[tileIndex][1],true);}
       else for(size_t stream=0;stream<2;stream++){encodeGather(command,a,64,compact,K,AS);projection(command,stream,controls[tileIndex][0],false);}
       Pointwise q{rows,N,OS,OS,OS,0,0};auto activationEncoder=[command computeCommandEncoder];[activationEncoder setComputePipelineState:(v>=2?activation32:activation)];[activationEncoder setBuffer:out[0] offset:64 atIndex:0];[activationEncoder setBuffer:out[1] offset:64 atIndex:1];[activationEncoder setBuffer:silu offset:64 atIndex:2];[activationEncoder setBytes:&q length:sizeof(q) atIndex:3];[activationEncoder dispatchThreads:MTLSizeMake(PHYSICAL*N,1,1) threadsPerThreadgroup:MTLSizeMake(v>=2?128:activation.maxTotalThreadsPerThreadgroup,1,1)];[activationEncoder endEncoding];encodeGather(command,silu,64,half,N,OS);
      }else{encodeGather(command,a,64,compact,K,AS);projection(command,0,candidates[1][tileIndex][1],true);}
      finish(command);double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();double gpuMs=(command.GPUEndTime-command.GPUStartTime)*1000;
      Need(elapsed>0&&gpuMs>0&&gpuMs<=elapsed,"Invalid timer interval");Need(!std::memcmp(gold.data(),half.contents,(halfWords+64)*2),"Timed output changed from frozen correctness control");verifyGuards();
      if(repetition>=0){gpu[v].push_back(gpuMs);wall[v].push_back(elapsed);}
     }
     for(size_t v=0;v<5;v++){Need(gpu[v].size()==9&&wall[v].size()==9,"Missing timed samples");row["variants"].push_back({{"label",labels[v]},{"gpu_ms",gpu[v]},{"wall_ms",wall[v]},{"full_half_output_exact",true},{"threads_silu",v==4?0:v>=2?128:activation.maxTotalThreadsPerThreadgroup}});}
     report["cases"].push_back(row);Report(output/"result1.json",report);std::cout<<J{{"cell",cell},{"timed",true}}.dump()<<'\n'<<std::flush;continue;
    }

    reset();auto cb=[queue commandBuffer];
    for(size_t stream=0;stream<2;stream++){encodeGather(cb,a,64,compact,K,AS);projection(cb,stream,controls[tileIndex][0],false);}
    Pointwise q{PHYSICAL,N,OS,OS,OS,0,0};auto e=[cb computeCommandEncoder];[e setComputePipelineState:activation];[e setBuffer:out[0] offset:64 atIndex:0];[e setBuffer:out[1] offset:64 atIndex:1];[e setBuffer:silu offset:64 atIndex:2];[e setBytes:&q length:sizeof(q) atIndex:3];[e dispatchThreads:MTLSizeMake(PHYSICAL*N,1,1) threadsPerThreadgroup:MTLSizeMake(activation.maxTotalThreadsPerThreadgroup,1,1)];[e endEncoding];encodeGather(cb,silu,64,half,N,OS);finish(cb);verifyGuards();
    std::array<std::vector<uint32_t>,2> gold;for(size_t stream=0;stream<2;stream++){gold[stream].assign((uint32_t*)out[stream].contents,(uint32_t*)out[stream].contents+matrixWords+32);saveMatrix(cell+"-control-"+(stream?"up":"gate")+".u32",out[stream]);}
    std::vector<uint16_t> goldHalf((uint16_t*)half.contents,(uint16_t*)half.contents+halfWords+64);saveMatrix(cell+"-control-half.u16",half);
    for(size_t stream=0;stream<2;stream++)for(size_t rr:{size_t(0),rows/2,rows-1})for(size_t col:{size_t(0),size_t(31),size_t(32),N/2,N-1}){
     double sum=0,sumabs=0;for(size_t k=0;k<K;k++){
      size_t at=col*(K/64)+k/64;uint32_t code=(hostWeights[stream][col*(K/8)+k/8]>>((k%8)*4))&15u;
      float scale=Garnet::Float16ToFloat32(hostScales[stream][at]),bias=Garnet::Float16ToFloat32(hostBiases[stream][at]);float w=Garnet::Float16ToFloat32(Garnet::Float32ToFloat16RNE(std::fma(float(code),scale,bias)));float x=Garnet::Float16ToFloat32(Garnet::Float32ToFloat16RNE(input[size_t(ids[rr])*K+k]));double term=double(x)*double(w);sum+=term;sumabs+=std::abs(term);
     }
     float actual;uint32_t word=gold[stream][16+size_t(ids[rr])*OS+col];std::memcpy(&actual,&word,4);double allowance=2e-6+4e-6*sumabs,fraction=std::abs(actual-sum)/allowance;Need(std::isfinite(actual)&&fraction<=1,"Original projection exceeds fixed FP64 gate");maximumFraction=std::max(maximumFraction,fraction);fp64Count++;row["fp64"].push_back({{"stream",stream},{"row",rr},{"column",col},{"sum",sum},{"sum_abs",sumabs},{"actual",actual},{"fraction",fraction}});
    }
    for(size_t math=0;math<2;math++)for(size_t variant=0;variant<3;variant++){
     reset();cb=[queue commandBuffer];encodeGather(cb,a,64,compact,K,AS);projection(cb,0,candidates[math][tileIndex][variant],true);finish(cb);verifyGuards();
     size_t gateMismatch=0,upMismatch=0;
     if(variant==0){gateMismatch=Differences(gold[0].data(),out[0].contents,matrixWords+32,4);upMismatch=Differences(gold[1].data(),out[1].contents,matrixWords+32,4);allF32Words+=2*(matrixWords+32);}
     else for(auto b:out)for(size_t i=0;i<matrixWords;i++)Need(((uint32_t*)b.contents)[16+i]==0,"Unobserved FP32 intermediates materialized");
     size_t mismatch=Differences(goldHalf.data(),half.contents,halfWords+64,2);allHalfWords+=halfWords+64;
     auto counts=(uint32_t*)writers.contents;for(size_t i=0;i<16;i++)Need(counts[i]==GUARD&&counts[16+halfWords+i]==GUARD,"Writer guard overwritten");for(size_t i=0;i<halfWords;i++)Need(counts[16+i]==(variant==2&&i<rows*N?1u:0u),"Duplicate/missing half writer");allWriterWords+=halfWords+32;
     std::string name=cell+"-"+(math?"fast":"safe")+"-"+(variant==0?"shadow":variant==1?"normal":"writer");saveMatrix(name+"-half.u16",half);
     if(variant==0){saveMatrix(name+"-gate.u32",out[0]);saveMatrix(name+"-up.u32",out[1]);}if(variant==2)saveMatrix(name+"-counts.u32",writers);
     row["variants"].push_back({{"math",math?"Fast":"Safe"},{"variant",variant},{"gate_mismatches",gateMismatch},{"up_mismatches",upMismatch},{"half_mismatches",mismatch},{"writers_exact",true},{"guards_exact",true},{"file",name+"-half.u16"}});
    }
    if(rows==512){reset();cb=[queue commandBuffer];encodeGather(cb,a,64,compact,K,AS);projection(cb,0,candidates[0][tileIndex][2],true,1);finish(cb);size_t missing=0;auto counts=(uint32_t*)writers.contents;for(size_t i=0;i<rows*N;i++)missing+=counts[16+i]==0;Need(missing==256*N,"Truncated grid fault not detected");row["truncated_grid_missing"]=missing;saveMatrix(cell+"-fault-counts.u32",writers);}
    report["cases"].push_back(row);Report(output/"result1.json",report);std::cout<<J{{"cell",cell},{"checked",true},{"half_mismatches_safe",row["variants"][1]["half_mismatches"]},{"half_mismatches_fast",row["variants"][4]["half_mismatches"]}}.dump()<<'\n'<<std::flush;
   }}
  }}
  report.update({{"completed",true},{"timing_mode",timed},{"ordering",ordering},{"timing_fixture",selected},{"compared_half_words",allHalfWords},{"compared_fp32_words",allF32Words},{"checked_writer_words",allWriterWords},{"fp64_values",fp64Count},{"maximum_fp64_fraction",maximumFraction},{"serving_speed_claim",false}});Report(output/"result1.json",report);return 0;
 }catch(const std::exception& error){report["error"]=error.what();if(!output.empty()&&fs::exists(output))Report(output/"result1.json",report);std::cerr<<error.what()<<'\n';return 1;}}
}
