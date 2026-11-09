// SPDX-License-Identifier: Apache-2.0
// Trained-input pilot for device-tensor operands, outside the native runtime.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <CommonCrypto/CommonDigest.h>
#include "nlohmann/json.hpp"
#include "src/model/float16_codec.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>
using J=nlohmann::json;namespace fs=std::filesystem;
void Need(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
template<class T>std::vector<T> Read(fs::path p,size_t count){std::ifstream f(p,std::ios::binary|std::ios::ate);Need(bool(f)&&size_t(f.tellg())==count*sizeof(T),"Fixture extent mismatch");f.seekg(0);std::vector<T> v(count);f.read((char*)v.data(),count*sizeof(T));Need(bool(f),"Fixture read failed");return v;}
std::string Sha(const void* data,size_t bytes){Need(bytes<=UINT32_MAX,"Hash extent overflow");unsigned char d[CC_SHA256_DIGEST_LENGTH];CC_SHA256(data,CC_LONG(bytes),d);std::ostringstream s;for(auto c:d)s<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(c);return s.str();}
void Save(fs::path p,const void* data,size_t bytes){std::ofstream f(p,std::ios::binary);f.write((const char*)data,bytes);Need(bool(f),"Payload write failed");}
void Report(fs::path p,const J& j){auto t=p;t+=".tmp";std::ofstream f(t);f<<j.dump(2)<<'\n';f.close();fs::rename(t,p);}
struct Params{uint64_t rows,inner,columns,aStride,oStride;};
int main(int argc,char** argv){fs::path dest;J report={{"completed",false},{"cases",J::array()},{"pipeline_resources",J::array()},{"prepared",J::array()},{"samples",J::array()}};
try{@autoreleasepool{
 Need(argc==6,"SHADER ASSETS CAPTURES FRESH_OUTPUT forward|reverse");bool reverse=std::string(argv[5])=="reverse";Need(reverse||std::string(argv[5])=="forward","Invalid method order");dest=argv[4];Need(!fs::exists(dest),"Fresh destination required");fs::create_directory(dest);
 auto device=MTLCreateSystemDefaultDevice();Need(device&&[device supportsFamily:MTLGPUFamilyApple10],"Apple10 profile required");auto queue=[device newCommandQueue];Need(queue!=nil,"Missing Metal queue");
 NSError* error=nil;auto text=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]] encoding:NSUTF8StringEncoding error:&error];Need(text!=nil,"Missing shader source");
 auto options=[MTLCompileOptions new];options.languageVersion=MTLLanguageVersion4_0;options.mathMode=MTLMathModeSafe;
 auto library=[device newLibraryWithSource:text options:options error:&error];if(!library)throw std::runtime_error(error.localizedDescription.UTF8String);
 auto pipe=[&](NSString* name,MTLFunctionConstantValues* c=nil){auto f=c?[library newFunctionWithName:name constantValues:c error:&error]:[library newFunctionWithName:name];if(!f)throw std::runtime_error(error.localizedDescription.UTF8String);auto p=[device newComputePipelineStateWithFunction:f error:&error];if(!p)throw std::runtime_error(error.localizedDescription.UTF8String);Need(p.threadExecutionWidth==32&&p.maxTotalThreadsPerThreadgroup>=128,"Pipeline geometry mismatch");return p;};
 auto buffer=[&](size_t bytes){Need(device.currentAllocatedSize+bytes+256*1024*1024<device.recommendedMaxWorkingSetSize,"Operator exceeds device allocation profile");auto b=[device newBufferWithLength:bytes options:MTLResourceStorageModeShared];Need(b!=nil,"Allocation failed");return b;};
 auto finish=[&](id<MTLCommandBuffer> cb){[cb commit];[cb waitUntilCompleted];if(cb.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(cb.error.localizedDescription.UTF8String);};
 auto gather=pipe(@"packed_affine_gather");bool no=false;
 constexpr size_t M=512;constexpr uint32_t G=0x1937abef;constexpr uint16_t H=0x35a5;
 const std::array<std::string,6> names{{"inline","tensor256","row512n64","row512n32","row256n64t128","row256n32t128"}};
 const std::array<size_t,6> tileM{{256,256,512,512,256,256}},tileN{{64,64,64,32,64,32}},threads{{256,256,256,256,128,128}};
 for(bool down:(reverse?std::vector<bool>{true,false}:std::vector<bool>{false,true})){@autoreleasepool{
  const size_t k=down?14336:4096,n=down?4096:14336;std::string shape=down?"down":"gate";auto assets=fs::path(argv[2])/"layer-15",capture=fs::path(argv[3])/"layer-15";
  auto input=Read<float>(capture/(down?"silu.f32":"norm-input.f32"),M*k),gold=Read<float>(capture/(down?"down.f32":"gate.f32"),M*n);
  auto weights=Read<uint32_t>(assets/(shape+"-weight.u32"),n*k/8);auto scales=Read<uint16_t>(assets/(shape+"-scales.f16"),n*k/64),biases=Read<uint16_t>(assets/(shape+"-biases.f16"),n*k/64);
  auto upload=[&](const auto& data){auto b=buffer(data.size()*sizeof(data[0]));std::memcpy(b.contents,data.data(),data.size()*sizeof(data[0]));return b;};
  auto w=upload(weights),sc=upload(scales),bi=upload(biases),x=buffer((M*k+32)*4),compact=buffer((M*k+64)*2),out=buffer((M*n+32)*4);
  std::fill_n((uint32_t*)x.contents,M*k+32,G);std::memcpy((float*)x.contents+16,input.data(),M*k*4);auto inputSha=Sha(x.contents,x.length);
  const auto wSha=Sha(w.contents,w.length),sSha=Sha(sc.contents,sc.length),bSha=Sha(bi.contents,bi.length);
  std::vector<uint16_t> expectedCompact(M*k+64,H);for(size_t at=0;at<M*k;++at)expectedCompact[32+at]=Garnet::Float32ToFloat16RNE(input[at]);
  auto constants=[MTLFunctionConstantValues new];uint32_t dims[]={uint32_t(k),uint32_t(n),uint32_t(k),uint32_t(n)};for(NSUInteger i=0;i<4;++i)[constants setConstantValue:dims+i type:MTLDataTypeUInt atIndex:10+i];[constants setConstantValue:&no type:MTLDataTypeBool atIndex:22];
  std::array<id<MTLComputePipelineState>,6> pipelines{{pipe(@"half_m256k128",constants),pipe(@"shared_tensor_m256",constants),pipe(@"row512n64",constants),pipe(@"row512n32",constants),pipe(@"row256n64t128",constants),pipe(@"row256n32t128",constants)}};
  for(size_t mode=0;mode<6;++mode){auto p=pipelines[mode];Need(p.maxTotalThreadsPerThreadgroup>=threads[mode],"Insufficient prepared threads");Need(p.staticThreadgroupMemoryLength==tileN[mode]*136*2,"Unexpected static shared bytes");report["pipeline_resources"].push_back({{"shape",shape},{"mode",names[mode]},{"threads",threads[mode]},{"maximum_threads",p.maxTotalThreadsPerThreadgroup},{"shared_bytes",p.staticThreadgroupMemoryLength},{"simd_width",p.threadExecutionWidth}});}
  std::vector<uint32_t> indices(M);std::iota(indices.begin(),indices.end(),0);Params params{M,k,n,k,n};
  auto encode=[&](size_t mode){
   auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];uint32_t gs[]={uint32_t(M),uint32_t(k),uint32_t(k),0};[e setComputePipelineState:gather];[e setBuffer:x offset:64 atIndex:0];[e setBuffer:compact offset:64 atIndex:1];[e setBytes:indices.data() length:indices.size()*4 atIndex:2];[e setBytes:gs length:sizeof(gs) atIndex:3];[e dispatchThreads:MTLSizeMake(M*k,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding];
   e=[cb computeCommandEncoder];[e setComputePipelineState:pipelines[mode]];[e setBuffer:compact offset:64 atIndex:0];
   [e setBuffer:w offset:0 atIndex:1];[e setBuffer:sc offset:0 atIndex:2];[e setBuffer:bi offset:0 atIndex:3];[e setBuffer:out offset:64 atIndex:4];[e setBytes:&params length:sizeof(params) atIndex:5];[e setBytes:indices.data() length:indices.size()*4 atIndex:6];
   [e dispatchThreadgroups:MTLSizeMake(n/tileN[mode],M/tileM[mode],1) threadsPerThreadgroup:MTLSizeMake(threads[mode],1,1)];[e endEncoding];
   finish(cb);return cb;
  };
  auto validate=[&](size_t mode,const char* phase){
   auto cb=encode(mode);
   size_t exact=0;double maximum=0;auto actual=(float*)out.contents+16;
   for(size_t i=0;i<M*n;++i){Need(std::isfinite(actual[i]),"Nonfinite candidate output");exact+=std::memcmp(actual+i,gold.data()+i,4)==0;maximum=std::max(maximum,std::abs(double(actual[i])-gold[i]));}
   for(size_t i=0;i<16;++i){Need(((uint32_t*)out.contents)[i]==G&&((uint32_t*)out.contents)[16+M*n+i]==G,"Output parent guard changed");}
   for(size_t i=0;i<32;++i)Need(((uint16_t*)compact.contents)[i]==H&&((uint16_t*)compact.contents)[32+M*k+i]==H,"Input gather guard changed");Need(Sha(x.contents,x.length)==inputSha,"Read-only input changed");Need(!std::memcmp(compact.contents,expectedCompact.data(),compact.length),"Compact input changed");Need(Sha(w.contents,w.length)==wSha&&Sha(sc.contents,sc.length)==sSha&&Sha(bi.contents,bi.length)==bSha,"Read-only quantized weights changed");
   Need(exact==M*n,"Candidate differs from unchanged native trained capture");

   report["cases"].push_back({{"shape",shape},{"phase",phase},{"mode",names[mode]},{"exact_words",exact},{"total_words",M*n},{"maximum_absolute_difference",maximum},{"parent_sha256",Sha(out.contents,out.length)},{"all_parents_unchanged",true}});
  };
  std::fill_n((uint32_t*)out.contents,M*n+32,G);std::fill_n((uint16_t*)compact.contents,M*k+64,H);
  for(size_t mode=0;mode<6;++mode)validate(mode,"pre");
  for(int round=-6;round<12;++round)for(int position=0;position<6;++position){
   int index=(position+round+6)%6;if(reverse)index=5-index;
   auto began=std::chrono::steady_clock::now();auto cb=encode(size_t(index));double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-began).count();double gpu=(cb.GPUEndTime-cb.GPUStartTime)*1000.0;Need(std::isfinite(gpu)&&gpu>0&&std::isfinite(wall)&&wall>0,"Missing timer");
   report["samples"].push_back({{"shape",shape},{"round",round},{"position",position},{"method_index",index},{"mode",names[size_t(index)]},{"gpu_start",cb.GPUStartTime},{"gpu_end",cb.GPUEndTime},{"gpu_ms",gpu},{"wall_ms",wall},{"warmup",round<0}});
  }
  for(size_t mode=0;mode<6;++mode)validate(mode,"post");
  Report(dest/"result1.json",report);std::cout<<J{{"timed_shape",shape},{"reverse",reverse},{"samples",report["samples"].size()}}.dump()<<'\n'<<std::flush;

  Save(dest/(shape+"-compact.u16"),compact.contents,compact.length);
 }}
 report.update({{"completed",true},{"device",device.name.UTF8String},{"no_native_integration",true},{"operator_timing_only",true},{"reverse",reverse},{"writer_map_qualified",false},{"performance_admitted",false},{"no_comparator_execution",true},{"goal_complete",false}});Report(dest/"result1.json",report);return 0;
}}catch(const std::exception& error){report["error"]=error.what();if(!dest.empty()&&fs::exists(dest))Report(dest/"result1.json",report);std::cerr<<error.what()<<'\n';return 1;}}
