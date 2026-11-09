// SPDX-License-Identifier: Apache-2.0
// Supersedes the invalid old eight-row collector: checked kernel transforms,
// fresh output poison on EVERY dispatch, indexed permutations and tail checks.
// From repo root: xcrun clang++ -std=c++17 -O3 -fobjc-arc -I.
// -Isrc/backends/metal -framework Foundation -framework Metal THIS_FILE -o PROBE
// Serial run: PROBE NEW_OUTPUT 15 > RAW_JSONL
// Known-broken writer must fail: PROBE NEW_NEGATIVE_OUTPUT 3 negative
// Native runtime still supports at most4 verification rows. This is research.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include "affine_int4_profile_shader.h"
#include "affine_int4_indexed_decode_shader.h"
#include "affine_int4_blocked_decode_shader.h"
#include "src/model/float16_codec.h"
#include "third_party/nlohmann/json.hpp"
using json=nlohmann::json;
void Need(bool b,const char* why){if(!b)throw std::runtime_error(why);}
size_t Replace(std::string& s,const std::string& from,const std::string& to){size_t at=0,count=0;while((at=s.find(from,at))!=std::string::npos){s.replace(at,from.size(),to);at+=to.size();++count;}return count;}
struct Params{uint64_t rows,inner,columns,aStride,oStride;};
struct Variant{std::string name;uint32_t columns,rows;bool split;id<MTLComputePipelineState> pipeline=nil;};
int main(int argc,char** argv){@autoreleasepool{try{
 Need(argc==3||argc==4,"Usage NEW_OUTPUT REPS [negative]");bool negative=argc==4;Need(!negative||std::string(argv[3])=="negative","Unknown mode");int reps=std::stoi(argv[2]);Need(reps>=3&&reps<=51,"reps bounds");
 auto root=std::filesystem::path(argv[1]);Need(std::filesystem::create_directory(root),"Output exists");auto device=MTLCreateSystemDefaultDevice();auto queue=[device newCommandQueue];NSError* error=nil;
 std::string source="#define GARNET_AFFINE_FP16_COEFFICIENTS 1\n#define GARNET_AFFINE_PROFILE 1\n#define GARNET_AFFINE_INDEX32 1\n";
 source+=Garnet::kAffineInt4ProfilePrelude;source+=Garnet::kAffineInt4IndexedDecodeShader;
 {
  std::string native=Garnet::kAffineInt4BlockedDecodeShader;
  Need(Replace(native,"BlockedQ4Params","NativeBlockedQ4Params")==3,"Native parameter rename");
  Need(Replace(native,"affine_q4_blocked","native_blocked4")==1,"Native kernel rename");
  source+="\n#define GARNET_BLOCKED_COLUMNS 4\n#define GARNET_BLOCKED_ROWS selected_decode_rows\n#define GARNET_BLOCKED_INDEXED 1\n"+native;
 }
 for(uint32_t columns:{2,4,8})for(uint32_t maximum:{4,8}) {
  std::string body=Garnet::kAffineInt4BlockedDecodeShader;auto suffix=std::to_string(columns)+"_m"+std::to_string(maximum);
  Need(Replace(body,"BlockedQ4Params","OccupancyParams"+suffix)==3,"Parameter transform changed");
  Need(Replace(body,"affine_q4_blocked","occupancy_c"+suffix)==1,"Kernel transform changed");
  if(maximum==8) {
   Need(Replace(body,"physical[4]","physical[8]")==1&&Replace(body,"sums[4]","sums[8]")==1,"Row storage transform changed");
   Need(Replace(body,"low[4]","low[8]")==1&&Replace(body,"high[4]","high[8]")==1,"Input transform changed");
   Need(Replace(body,"row<4","row<8")==4,"Every row loop must widen");
  }
  source+="\n#undef GARNET_BLOCKED_COLUMNS\n#define GARNET_BLOCKED_COLUMNS "+std::to_string(columns)+"\n"+body;
 }
 {
  std::string broken=Garnet::kAffineInt4BlockedDecodeShader;
  Need(Replace(broken,"BlockedQ4Params","BrokenParams")==3,"Negative parameter rename");
  Need(Replace(broken,"affine_q4_blocked","occupancy_c4_broken")==1,"Negative kernel rename");
  Need(Replace(broken,"physical[4]","physical[8]")==1&&Replace(broken,"sums[4]","sums[8]")==1,"Negative row storage");
  Need(Replace(broken,"low[4]","low[8]")==1&&Replace(broken,"high[4]","high[8]")==1,"Negative input storage");
  // Deliberately retain row<4: poisoning must reject the missing active rows.
  source+="\n#undef GARNET_BLOCKED_COLUMNS\n#define GARNET_BLOCKED_COLUMNS 4\n"+broken;
 }
 auto options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;options.languageVersion=MTLLanguageVersion4_0;
 auto library=[device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()] options:options error:&error];Need(library!=nil,error?error.localizedDescription.UTF8String:"compile");
 NSString* fs=@"#include <metal_stdlib>\nusing namespace metal;kernel void sweep(device const float* a [[buffer(0)]],device float* b [[buffer(1)]],uint id [[thread_position_in_grid]]){float sum=0;for(uint j=0;j<256;++j)sum+=a[id*256+j];b[id]=sum;}";
 auto fl=[device newLibraryWithSource:fs options:options error:&error];auto fp=[device newComputePipelineStateWithFunction:[fl newFunctionWithName:@"sweep"] error:&error];Need(fp!=nil,"sweep pipeline");
 auto trash=[device newBufferWithLength:256ull*1024*1024 options:MTLResourceStorageModeShared],sink=[device newBufferWithLength:1024*1024 options:MTLResourceStorageModeShared];Need(trash&&sink,"sweep allocation");std::fill_n((float*)trash.contents,64*1024*1024,.125f);
 auto sweep=[&](){auto cb=[queue commandBuffer];auto enc=[cb computeCommandEncoder];[enc setComputePipelineState:fp];[enc setBuffer:trash offset:0 atIndex:0];[enc setBuffer:sink offset:0 atIndex:1];[enc dispatchThreads:MTLSizeMake(256*1024,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[enc endEncoding];[cb commit];[cb waitUntilCompleted];Need(cb.status==MTLCommandBufferStatusCompleted,"sweep failed");};
 std::cout<<json{{"kind","metadata"},{"device",device.name.UTF8String},{"reps",reps},{"warmups",3},{"poison",0x7fc0abcdu},{"negative",negative},{"scope","Indexed projection occupancy, exact F32 controls/full FP64 bound. Entire output is poisoned before EVERY dispatch outside timed spans. Sweep not proof of cold cache. No model qualification."}}.dump()<<'\n'<<std::flush;
 size_t cases=0,checked=0,referenceWords=0;
 auto shapes=negative?std::vector<std::array<uint32_t,2>>{{7,64}}:std::vector<std::array<uint32_t,2>>{{1,64},{7,192},{17,64},{33,128},{4096,4096},{1024,4096},{14336,4096},{4096,14336},{32768,4096}};
 for(auto shape:shapes)for(uint32_t count:negative?std::vector<uint32_t>{8}:std::vector<uint32_t>{1,2,3,4,8}){@autoreleasepool{
  uint32_t n=shape[0],k=shape[1],physical=count==8?8:4;Params p{count,k,n,k,n};
  std::vector<uint32_t> indices=count==1?std::vector<uint32_t>{3}:count==2?std::vector<uint32_t>{3,0}:count==3?std::vector<uint32_t>{3,0,2}:count==4?std::vector<uint32_t>{3,0,2,1}:std::vector<uint32_t>{7,0,6,1,5,2,4,3};
  auto alloc=[&](size_t bytes){auto b=[device newBufferWithLength:bytes options:MTLResourceStorageModeShared];Need(b!=nil,"allocation");return b;};
  auto a=alloc(size_t(physical)*k*4),w=alloc(size_t(n)*k/2),scales=alloc(size_t(n)*k/32),biases=alloc(size_t(n)*k/32),out=alloc(size_t(physical)*n*4);
  uint32_t seed=37;for(size_t i=0;i<size_t(physical)*k;++i){seed=1664525u*seed+1013904223u;((float*)a.contents)[i]=float(int((seed>>8)%65521)-32760)*0x1p-18f;}
  for(size_t i=0;i<size_t(n)*k/8;++i)((uint32_t*)w.contents)[i]=uint32_t(i)*2654435761u+131u;
  for(size_t i=0;i<size_t(n)*k/64;++i){((uint16_t*)scales.contents)[i]=uint16_t(0x2c00+i%8);((uint16_t*)biases.contents)[i]=uint16_t(0xb000+i%7);}
  bool admitted=(k==4096&&(n==4096||n==14336||n==32768))||(k==14336&&n==4096);uint32_t controlColumns=(count==4||count==8)&&admitted?4:1;
  std::vector<Variant> variants{{"control",controlColumns,count==8?4u:count,count==8,nil},{"c2",2,count,false,nil},{"c4",4,count,false,nil},{"c8",8,count,false,nil}};
  if(negative)variants={{"control",controlColumns,4,true,nil},{"broken",4,8,false,nil}};
  for(auto& v:variants){auto constants=[MTLFunctionConstantValues new];uint32_t dims[]{k,n,k,n};for(NSUInteger i=0;i<4;++i)[constants setConstantValue:dims+i type:MTLDataTypeUInt atIndex:10+i];[constants setConstantValue:&v.rows type:MTLDataTypeUInt atIndex:0];NSString* name=v.name=="broken"?@"occupancy_c4_broken":v.name=="control"&&v.columns==4?@"native_blocked4":v.columns==1?@"affine_q4_indexed_decode":[NSString stringWithFormat:@"occupancy_c%u_m%u",v.columns,physical];auto fn=[library newFunctionWithName:name constantValues:constants error:&error];Need(fn!=nil,error?error.localizedDescription.UTF8String:"specialize");v.pipeline=[device newComputePipelineStateWithFunction:fn error:&error];Need(v.pipeline&&v.pipeline.maxTotalThreadsPerThreadgroup>=128,"pipeline bound");}
  auto run=[&](const Variant& v){
   // This is deliberately outside both host/GPU measured spans.
   std::fill_n((uint32_t*)out.contents,size_t(physical)*n,0x7fc0abcdu);
   auto start=std::chrono::steady_clock::now();auto cb=[queue commandBuffer];auto enc=[cb computeCommandEncoder];[enc setComputePipelineState:v.pipeline];
   for(uint32_t pass=0;pass<(v.split?2u:1u);++pass){Params actual=p;actual.rows=v.rows;std::vector<uint32_t> selected=v.split?std::vector<uint32_t>{3,0,2,1}:indices;
    [enc setBuffer:a offset:pass*4ull*k*4 atIndex:0];[enc setBuffer:w offset:0 atIndex:1];[enc setBuffer:scales offset:0 atIndex:2];[enc setBuffer:biases offset:0 atIndex:3];[enc setBuffer:out offset:pass*4ull*n*4 atIndex:4];[enc setBytes:&actual length:sizeof(actual) atIndex:5];[enc setBytes:selected.data() length:selected.size()*4 atIndex:6];[enc dispatchThreadgroups:MTLSizeMake((n+4*v.columns-1)/(4*v.columns),1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
   }
   [enc endEncoding];[cb commit];[cb waitUntilCompleted];Need(cb.status==MTLCommandBufferStatusCompleted,"dispatch failed");return std::array<double,2>{(cb.GPUEndTime-cb.GPUStartTime)*1000.,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()};
  };
  std::vector<double> expected(size_t(physical)*n),limit(expected.size());std::vector<bool> active(physical,false);for(auto row:indices)active[row]=true;
  for(uint32_t col=0;col<n;++col){std::array<double,8> sum{},absolute{};for(uint32_t j=0;j<k;++j){size_t c=size_t(col)*(k/64)+j/64;uint32_t q=(((uint32_t*)w.contents)[size_t(col)*(k/8)+j/8]>>(4*(j%8)))&15;float weight=std::fma(float(q),Garnet::Float16ToFloat32(((uint16_t*)scales.contents)[c]),Garnet::Float16ToFloat32(((uint16_t*)biases.contents)[c]));for(uint32_t row=0;row<physical;++row)if(active[row]){double product=double(((float*)a.contents)[row*k+j])*weight;sum[row]+=product;absolute[row]+=std::abs(product);}}for(uint32_t row=0;row<physical;++row)if(active[row]){expected[row*n+col]=sum[row];limit[row*n+col]=2e-6+4e-6*absolute[row];}}
  referenceWords+=size_t(count)*n;run(variants[0]);std::vector<uint32_t> reference(size_t(physical)*n);std::memcpy(reference.data(),out.contents,reference.size()*4);
  for(bool swept:{false,true}){std::vector<std::vector<double>> gpu(variants.size()),wall(variants.size());std::vector<double> maximum(variants.size());
   for(int rep=-3;rep<reps;++rep)for(size_t order=0;order<variants.size();++order){size_t v=(order+size_t(rep+3))%variants.size();if((rep+3)&1)v=variants.size()-1-v;if(swept)sweep();auto timing=run(variants[v]);
    for(size_t i=0;i<reference.size();++i){uint32_t bits=((uint32_t*)out.contents)[i];float actual=((float*)out.contents)[i];if(!active[i/n])Need(bits==0x7fc0abcdu,"Inactive row was overwritten");else{if(!std::isfinite(actual))std::cerr<<"missing-active-write variant="<<variants[v].name<<" rows="<<count<<" row="<<i/n<<" column="<<i%n<<'\n';Need(std::isfinite(actual),"Active output not written after poison");double fraction=std::abs(double(actual)-expected[i])/limit[i];Need(fraction<=1.,"Original FP64 bound exceeded");Need(bits==reference[i],"Active output differs from native control bits");maximum[v]=std::max(maximum[v],fraction);}++checked;}
    if(rep>=0){gpu[v].push_back(timing[0]);wall[v].push_back(timing[1]);}
   }
   for(size_t v=0;v<variants.size();++v){run(variants[v]);auto file="case-"+std::to_string(cases)+"-"+(swept?"swept":"hot")+"-"+variants[v].name+".bin";std::ofstream f(root/file,std::ios::binary);f.write((char*)out.contents,reference.size()*4);Need(bool(f),"write");std::cout<<json{{"kind","cell"},{"index",cases},{"rows",count},{"physical_rows",physical},{"indices",indices},{"outputs",n},{"inner",k},{"variant",variants[v].name},{"control_columns",controlColumns},{"swept",swept},{"max_error_fraction",maximum[v]},{"gpu_ms",gpu[v]},{"wall_ms",wall[v]},{"output",file}}.dump()<<'\n'<<std::flush;}
  }++cases;
 }}
 std::cout<<json{{"kind","closed"},{"cases",cases},{"checked_physical_words",checked},{"independent_fp64_outputs",referenceWords}}.dump()<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}}
