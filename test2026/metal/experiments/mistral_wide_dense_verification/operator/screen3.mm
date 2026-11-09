// Strict FP32 operator gate for up-to16 selected physical rows. No runtime claim.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "nlohmann/json.hpp"
#include <CommonCrypto/CommonDigest.h>
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
#include <vector>
using J=nlohmann::json;namespace fs=std::filesystem;
void Need(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
std::vector<uint8_t> Read(fs::path p){std::ifstream f(p,std::ios::binary|std::ios::ate);Need(bool(f),"Input open failed");auto n=f.tellg();Need(n>=0,"Input length failed");std::vector<uint8_t> v(size_t(n),0);f.seekg(0);f.read((char*)v.data(),n);Need(bool(f),"Input read failed");return v;}
J Json(fs::path p){std::ifstream f(p);J j;f>>j;return j;}
void Save(fs::path p,const J& j){std::ofstream f(p);f<<j.dump(2)<<'\n';Need(bool(f),"JSON write failed");}
void Binary(fs::path p,const void* b,size_t n){std::ofstream f(p,std::ios::binary);f.write((const char*)b,n);Need(bool(f),"Binary write failed");}
std::string Sha(const void* b,size_t n){Need(n<=UINT32_MAX,"SHA size overflow");unsigned char d[32];CC_SHA256(b,CC_LONG(n),d);std::ostringstream s;for(auto v:d)s<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(v);return s.str();}
struct Params{uint64_t rows,inner,columns,aStride,oStride;};
struct Clock{double start,end;uint64_t hostStart,hostEnd;};
uint64_t Now(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
int main(int argc,char** argv){fs::path out;J report={{"completed",false},{"cases",J::array()}};try{@autoreleasepool{
 Need(argc==4,"STUDY FRESH-OUTPUT COHORT");fs::path root=argv[1];out=argv[2];int cohort=std::stoi(argv[3]);Need(cohort==1||cohort==2,"Bad cohort");Need(!fs::exists(out),"Never overwrite output");fs::create_directory(out);auto plan=Json(root/"plan1.json");
 auto device=MTLCreateSystemDefaultDevice();Need(device!=nil,"No Metal device");const auto baseline=device.currentAllocatedSize;report["baseline_allocated_bytes"]=baseline;report["cohort"]=cohort;report["device"]=device.name.UTF8String;
 {@autoreleasepool{
 auto queue=[device newCommandQueue];Need(queue!=nil,"No queue");NSError* error=nil;std::array<id<MTLLibrary>,3> libraries;
 for(int m=0;m<3;++m){auto path=root/(m==0?"control1.metal":m==1?"interleaved3.metal":"row-major3.metal");auto source=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:path.c_str()] encoding:NSUTF8StringEncoding error:&error];Need(source!=nil,"No shader");auto options=[MTLCompileOptions new];options.languageVersion=MTLLanguageVersion3_1;options.fastMathEnabled=NO;libraries[m]=[device newLibraryWithSource:source options:options error:&error];if(!libraries[m])throw std::runtime_error(error.localizedDescription.UTF8String);}
 constexpr uint32_t guard=0x7f1ab37du;uint64_t parentPairs=0;
 for(auto definition:plan["cases"]){@autoreleasepool{
  const std::string name=definition["name"];const size_t k=definition["k"],n=definition["n"];
  std::array<std::vector<uint8_t>,3> raw{Read(definition["weight"].get<std::string>()),Read(definition["scales"].get<std::string>()),Read(definition["biases"].get<std::string>())};
  Need(raw[0].size()==n*k/2&&raw[1].size()==n*k/32&&raw[2].size()==raw[1].size(),"Weight layout differs");
  std::array<id<MTLBuffer>,3> weights;for(int i=0;i<3;++i){weights[i]=[device newBufferWithBytes:raw[i].data() length:raw[i].size() options:MTLResourceStorageModeShared];Need(weights[i]!=nil,"Weight upload failed");}
  auto captured=Read(definition["input"].get<std::string>());Need(captured.size()>=16*k*4,"Input too short");J item={{"name",name},{"k",k},{"n",n},{"layouts",J::array()}};
  for(int layout=0;layout<2;++layout){@autoreleasepool{
   const size_t off=layout?16:0,pad=layout?8:0;Params p{16,k,n,k+pad,n+pad};
   std::vector<uint32_t> input(off+16*p.aStride+16,guard),blank(off+16*p.oStride+16,guard);
   for(size_t row=0;row<16;++row){std::memcpy(input.data()+off+row*p.aStride,captured.data()+row*k*4,k*4);std::fill_n(blank.data()+off+row*p.oStride,n,0);}
   auto source=[device newBufferWithBytes:input.data() length:input.size()*4 options:MTLResourceStorageModeShared];auto output=[device newBufferWithLength:blank.size()*4 options:MTLResourceStorageModeShared];Need(source&&output,"Matrix allocation failed");
   std::array<id<MTLComputePipelineState>,4> scalar,blocked;std::array<std::array<id<MTLComputePipelineState>,16>,2> wide;
   auto make=[&](int lib,NSString* function,uint32_t rows){auto fc=[MTLFunctionConstantValues new];const uint32_t dims[4]{uint32_t(k),uint32_t(n),uint32_t(p.aStride),uint32_t(p.oStride)};for(NSUInteger i=0;i<4;++i)[fc setConstantValue:&dims[i] type:MTLDataTypeUInt atIndex:10+i];if(rows)[fc setConstantValue:&rows type:MTLDataTypeUInt atIndex:0];auto fn=[libraries[lib] newFunctionWithName:function constantValues:fc error:&error];if(!fn)throw std::runtime_error(error.localizedDescription.UTF8String);auto pipe=[device newComputePipelineStateWithFunction:fn error:&error];if(!pipe)throw std::runtime_error(error.localizedDescription.UTF8String);Need(pipe.threadExecutionWidth==32&&pipe.maxTotalThreadsPerThreadgroup>=128&&pipe.staticThreadgroupMemoryLength<=device.maxThreadgroupMemoryLength,"Invalid pipeline bound");return pipe;};
   for(uint32_t count=1;count<=4;++count){scalar[count-1]=make(0,@"affine_q4_indexed_decode",count);blocked[count-1]=make(0,@"affine_q4_blocked",count);}for(uint32_t count=1;count<=16;++count){wide[0][count-1]=make(1,@"affine_q4_wide",count);wide[1][count-1]=make(2,@"affine_q4_wide",count);}
   auto dispatch=[&](int method,const std::vector<uint32_t>& indices,bool single){Clock clock{};clock.hostStart=Now();@autoreleasepool{
    auto cb=[queue commandBuffer];Need(cb!=nil,"Command allocation failed");const uint32_t count=uint32_t(indices.size());
    auto bind=[&](id<MTLComputeCommandEncoder> e,const uint32_t* selected,size_t selectedCount){[e setBuffer:source offset:off*4 atIndex:0];for(NSUInteger i=0;i<3;++i)[e setBuffer:weights[i] offset:0 atIndex:i+1];[e setBuffer:output offset:off*4 atIndex:4];[e setBytes:&p length:sizeof(p) atIndex:5];[e setBytes:selected length:selectedCount*4 atIndex:6];};
    if(method==0){const size_t step=single?1:4;for(size_t first=0;first<indices.size();first+=step){const size_t live=std::min(step,indices.size()-first);const bool block=!single&&live==4&&n>=4096;auto e=[cb computeCommandEncoder];Need(e!=nil,"Control encoder failed");[e setComputePipelineState:block?blocked[live-1]:scalar[live-1]];bind(e,indices.data()+first,live);size_t cols=block?16:4;[e dispatchThreadgroups:MTLSizeMake((n+cols-1)/cols,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[e endEncoding];}}
    else if(count){auto e=[cb computeCommandEncoder];Need(e!=nil,"Wide encoder failed");[e setComputePipelineState:wide[method-1][count-1]];bind(e,indices.data(),indices.size());[e setBytes:&count length:4 atIndex:7];[e dispatchThreadgroups:MTLSizeMake(((n+15)/16)*((count+3)/4),1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[e endEncoding];}
    [cb commit];[cb waitUntilCompleted];Need(cb.status==MTLCommandBufferStatusCompleted,"GPU failed");clock.start=cb.GPUStartTime;clock.end=cb.GPUEndTime;
   }clock.hostEnd=Now();return clock;};
   std::vector<uint32_t> all(16);std::iota(all.begin(),all.end(),0);std::memcpy(output.contents,blank.data(),output.length);dispatch(0,all,true);std::vector<uint32_t> reference(blank.size());std::memcpy(reference.data(),output.contents,output.length);
   const std::string stem=name+"-l"+std::to_string(layout);Binary(out/(stem+"-input.u32"),input.data(),input.size()*4);Binary(out/(stem+"-single-row-reference.u32"),reference.data(),reference.size()*4);
   J cell={{"layout",layout},{"offset_words",off},{"a_stride",p.aStride},{"o_stride",p.oStride},{"parent_words",blank.size()},{"reference_sha256",Sha(reference.data(),reference.size()*4)},{"cells",J::array()},{"timing",J::array()}};
   std::vector<std::vector<uint32_t>> selections;for(size_t count=0;count<=16;++count)selections.emplace_back(all.begin(),all.begin()+count);for(auto s:plan["extra_masks"])selections.push_back(s.get<std::vector<uint32_t>>());
   for(size_t c=0;c<selections.size();++c){auto& selected=selections[c];auto expected=blank;for(auto row:selected)std::copy_n(reference.data()+off+row*p.oStride,n,expected.data()+off+row*p.oStride);
    J record={{"selection",selected},{"id",c},{"outputs",J::array()}};
    for(int order=0;order<3;++order){int method=cohort==1?order:2-order;std::memcpy(output.contents,blank.data(),output.length);dispatch(method,selected,false);Need(!std::memcmp(output.contents,expected.data(),output.length),"Wide projection differs from ordinary FP32 parent");parentPairs+=blank.size();std::string file=stem+"-c"+std::to_string(c)+"-m"+std::to_string(method)+".u32";Binary(out/file,output.contents,output.length);record["outputs"].push_back({{"method",method},{"file",file},{"sha256",Sha(output.contents,output.length)},{"exact",true}});}
    cell["cells"].push_back(record);
   }
   // All pipelines, weights, controls and expected outputs exist before timing.
   // No hashing, file I/O or builds inside the timed collection.
   if(layout==0)for(size_t count:plan["timed_counts"].get<std::vector<size_t>>()){
    std::vector<uint32_t> selected(all.begin(),all.begin()+count);auto expected=blank;for(auto row:selected)std::copy_n(reference.data()+off+row*p.oStride,n,expected.data()+off+row*p.oStride);
    for(int round=-4;round<17;++round)for(int order=0;order<3;++order){int method=(order+(round+4)%3)%3;if(cohort==2)method=2-method;std::memcpy(output.contents,blank.data(),output.length);auto clock=dispatch(method,selected,false);Need(!std::memcmp(output.contents,expected.data(),output.length),"Timed projection drift");if(round>=0){Need(std::isfinite(clock.start)&&std::isfinite(clock.end)&&clock.end>clock.start,"Invalid GPU clock");cell["timing"].push_back({{"count",count},{"method",method},{"round",round},{"order",order},{"gpu_start_seconds",clock.start},{"gpu_end_seconds",clock.end},{"gpu_ms",(clock.end-clock.start)*1000},{"host_start_ns",clock.hostStart},{"host_end_ns",clock.hostEnd}});}}
   }
   Need(!std::memcmp(source.contents,input.data(),source.length),"Input parent changed");Binary(out/(stem+"-input-after.u32"),source.contents,source.length);for(int i=0;i<3;++i)Need(!std::memcmp(weights[i].contents,raw[i].data(),weights[i].length),"Weights changed");cell["weights_readonly"]=true;cell["input_parent_readonly"]=true;item["layouts"].push_back(cell);
  }}
  report["cases"].push_back(item);report["parent_word_pairs"]=parentPairs;Save(out/"result1.json",report);std::cout<<J{{"cohort",cohort},{"case",name},{"parent_word_pairs",parentPairs}}.dump()<<'\n'<<std::flush;
 }}
 }}
 report["close_allocated_bytes"]=device.currentAllocatedSize;Need(device.currentAllocatedSize<=baseline+4194304,"Metal allocations retained");report["completed"]=true;report["all_parent_bits_exact_ordinary_fp32"]=true;report["compile_language"]=std::string("3.1");report["fast_math"]=false;report["no_runtime_integration"]=true;report["serving_speed_claim"]=false;report["goal_complete"]=false;Save(out/"result1.json",report);return 0;
}}catch(const std::exception& e){report["error"]=e.what();if(!out.empty()&&fs::exists(out))Save(out/"result1.json",report);std::cerr<<e.what()<<'\n';return 1;}}
