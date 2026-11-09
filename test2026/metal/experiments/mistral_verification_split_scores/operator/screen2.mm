// Isolated B1/T4 verification score scheduling screen; native runtime untouched.
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
#include <sstream>
#include <vector>
using J=nlohmann::json;namespace fs=std::filesystem;
void Need(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void Save(fs::path path,const J& data){auto tmp=path;tmp+=".tmp";std::ofstream f(tmp);Need(bool(f),"JSON open failed");f<<data.dump(2)<<'\n';f.close();fs::rename(tmp,path);}
void Binary(fs::path path,const void* bytes,size_t count){std::ofstream f(path,std::ios::binary);f.write((const char*)bytes,count);Need(bool(f),"Binary write failed");}
std::string Sha(const void* ptr,size_t bytes){Need(bytes<=UINT32_MAX,"SHA length overflow");unsigned char d[32];CC_SHA256(ptr,CC_LONG(bytes),d);std::ostringstream s;for(auto b:d)s<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(b);return s.str();}
uint64_t Now(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
struct Params{uint64_t batch,chunk,heads,kvHeads,dim,capacity,qStride,kStride,vStride,kcStride,vcStride,sStride,aStride,gStride,pStride,oStride,gated;};
struct Clock{double begin,end;uint64_t hostBegin,hostEnd;};
int main(int argc,char** argv){fs::path out;J report={{"completed",false},{"cells",J::array()},{"timing",J::array()},{"inputs",J::array()},{"resources",J::array()}};
try{@autoreleasepool{
 Need(argc==4,"STUDY FRESH-OUTPUT COHORT(1|2)");fs::path root=argv[1];out=argv[2];int cohort=std::stoi(argv[3]);Need(cohort==1||cohort==2,"Invalid cohort");Need(!fs::exists(out),"Never overwrite screen output");fs::create_directory(out);report["cohort"]=cohort;
 auto device=MTLCreateSystemDefaultDevice();Need(device!=nil,"No Metal device");const auto baseline=device.currentAllocatedSize;report["baseline_allocated_bytes"]=baseline;report["device"]=device.name.UTF8String;
 {@autoreleasepool{
 auto queue=[device newCommandQueue];Need(queue!=nil,"Queue creation failed");NSError* error=nil;std::array<id<MTLLibrary>,2> libs;
 for(int i=0;i<2;++i){auto path=root/(i?"candidate1.metal":"control1.metal");auto text=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:path.c_str()] encoding:NSUTF8StringEncoding error:&error];Need(text!=nil,"Missing MSL");libs[i]=[device newLibraryWithSource:text options:nil error:&error];if(!libs[i])throw std::runtime_error(error.localizedDescription.UTF8String);}
 std::array<id<MTLComputePipelineState>,3> pipelines;std::array<const char*,3> names{"verification_attention_scores","verification_split_score_dots","verification_split_score_softmax"};
 for(int i=0;i<3;++i){id<MTLFunction> fn;if(i==1){auto fc=[MTLFunctionConstantValues new];uint32_t tile=128;[fc setConstantValue:&tile type:MTLDataTypeUInt atIndex:1];fn=[libs[1] newFunctionWithName:@"verification_split_score_dots" constantValues:fc error:&error];}else fn=[libs[i?1:0] newFunctionWithName:[NSString stringWithUTF8String:names[i]]];Need(fn!=nil,"Missing function");pipelines[i]=[device newComputePipelineStateWithFunction:fn error:&error];if(!pipelines[i])throw std::runtime_error(error.localizedDescription.UTF8String);auto p=pipelines[i];Need(p.threadExecutionWidth==32&&p.maxTotalThreadsPerThreadgroup>=128&&p.staticThreadgroupMemoryLength+(i==1?0:16384)<=device.maxThreadgroupMemoryLength,"Pipeline resource limit");report["resources"].push_back({{"name",names[i]},{"simd_width",p.threadExecutionWidth},{"max_threads",p.maxTotalThreadsPerThreadgroup},{"static_threadgroup_bytes",p.staticThreadgroupMemoryLength},{"dynamic_threadgroup_bytes",i==1?0:16384}});}
 report["compile_options"]=nullptr;report["same_native_default_compiler"]=true;report["all_pipelines_prepared_before_collection"]=true;constexpr uint32_t guard=0x1937abef;
 uint64_t parentWords=0,checkedWords=0;
 for(int layout=0;layout<2;++layout){@autoreleasepool{
  size_t off=layout?16:0,pad=layout?8:0;Params p{1,4,32,8,128,4096,4096+pad,1024+pad,1024+pad,8*4096*128+pad,8*4096*128+pad,4+pad,4+pad,4096+pad,32*4096+pad,4096+pad,0};
  std::array<std::vector<uint32_t>,3> input;input[0].assign(off+4*p.qStride+16,guard);input[1].assign(off+4*p.kStride+16,guard);input[2].assign(off+p.kcStride+16,guard);
  uint32_t state=0x6a09e667u+layout;auto random=[&](){state^=state<<13;state^=state>>17;state^=state<<5;return (float(int(state&65535)-32768)/32768.0f)*0.5f;};
  for(size_t row=0;row<4;++row){for(size_t j=0;j<4096;++j){float v=random();std::memcpy(input[0].data()+off+row*p.qStride+j,&v,4);}for(size_t j=0;j<1024;++j){float v=random();std::memcpy(input[1].data()+off+row*p.kStride+j,&v,4);}}
  for(size_t j=0;j<8*4096*128;++j){float v=random();std::memcpy(input[2].data()+off+j,&v,4);}
  auto upload=[&](const std::vector<uint32_t>& v){auto b=[device newBufferWithBytes:v.data() length:v.size()*4 options:MTLResourceStorageModeShared];Need(b!=nil,"Upload failed");return b;};std::array<id<MTLBuffer>,3> buffers;for(int i=0;i<3;++i){buffers[i]=upload(input[i]);std::string file="layout"+std::to_string(layout)+"-input"+std::to_string(i)+".u32";Binary(out/file,input[i].data(),input[i].size()*4);report["inputs"].push_back({{"layout",layout},{"input",i},{"file",file},{"words",input[i].size()},{"sha256",Sha(input[i].data(),input[i].size()*4)}});}
  std::vector<uint32_t> slotWords(off+p.sStride+16,guard),activeWords(off+p.aStride+16,guard);auto slots=upload(slotWords),active=upload(activeWords);const size_t outputWords=off+4*p.pStride+16;
  auto output=[device newBufferWithLength:outputWords*4 options:MTLResourceStorageModeShared];Need(output!=nil,"Output allocation failed");std::vector<uint32_t> blank(outputWords,guard);
  auto setCase=[&](size_t first,unsigned mask){for(size_t t=0;t<4;++t){float s=float(first+t),a=(mask&(1u<<t))?1.0f:0.0f;std::memcpy(slotWords.data()+off+t,&s,4);std::memcpy(activeWords.data()+off+t,&a,4);}std::memcpy(slots.contents,slotWords.data(),slots.length);std::memcpy(active.contents,activeWords.data(),active.length);};
  auto clear=[&](){std::memcpy(output.contents,blank.data(),output.length);};
  auto encode=[&](bool candidate,size_t first,unsigned mask){Clock c{};@autoreleasepool{c.hostBegin=Now();auto cb=[queue commandBuffer];Need(cb!=nil,"Command creation failed");auto bind=[&](id<MTLComputeCommandEncoder> e){for(int i=0;i<3;++i)[e setBuffer:buffers[i] offset:off*4 atIndex:i];[e setBuffer:slots offset:off*4 atIndex:3];[e setBuffer:active offset:off*4 atIndex:4];[e setBuffer:output offset:off*4 atIndex:5];[e setBytes:&p length:sizeof(p) atIndex:6];};
   size_t visible=0;for(size_t t=0;t<4;++t)if(mask&(1u<<t))visible=std::max(visible,first+t+1);
   if(candidate&&visible){auto e=[cb computeCommandEncoder];Need(e!=nil,"Dot encoder failed");[e setComputePipelineState:pipelines[1]];bind(e);[e dispatchThreadgroups:MTLSizeMake(128,(visible+127)/128,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[e endEncoding];}
   auto e=[cb computeCommandEncoder];Need(e!=nil,"Score/softmax encoder failed");[e setComputePipelineState:pipelines[candidate?2:0]];bind(e);[e setThreadgroupMemoryLength:16384 atIndex:0];[e dispatchThreadgroups:MTLSizeMake(128,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[e endEncoding];[cb commit];[cb waitUntilCompleted];c.hostEnd=Now();Need(cb.status==MTLCommandBufferStatusCompleted,"GPU command failed");c.begin=cb.GPUStartTime;c.end=cb.GPUEndTime;Need(std::isfinite(c.begin)&&std::isfinite(c.end)&&c.end>c.begin,"Invalid GPU clock");}return c;};
  auto readonly=[&](){for(int i=0;i<3;++i)Need(!std::memcmp(buffers[i].contents,input[i].data(),buffers[i].length),"Read-only input changed");Need(!std::memcmp(slots.contents,slotWords.data(),slots.length)&&!std::memcmp(active.contents,activeWords.data(),active.length),"Slots/active changed");};
  auto validate=[&](const std::vector<uint32_t>& words,size_t first,unsigned mask){for(size_t i=0;i<off;++i)Need(words[i]==guard,"Output prefix guard changed");for(size_t t=0;t<4;++t){bool on=mask&(1u<<t);for(size_t h=0;h<32;++h){double sum=0;for(size_t s=0;s<4096;++s){auto bits=words[off+t*p.pStride+h*4096+s];float v;std::memcpy(&v,&bits,4);if(!on||s>first+t)Need(bits==0,"Inactive/suffix not positive zero");else{Need(std::isfinite(v)&&v>=0.0f&&v<=1.0f,"Probability invalid");sum+=v;}++checkedWords;}if(on)Need(std::abs(sum-1.0)<2e-5,"Probability row sum");}for(size_t j=32*4096;j<p.pStride;++j)Need(words[off+t*p.pStride+j]==guard,"Output stride padding changed");}for(size_t i=off+4*p.pStride;i<words.size();++i)Need(words[i]==guard,"Output suffix guard changed");};
  std::array<size_t,9> firsts{0,1,127,511,2048,4092,4093,4094,4095};
  // Cohort2 repeats the complete retained-word check, in reverse method order.
  for(size_t first:firsts)for(unsigned mask=0;mask<16;++mask){bool valid=true;for(size_t t=0;t<4;++t)if((mask&(1u<<t))&&first+t>=4096)valid=false;if(!valid)continue;setCase(first,mask);
   std::array<std::vector<uint32_t>,2> results;std::string id="l"+std::to_string(layout)+"-f"+std::to_string(first)+"-m"+std::to_string(mask);
   for(int order=0;order<2;++order){int method=cohort==1?order:1-order;clear();encode(method,first,mask);results[method].resize(outputWords);std::memcpy(results[method].data(),output.contents,output.length);std::string file=id+(method?"-split.u32":"-control.u32");Binary(out/file,results[method].data(),output.length);validate(results[method],first,mask);readonly();}
   size_t mismatch=0;for(size_t i=0;i<outputWords;++i)if(results[0][i]!=results[1][i])++mismatch;parentWords+=outputWords;report["cells"].push_back({{"id",id},{"layout",layout},{"first",first},{"mask",mask},{"params",std::vector<uint64_t>((uint64_t*)&p,(uint64_t*)&p+17)},{"offset_words",off},{"parent_words",outputWords},{"control_sha256",Sha(results[0].data(),outputWords*4)},{"split_sha256",Sha(results[1].data(),outputWords*4)},{"nonidentical_words",mismatch}});if(mismatch){Save(out/"result1.json",report);Need(false,"Split score differs from original probability bits");}
  }
  report["parent_words_checked"]=parentWords;report["probability_words_checked"]=checkedWords;Save(out/"result1.json",report);
  // Buffers/pipelines exist before collection. No hashes, builds or file I/O in
  // this phase. Complete candidate command includes both tracked encoders.
  if(layout==0)for(size_t first:std::array<size_t,3>{508,2044,4092})for(unsigned mask:{1u,15u}){
   setCase(first,mask);clear();encode(false,first,mask);std::vector<uint32_t> expected(outputWords);std::memcpy(expected.data(),output.contents,output.length);
   for(int round=-4;round<21;++round)for(int order=0;order<2;++order){bool method=((round+4+order+(cohort-1))&1)!=0;clear();auto c=encode(method,first,mask);Need(!std::memcmp(expected.data(),output.contents,output.length),"Timed probability drift");if(round>=0)report["timing"].push_back({{"first",first},{"mask",mask},{"method",method?"split":"control"},{"round",round},{"position",order},{"gpu_start_seconds",c.begin},{"gpu_end_seconds",c.end},{"gpu_ms",(c.end-c.begin)*1000},{"host_begin_ns",c.hostBegin},{"host_end_ns",c.hostEnd}});}
  }
  readonly();for(int i=0;i<3;++i){std::string file="layout"+std::to_string(layout)+"-input"+std::to_string(i)+"-after.u32";Binary(out/file,buffers[i].contents,buffers[i].length);}
  std::cout<<J{{"cohort",cohort},{"layout",layout},{"cells",report["cells"].size()},{"timing_samples",report["timing"].size()},{"parent_words",parentWords}}.dump()<<'\n'<<std::flush;
 }}
 }}
 report["close_allocated_bytes"]=device.currentAllocatedSize;Need(device.currentAllocatedSize<=baseline+4194304,"Standalone Metal resources retained");report["all_probability_parent_bits_exact"]=true;report["readonly_guards_preserved"]=true;report["completed"]=true;report["no_runtime_integration"]=true;report["serving_speed_claim"]=false;report["goal_complete"]=false;Save(out/"result1.json",report);return 0;
}}catch(const std::exception& e){report["error"]=e.what();if(!out.empty()&&fs::exists(out))Save(out/"result1.json",report);std::cerr<<e.what()<<'\n';return 1;}}
