// Fixed-profile GQA score sharing screen. No inference-runtime modifications.
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
void Need(bool x,const char* why){if(!x)throw std::runtime_error(why);}
J Read(fs::path p){std::ifstream f(p);J j;f>>j;return j;}
void Save(fs::path p,const J& j){std::ofstream f(p);f<<j.dump(2)<<'\n';Need(bool(f),"JSON write failed");}
void Binary(fs::path p,const void* v,size_t n){std::ofstream f(p,std::ios::binary);f.write((const char*)v,n);Need(bool(f),"Binary write failed");}
std::string Sha(const void* ptr,size_t n){Need(n<=UINT32_MAX,"Hash bound");unsigned char d[32];CC_SHA256(ptr,CC_LONG(n),d);std::ostringstream s;for(auto b:d)s<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(b);return s.str();}
uint64_t Now(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
struct Params{uint64_t batch,chunk,heads,kvHeads,dim,capacity,qStride,kStride,vStride,kcStride,vcStride,sStride,aStride,gStride,pStride,oStride,gated;};
struct Clock{double start,end;uint64_t hostStart,hostEnd;};
struct Case{std::vector<size_t> firsts;std::vector<unsigned> masks;};
int main(int argc,char** argv){fs::path out;J report={{"completed",false},{"cases",J::array()},{"inputs",J::array()},{"resources",J::array()},{"timing",J::array()}};
try{@autoreleasepool{
 Need(argc==4,"STUDY FRESH-OUTPUT COHORT");fs::path root=argv[1];out=argv[2];int cohort=std::stoi(argv[3]);Need(cohort==1||cohort==2,"Cohort");Need(!fs::exists(out),"Output exists");fs::create_directory(out);auto plan=Read(root/"plan1.json");report["cohort"]=cohort;
 auto device=MTLCreateSystemDefaultDevice();Need(device!=nil,"No Metal device");const auto baseline=device.currentAllocatedSize;report["baseline_allocated_bytes"]=baseline;report["device"]=device.name.UTF8String;uint64_t pairs=0;
 {@autoreleasepool{
 auto queue=[device newCommandQueue];Need(queue!=nil,"No queue");NSError* error=nil;
 std::vector<size_t> batches{1,4};if(cohort==2)std::reverse(batches.begin(),batches.end());
 for(size_t batch:batches){@autoreleasepool{
  std::array<id<MTLLibrary>,2> libs;for(int i=0;i<2;++i){auto path=root/(i?"candidate1.metal":"control1.metal");auto s=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:path.c_str()] encoding:NSUTF8StringEncoding error:&error];Need(s!=nil,"No MSL");if(batch==1)s=[@"#define GARNET_VERIFY_MISTRAL_LAYOUT 1\n" stringByAppendingString:s];libs[i]=[device newLibraryWithSource:s options:nil error:&error];if(!libs[i])throw std::runtime_error(error.localizedDescription.UTF8String);}
  std::array<const char*,5> names{"verification_attention_scores","verification_split_score_dots","verification_split_score_softmax","verification_gqa16_dots","verification_gqa16_softmax"};std::array<id<MTLComputePipelineState>,5> psos;
  for(int i=0;i<5;++i){id<MTLFunction> fn;if(i==1){auto fc=[MTLFunctionConstantValues new];uint32_t tile=128;[fc setConstantValue:&tile type:MTLDataTypeUInt atIndex:1];fn=[libs[0] newFunctionWithName:@"verification_split_score_dots" constantValues:fc error:&error];}else fn=[libs[i<3?0:1] newFunctionWithName:[NSString stringWithUTF8String:names[i]]];Need(fn!=nil,"No MSL function");auto p=[device newComputePipelineStateWithFunction:fn error:&error];if(!p)throw std::runtime_error(error.localizedDescription.UTF8String);size_t dynamic=(i==0||i==2||i==4)?16384:0;Need(p.threadExecutionWidth==32&&p.maxTotalThreadsPerThreadgroup>=128&&p.staticThreadgroupMemoryLength+dynamic<=device.maxThreadgroupMemoryLength,"PSO resources");psos[i]=p;report["resources"].push_back({{"batch",batch},{"name",names[i]},{"simd_width",p.threadExecutionWidth},{"max_threads",p.maxTotalThreadsPerThreadgroup},{"static_threadgroup_bytes",p.staticThreadgroupMemoryLength},{"dynamic_threadgroup_bytes",dynamic}});}
  for(size_t layout=0;layout<2;++layout){@autoreleasepool{
   constexpr uint32_t guard=0x1937abef;size_t off=layout?16:0,pad=layout?8:0;Params p{batch,4,32,8,128,4096,4096+pad,1024+pad,1024+pad,8*4096*128+pad,8*4096*128+pad,4+pad,4+pad,4096+pad,32*4096+pad,4096+pad,0};
   std::array<std::vector<uint32_t>,3> input;input[0].assign(off+batch*4*p.qStride+16,guard);input[1].assign(off+batch*4*p.kStride+16,guard);input[2].assign(off+batch*p.kcStride+16,guard);
   uint32_t state=0x6a09e667u+layout+uint32_t(batch*77);auto random=[&](){state^=state<<13;state^=state>>17;state^=state<<5;return float(int(state&65535)-32768)/65536.0f;};
   for(size_t r=0;r<batch*4;++r){for(size_t j=0;j<4096;++j){float v=random();std::memcpy(input[0].data()+off+r*p.qStride+j,&v,4);}for(size_t j=0;j<1024;++j){float v=random();std::memcpy(input[1].data()+off+r*p.kStride+j,&v,4);}}
   for(size_t r=0;r<batch;++r)for(size_t j=0;j<8*4096*128;++j){float v=random();std::memcpy(input[2].data()+off+r*p.kcStride+j,&v,4);}
   auto upload=[&](const std::vector<uint32_t>& v){auto b=[device newBufferWithBytes:v.data() length:v.size()*4 options:MTLResourceStorageModeShared];Need(b!=nil,"Buffer allocation");return b;};std::array<id<MTLBuffer>,3> buffers;
   std::string stem="b"+std::to_string(batch)+"-l"+std::to_string(layout);
   for(int i=0;i<3;++i){buffers[i]=upload(input[i]);std::string file=stem+"-input"+std::to_string(i)+".u32";Binary(out/file,input[i].data(),input[i].size()*4);report["inputs"].push_back({{"batch",batch},{"layout",layout},{"input",i},{"file",file},{"words",input[i].size()},{"sha256",Sha(input[i].data(),input[i].size()*4)}});}
   std::vector<uint32_t> slotsWords(off+batch*p.sStride+16,guard),activeWords(off+batch*p.aStride+16,guard),blank(off+batch*4*p.pStride+16,guard);auto slots=upload(slotsWords),active=upload(activeWords),output=upload(blank);
   auto setCase=[&](const Case& c){size_t visible=0;for(size_t r=0;r<batch;++r)for(size_t t=0;t<4;++t){float s=float(c.firsts[r]+t),a=(c.masks[r]&(1u<<t))?1.0f:0.0f;Need(!a||s<4096,"Active slot out of bounds");std::memcpy(slotsWords.data()+off+r*p.sStride+t,&s,4);std::memcpy(activeWords.data()+off+r*p.aStride+t,&a,4);if(a)visible=std::max(visible,c.firsts[r]+t+1);}std::memcpy(slots.contents,slotsWords.data(),slots.length);std::memcpy(active.contents,activeWords.data(),active.length);return visible;};
   auto encode=[&](int method,size_t visible){Clock c{};c.hostStart=Now();@autoreleasepool{
    auto cb=[queue commandBuffer];Need(cb!=nil,"No command");auto bind=[&](id<MTLComputeCommandEncoder> e){for(int i=0;i<3;++i)[e setBuffer:buffers[i] offset:off*4 atIndex:i];[e setBuffer:slots offset:off*4 atIndex:3];[e setBuffer:active offset:off*4 atIndex:4];[e setBuffer:output offset:off*4 atIndex:5];[e setBytes:&p length:sizeof(p) atIndex:6];};
    bool split=method==1&&batch==1&&visible>=512;bool gqa=method==2;
    if((split||gqa)&&visible){auto e=[cb computeCommandEncoder];Need(e!=nil,"No dots encoder");bind(e);[e setComputePipelineState:psos[gqa?3:1]];size_t chunk=gqa?32:128;[e dispatchThreadgroups:MTLSizeMake(gqa?batch*8:batch*4*32,(visible+chunk-1)/chunk,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[e endEncoding];}
    auto e=[cb computeCommandEncoder];Need(e!=nil,"No softmax encoder");bind(e);[e setComputePipelineState:psos[gqa?4:split?2:0]];[e setThreadgroupMemoryLength:16384 atIndex:0];[e dispatchThreadgroups:MTLSizeMake(batch*4*32,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[e endEncoding];[cb commit];[cb waitUntilCompleted];Need(cb.status==MTLCommandBufferStatusCompleted,"GPU command failed");c.start=cb.GPUStartTime;c.end=cb.GPUEndTime;
   }c.hostEnd=Now();Need(std::isfinite(c.start)&&std::isfinite(c.end)&&c.end>c.start,"GPU clock");return c;};
   auto readonly=[&](){for(int i=0;i<3;++i)Need(!std::memcmp(buffers[i].contents,input[i].data(),buffers[i].length),"Input modified");Need(!std::memcmp(slots.contents,slotsWords.data(),slots.length)&&!std::memcmp(active.contents,activeWords.data(),active.length),"Control modified");};
   auto validate=[&](const std::vector<uint32_t>& a,const Case& c){for(size_t i=0;i<off;++i)Need(a[i]==guard,"Prefix guard");for(size_t r=0;r<batch;++r)for(size_t t=0;t<4;++t){const size_t base=off+(r*4+t)*p.pStride;for(size_t h=0;h<32;++h){double sum=0;for(size_t s=0;s<4096;++s){uint32_t word=a[base+h*4096+s];float v;std::memcpy(&v,&word,4);if(!(c.masks[r]&(1u<<t))||s>c.firsts[r]+t)Need(word==0,"Inactive/suffix positive zero");else{Need(std::isfinite(v)&&v>=0&&v<=1,"Probability bound");sum+=v;}}if(c.masks[r]&(1u<<t))Need(std::abs(sum-1)<2e-5,"Row sum");}for(size_t j=32*4096;j<p.pStride;++j)Need(a[base+j]==guard,"Stride guard");}for(size_t i=off+batch*4*p.pStride;i<a.size();++i)Need(a[i]==guard,"Suffix guard");};
   std::vector<Case> cases;
   for(size_t first:plan["firsts"].get<std::vector<size_t>>())for(unsigned mask:plan["all_uniform_masks"].get<std::vector<unsigned>>()){bool valid=true;for(size_t t=0;t<4;++t)if((mask&(1u<<t))&&first+t>=4096)valid=false;if(valid)cases.push_back({std::vector<size_t>(batch,first),std::vector<unsigned>(batch,mask)});}
   if(batch==4)for(auto f:plan["mixed_firsts"])for(auto m:plan["mixed_masks"]){Case c{f.get<std::vector<size_t>>(),m.get<std::vector<unsigned>>()};bool valid=true;for(size_t r=0;r<batch;++r)for(size_t t=0;t<4;++t)if((c.masks[r]&(1u<<t))&&c.firsts[r]+t>=4096)valid=false;if(valid)cases.push_back(c);}
   for(size_t index=0;index<cases.size();++index){const auto& c=cases[index];size_t visible=setCase(c);std::string id=stem+"-c"+std::to_string(index);std::array<std::vector<uint32_t>,3> results;J record={{"id",id},{"batch",batch},{"layout",layout},{"index",index},{"firsts",c.firsts},{"masks",c.masks},{"params",std::vector<uint64_t>((uint64_t*)&p,(uint64_t*)&p+17)},{"offset_words",off},{"parent_words",blank.size()},{"outputs",J::array()}};
    for(int order=0;order<3;++order){int method=cohort==1?order:2-order;std::memcpy(output.contents,blank.data(),output.length);encode(method,visible);results[method].resize(blank.size());std::memcpy(results[method].data(),output.contents,output.length);validate(results[method],c);std::string file=id+"-m"+std::to_string(method)+".u32";Binary(out/file,results[method].data(),output.length);record["outputs"].push_back({{"method",method},{"file",file},{"sha256",Sha(results[method].data(),output.length)}});}
    report["cases"].push_back(record);for(int m=1;m<3;++m){size_t mismatch=0;for(size_t i=0;i<blank.size();++i)if(results[0][i]!=results[m][i])++mismatch;if(mismatch){report["mismatch"]={{"id",id},{"method",m},{"words",mismatch}};Save(out/"result1.json",report);Need(false,"GQA probability differs from native control bits");}pairs+=blank.size();}
   }
   Save(out/"result1.json",report);
   // No hashing, file I/O, compilation or concurrent GPU workers during timing.
   if(layout==0)for(size_t first:plan["timed_firsts"].get<std::vector<size_t>>())for(unsigned mask:plan["timed_masks"].get<std::vector<unsigned>>()){
    Case c{std::vector<size_t>(batch,first),std::vector<unsigned>(batch,mask)};size_t visible=setCase(c);std::memcpy(output.contents,blank.data(),output.length);encode(0,visible);std::vector<uint32_t> expected(blank.size());std::memcpy(expected.data(),output.contents,output.length);
    for(int round=-4;round<17;++round)for(int order=0;order<3;++order){int method=(order+(round+4)%3)%3;if(cohort==2)method=2-method;std::memcpy(output.contents,blank.data(),output.length);auto c=encode(method,visible);Need(!std::memcmp(expected.data(),output.contents,output.length),"Timed probability drift");if(round>=0)report["timing"].push_back({{"batch",batch},{"first",first},{"mask",mask},{"method",method},{"round",round},{"order",order},{"gpu_start_seconds",c.start},{"gpu_end_seconds",c.end},{"gpu_ms",(c.end-c.start)*1000},{"host_start_ns",c.hostStart},{"host_end_ns",c.hostEnd}});}
   }
   readonly();for(int i=0;i<3;++i)Binary(out/(stem+"-input"+std::to_string(i)+"-after.u32"),buffers[i].contents,buffers[i].length);report["parent_word_pairs"]=pairs;std::cout<<J{{"cohort",cohort},{"batch",batch},{"layout",layout},{"cases",report["cases"].size()},{"parent_word_pairs",pairs}}.dump()<<'\n'<<std::flush;
  }}
 }}
 }}
 report["close_allocated_bytes"]=device.currentAllocatedSize;Need(device.currentAllocatedSize<=baseline+4194304,"Resources retained");report["compiler_options"]=nullptr;report["all_pipelines_prepared_before_collection"]=true;report["readonly_guards_preserved"]=true;report["all_probability_parent_bits_exact"]=true;report["no_runtime_integration"]=true;report["serving_speed_claim"]=false;report["goal_complete"]=false;report["completed"]=true;Save(out/"result1.json",report);return 0;
}}catch(const std::exception& e){report["error"]=e.what();if(!out.empty()&&fs::exists(out))Save(out/"result1.json",report);std::cerr<<e.what()<<'\n';return 1;}}
