// SPDX-License-Identifier: Apache-2.0
// Isolated FP32 Q4 decode fusion; no serving/runtime claim.
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
#include <map>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>
using J=nlohmann::json;namespace fs=std::filesystem;
void Need(bool value,const char* why){if(!value)throw std::runtime_error(why);}
template<class T>std::vector<T> Read(fs::path p,size_t count){std::ifstream f(p,std::ios::binary|std::ios::ate);Need(bool(f)&&size_t(f.tellg())==count*sizeof(T),"Fixture size mismatch");f.seekg(0);std::vector<T> v(count);f.read((char*)v.data(),count*sizeof(T));Need(bool(f),"Fixture read failed");return v;}
std::string Sha(const void* p,size_t bytes){Need(bytes<=UINT32_MAX,"Hash too large");unsigned char d[32];CC_SHA256(p,CC_LONG(bytes),d);std::ostringstream s;for(auto b:d)s<<std::hex<<std::setfill('0')<<std::setw(2)<<unsigned(b);return s.str();}
void Save(fs::path p,const void* data,size_t bytes){std::ofstream f(p,std::ios::binary);f.write((const char*)data,bytes);Need(bool(f),"Save failed");}
void Report(fs::path p,const J& j){auto t=p;t+=".tmp";std::ofstream f(t);f<<j.dump(2)<<'\n';f.close();fs::rename(t,p);}
struct Params{uint64_t rows,inner,columns,aStride,oStride;};
struct Activation{uint64_t rows,columns,aStride,bStride,cStride;float epsilon,offset;};
int main(int argc,char** argv){J report={{"completed",false},{"cases",J::array()},{"fp64",J::array()},{"samples",J::array()},{"pipelines",J::array()}};fs::path dest;
try{@autoreleasepool{
 Need(argc==10,"STUDY ASSETS CAPTURES FRESH_OUTPUT correctness|warm|pressure ORDER ROUNDS contiguous|padded CANDIDATE(-1=all)");
 fs::path study=argv[1];dest=argv[4];Need(!fs::exists(dest),"Never overwrite output");fs::create_directory(dest);
 std::string mode=argv[5];bool correctness=mode=="correctness",pressure=mode=="pressure",padded=std::string(argv[8])=="padded";
 int order=std::stoi(argv[6]),rounds=std::stoi(argv[7]),selected=std::stoi(argv[9]);Need((correctness||mode=="warm"||pressure)&&(order==0||order==1)&&rounds>=3&&rounds<=31&&selected>=-1&&selected<=2,"Invalid mode/policy");
 Need(padded||std::string(argv[8])=="contiguous","Invalid layout");Need(correctness||selected>=0,"Timing requires an audited candidate");
 auto device=MTLCreateSystemDefaultDevice();Need(device!=nil,"Metal unavailable");size_t baseline=device.currentAllocatedSize;
 report.update({{"mode",mode},{"order",order},{"rounds",rounds},{"warmups",5},{"padded",padded},{"device",device.name.UTF8String},{"candidate_selected",selected},{"baseline_allocated_bytes",baseline},{"pressure_bytes",pressure?268435456:0}});
 {@autoreleasepool{
 auto queue=[device newCommandQueue];Need(queue!=nil,"Command queue absent");NSError* error=nil;
 auto strict=[MTLCompileOptions new];strict.languageVersion=MTLLanguageVersion3_1;strict.fastMathEnabled=NO;
 auto library=[&](const char* name,MTLCompileOptions* options,const char* exp){
  auto path=study/name;NSString* text=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:path.c_str()] encoding:NSUTF8StringEncoding error:&error];Need(text!=nil,"Shader missing");
  if(exp)text=[[NSString stringWithFormat:@"#define FUSED_EXP %s\n",exp] stringByAppendingString:text];
  auto lib=[device newLibraryWithSource:text options:options error:&error];if(!lib)throw std::runtime_error(error.localizedDescription.UTF8String);return lib;
 };
 auto ordinary=library("ordinary1.metal",strict,nullptr),indexed=library("decode1.metal",strict,"exp"),pointwise=library("activation1.metal",nil,nullptr);
 std::array<id<MTLLibrary>,3> candidates{indexed,library("decode1.metal",strict,"fast::exp"),library("decode1.metal",nil,"exp")};
 auto simple=[&](id<MTLLibrary> lib,const char* name){auto p=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:[NSString stringWithUTF8String:name]] error:&error];if(!p)throw std::runtime_error(error.localizedDescription.UTF8String);return p;};
 auto silu=simple(pointwise,"silu_mul"),pressurePipe=simple(pointwise,"decode_pressure");
 auto buffer=[&](size_t bytes){auto b=[device newBufferWithLength:bytes options:MTLResourceStorageModeShared];Need(b!=nil,"Allocation failed");return b;};
 auto finish=[&](id<MTLCommandBuffer> cb){[cb commit];[cb waitUntilCompleted];if(cb.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(cb.error.localizedDescription.UTF8String);};
 id<MTLBuffer> pressureData=nil;if(pressure){pressureData=buffer(268435456);std::fill_n((uint32_t*)pressureData.contents,pressureData.length/4,37);}
 // Method -1 is actual native scalar/blocked routing; 0/1/2 are compiler policies.
 using Key=std::tuple<size_t,size_t,bool,int,bool>;std::map<Key,id<MTLComputePipelineState>> cache;bool sealed=false;
 auto pipeline=[&](size_t n,size_t active,bool indices,int method,bool shadow){
  Key key{n,active,indices,method,shadow};auto found=cache.find(key);if(found!=cache.end())return found->second;Need(!sealed,"Late pipeline compilation");
  auto constants=[MTLFunctionConstantValues new];uint32_t rows=uint32_t(active),dims[]={4096,uint32_t(n),uint32_t(4096+(padded?3:0)),uint32_t(n+(padded?5:0))};[constants setConstantValue:&rows type:MTLDataTypeUInt atIndex:0];
  for(NSUInteger i=0;i<4;++i)[constants setConstantValue:dims+i type:MTLDataTypeUInt atIndex:10+i];
  bool blocked=active==4&&n==14336;id<MTLLibrary> lib=indices?indexed:ordinary;
  const char* name=indices?"affine_q4_indexed_decode":"affine_q4_decode";
  if(method<0&&blocked){lib=indexed;name="affine_q4_blocked";}
  if(method>=0){lib=candidates[method];name="fused_q4_decode";uint32_t columns=blocked?4:1;[constants setConstantValue:&columns type:MTLDataTypeUInt atIndex:21];[constants setConstantValue:&shadow type:MTLDataTypeBool atIndex:20];}
  auto f=[lib newFunctionWithName:[NSString stringWithUTF8String:name] constantValues:constants error:&error];if(!f)throw std::runtime_error(error.localizedDescription.UTF8String);
  auto p=[device newComputePipelineStateWithFunction:f error:&error];if(!p)throw std::runtime_error(error.localizedDescription.UTF8String);
  Need(p.threadExecutionWidth==32&&p.maxTotalThreadsPerThreadgroup>=128&&p.staticThreadgroupMemoryLength<=device.maxThreadgroupMemoryLength,"Pipeline bound exceeded");cache[key]=p;
  report["pipelines"].push_back({{"columns",n},{"active",active},{"indexed",indices},{"method",method},{"shadow",shadow},{"max_threads",p.maxTotalThreadsPerThreadgroup},{"simd_width",p.threadExecutionWidth},{"threadgroup_bytes",p.staticThreadgroupMemoryLength}});return p;
 };
 std::vector<size_t> columns=correctness?std::vector<size_t>{14336,1,7,17,33}:std::vector<size_t>{14336};
 for(size_t n:columns)for(size_t rows=1;rows<=4;++rows)for(bool indices:{false,true}){
  pipeline(n,rows,indices,-1,false);
  for(int method=0;method<3;++method)if(selected<0||method==selected)for(bool shadow:{false,true})pipeline(n,rows,indices,method,shadow);
 }
 sealed=true;report["pipelines_prepared_before_collection"]=true;report["pipeline_count"]=cache.size();
 size_t caseIndex=0,exactWords=0;std::array<size_t,3> mismatch{};
 std::vector<size_t> layers=correctness?std::vector<size_t>{0,15,31}:std::vector<size_t>{15};
 for(size_t layer:layers)for(size_t n:columns){@autoreleasepool{
  const size_t k=4096,aStride=k+(padded?3:0),oStride=n+(padded?5:0),xWords=32+4*aStride,yWords=32+4*oStride;const uint32_t G=0x1937abef;
  auto asset=fs::path(argv[2])/("layer-"+std::to_string(layer)),capture=fs::path(argv[3])/("layer-"+std::to_string(layer));
  auto input=Read<float>(capture/"norm-input.f32",512*k);
  std::array<std::vector<uint32_t>,2> weights;std::array<std::vector<uint16_t>,2> scales,biases;std::array<id<MTLBuffer>,6> wb;
  auto guarded=[&](const auto& values){auto b=buffer(values.size()*sizeof(values[0])+128);std::fill_n((uint32_t*)b.contents,b.length/4,G);std::memcpy((char*)b.contents+64,values.data(),values.size()*sizeof(values[0]));return b;};
  for(int role=0;role<2;++role){std::string name=role?"up":"gate";weights[role]=Read<uint32_t>(asset/(name+"-weight.u32"),14336*k/8);scales[role]=Read<uint16_t>(asset/(name+"-scales.f16"),14336*k/64);biases[role]=Read<uint16_t>(asset/(name+"-biases.f16"),14336*k/64);weights[role].resize(n*k/8);scales[role].resize(n*k/64);biases[role].resize(n*k/64);wb[role*3]=guarded(weights[role]);wb[role*3+1]=guarded(scales[role]);wb[role*3+2]=guarded(biases[role]);}
  auto x=buffer(xWords*4),gate=buffer(yWords*4),upper=buffer(yWords*4),out=buffer(yWords*4),shadowG=buffer(yWords*4),shadowU=buffer(yWords*4);
  std::fill_n((uint32_t*)x.contents,xWords,G);std::array<size_t,4> sourceRows{0,127,256,511};for(size_t row=0;row<4;++row)std::memcpy((float*)x.contents+16+row*aStride,input.data()+sourceRows[row]*k,k*4);
  std::vector<std::string> readOnly;readOnly.push_back(Sha(x.contents,x.length));for(auto b:wb)readOnly.push_back(Sha(b.contents,b.length));
  struct Selection{size_t physical,mask;bool indexed;};std::vector<Selection> selections;
  if(correctness){for(size_t rows=1;rows<=4;++rows)selections.push_back({rows,(1ul<<rows)-1,false});for(size_t mask=0;mask<16;++mask)selections.push_back({4,mask,true});}
  else selections={{4,1,true},{4,8,true},{4,15,true}};
  for(auto selection:selections){
   std::vector<uint32_t> ids;for(size_t row=0;row<selection.physical;++row)if(selection.mask&(1ul<<row))ids.push_back(uint32_t(row));
   double pressureStart=0,pressureEnd=0;
   auto reset=[&]{for(auto b:{gate,upper,out,shadowG,shadowU})std::fill_n((uint32_t*)b.contents,yWords,G);};
   auto encode=[&](int method,bool shadow){
    if(pressure){auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];[e setComputePipelineState:pressurePipe];[e setBuffer:pressureData offset:0 atIndex:0];[e dispatchThreads:MTLSizeMake(pressureData.length/4,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding];finish(cb);pressureStart=cb.GPUStartTime;pressureEnd=cb.GPUEndTime;}
    auto cb=[queue commandBuffer];bool sparse=selection.indexed&&ids.size()<selection.physical;
    if(sparse){auto b=[cb blitCommandEncoder];
     if(method<0){for(auto target:{gate,upper})[b fillBuffer:target range:NSMakeRange(64,((selection.physical-1)*oStride+n)*4) value:0];}
     else {for(size_t row=0;row<selection.physical;++row)[b fillBuffer:out range:NSMakeRange(64+row*oStride*4,n*4) value:0];if(shadow)for(auto target:{shadowG,shadowU})[b fillBuffer:target range:NSMakeRange(64,((selection.physical-1)*oStride+n)*4) value:0];}
     [b endEncoding];}
    if(!ids.empty()){
     Params params{ids.size(),k,n,aStride,oStride};bool blocked=ids.size()==4&&n==14336;size_t width=blocked?16:4;
     if(method<0)for(int role=0;role<2;++role){auto e=[cb computeCommandEncoder];[e setComputePipelineState:cache.at({n,ids.size(),selection.indexed,-1,false})];[e setBuffer:x offset:64 atIndex:0];for(int j=0;j<3;++j)[e setBuffer:wb[role*3+j] offset:64 atIndex:1+j];[e setBuffer:(role?upper:gate) offset:64 atIndex:4];[e setBytes:&params length:sizeof(params) atIndex:5];if(selection.indexed||blocked)[e setBytes:ids.data() length:ids.size()*4 atIndex:6];[e dispatchThreadgroups:MTLSizeMake((n+width-1)/width,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[e endEncoding];}
     else {auto e=[cb computeCommandEncoder];[e setComputePipelineState:cache.at({n,ids.size(),selection.indexed,method,shadow})];[e setBuffer:x offset:64 atIndex:0];for(int j=0;j<6;++j)[e setBuffer:wb[j] offset:64 atIndex:1+j];[e setBuffer:out offset:64 atIndex:7];[e setBytes:&params length:sizeof(params) atIndex:8];[e setBytes:ids.data() length:ids.size()*4 atIndex:9];[e setBuffer:shadowG offset:64 atIndex:10];[e setBuffer:shadowU offset:64 atIndex:11];[e dispatchThreadgroups:MTLSizeMake((n+width-1)/width,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[e endEncoding];}
    }
    if(method<0){Activation params{selection.physical,n,oStride,oStride,oStride,0,0};auto e=[cb computeCommandEncoder];[e setComputePipelineState:silu];[e setBuffer:gate offset:64 atIndex:0];[e setBuffer:upper offset:64 atIndex:1];[e setBuffer:out offset:64 atIndex:2];[e setBytes:&params length:sizeof(params) atIndex:3];[e dispatchThreads:MTLSizeMake(selection.physical*n,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[e endEncoding];}
    finish(cb);return cb;
   };
   if(correctness){
    reset();encode(-1,false);std::array<std::vector<uint32_t>,3> gold;
    int at=0;for(auto b:{out,gate,upper}){gold[at].resize(yWords);std::memcpy(gold[at].data(),b.contents,yWords*4);++at;}
    for(size_t i=0;i<yWords;++i){bool payload=i>=16&&i<16+selection.physical*oStride;size_t rel=i>=16?i-16:0,row=rel/oStride,col=rel%oStride;
     if(payload&&col<n){float value;std::memcpy(&value,gold[0].data()+i,4);Need(std::isfinite(value)&&gold[0][i]!=G,"Control output invalid");if(!(selection.mask&(1ul<<row)))Need(gold[0][i]==0,"Inactive product not zero");}
     else Need(gold[0][i]==G,"Control product guard/padding changed");}
    for(auto row:ids)for(size_t col:std::array<size_t,3>{0,n/2,n-1})for(int role=0;role<2;++role){double sum=0,absolute=0;for(size_t kk=0;kk<k;++kk){size_t ci=col*(k/64)+kk/64;uint32_t q=(weights[role][col*(k/8)+kk/8]>>((kk%8)*4))&15;float sw=Garnet::Float16ToFloat32(scales[role][ci]),bw=Garnet::Float16ToFloat32(biases[role][ci]),weight=std::fma(float(q),sw,bw);double product=double(((float*)x.contents)[16+row*aStride+kk])*double(weight);sum+=product;absolute+=std::abs(product);}float value;std::memcpy(&value,gold[role+1].data()+16+row*oStride+col,4);double fraction=std::abs(double(value)-sum)/(2e-6+4e-6*absolute);Need(fraction<=1,"Original FP64 bound failed");report["fp64"].push_back({{"case",caseIndex},{"layer",layer},{"columns",n},{"row",row},{"column",col},{"role",role},{"actual",value},{"sum",sum},{"sum_abs",absolute},{"fraction",fraction}});}
    std::array<std::string,3> controlNames;for(size_t role=0;role<3;++role){controlNames[role]="case"+std::to_string(caseIndex)+"-control"+std::to_string(role)+".u32";Save(dest/controlNames[role],gold[role].data(),yWords*4);}
    for(int method=0;method<3;++method)if(selected<0||method==selected)for(bool shadow:{false,true}){
     reset();encode(method,shadow);size_t different=0,shadowDifferent=0;const auto* got=(uint32_t*)out.contents;
     for(size_t i=0;i<yWords;++i){different+=got[i]!=gold[0][i];if(shadow)shadowDifferent+=((uint32_t*)shadowG.contents)[i]!=gold[1][i]||((uint32_t*)shadowU.contents)[i]!=gold[2][i];}
     mismatch[method]+=different+shadowDifferent;if(!different&&!shadowDifferent)exactWords+=yWords*(shadow?3:1);
     std::string raw="case"+std::to_string(caseIndex)+"-method"+std::to_string(method)+(shadow?"-shadow":"-normal")+".u32";Save(dest/raw,out.contents,yWords*4);
     J cell={{"case",caseIndex},{"layer",layer},{"columns",n},{"physical_rows",selection.physical},{"mask",selection.mask},{"indexed",selection.indexed},{"method",method},{"shadow",shadow},{"parent_words",yWords},{"product_mismatches",different},{"shadow_mismatches",shadowDifferent},{"product_sha256",Sha(out.contents,yWords*4)},{"control_product_sha256",Sha(gold[0].data(),yWords*4)},{"raw_product",raw},{"raw_controls",controlNames}};
     if(shadow){std::array<std::string,2> names;for(size_t role=0;role<2;++role){names[role]=raw+(role?"-up":"-gate");auto b=role?shadowU:shadowG;Save(dest/names[role],b.contents,yWords*4);}cell["raw_shadows"]=names;}
     report["cases"].push_back(cell);
    }
   }else {
    // Normal kernels only. Both policies first pass exact full-parent checks.
    reset();encode(-1,false);std::vector<uint32_t> expected(yWords);std::memcpy(expected.data(),out.contents,yWords*4);reset();encode(selected,false);Need(!std::memcmp(expected.data(),out.contents,yWords*4),"Timing candidate differs from native control");
    for(int round=-5;round<rounds;++round)for(int position=0;position<2;++position){int which=(position+(round+5)%2)%2;if(((round+5)+order)&1)which=1-which;int method=which?selected:-1;
     auto began=std::chrono::steady_clock::now();auto cb=encode(method,false);double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-began).count(),gpu=(cb.GPUEndTime-cb.GPUStartTime)*1000;Need(gpu>0&&std::isfinite(gpu)&&wall>=gpu,"Invalid command clock");Need(!std::memcmp(expected.data(),out.contents,yWords*4),"Timed output changed");
     if(round>=0)report["samples"].push_back({{"case",caseIndex},{"layer",layer},{"mask",selection.mask},{"method",method},{"round",round},{"position",position},{"gpu_ms",gpu},{"wall_including_pressure_ms",wall},{"gpu_start_seconds",cb.GPUStartTime},{"gpu_end_seconds",cb.GPUEndTime},{"pressure_gpu_start_seconds",pressureStart},{"pressure_gpu_end_seconds",pressureEnd}});
    }
   }++caseIndex;
  }
  Need(Sha(x.contents,x.length)==readOnly[0],"Input parent changed");for(size_t role=0;role<6;++role)Need(Sha(wb[role].contents,wb[role].length)==readOnly[role+1],"Weight parent/guards changed");
  report["readonly_checks"].push_back({{"layer",layer},{"columns",n},{"sha256",readOnly},{"unchanged",true}});
  Report(dest/"result1.json",report);std::cout<<J{{"layer",layer},{"columns",n},{"cases",report["cases"].size()},{"samples",report["samples"].size()}}.dump()<<'\n'<<std::flush;
 }}
 report.update({{"case_combinations",caseIndex},{"exact_parent_words",exactWords},{"mismatches_by_method",mismatch},{"maximum_device_allocated_bytes_before_close",device.currentAllocatedSize}});
 }}
 size_t after=device.currentAllocatedSize;Need(after<=baseline+4194304,"Owned GPU allocation retained");report.update({{"completed",true},{"after_close_allocated_bytes",after},{"no_native_runtime_integration",true},{"serving_speed_claim",false},{"goal_complete",false}});Report(dest/"result1.json",report);std::cout<<J{{"completed",true},{"cases",report["cases"].size()},{"mismatches",report["mismatches_by_method"]}}.dump()<<'\n';return 0;
}}catch(const std::exception& error){report["error"]=error.what();if(!dest.empty()&&fs::exists(dest))Report(dest/"result1.json",report);std::cerr<<error.what()<<'\n';return 1;}}
