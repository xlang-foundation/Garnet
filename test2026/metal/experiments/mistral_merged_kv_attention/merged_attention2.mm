// SPDX-License-Identifier: Apache-2.0
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <nlohmann/json.hpp>
#include "chunk_mps_contiguous_policy.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>
using json=nlohmann::json;namespace fs=std::filesystem;
struct Params{uint64_t batch,chunk,heads,kvHeads,dim,capacity,qStride,kStride,vStride,kcStride,vcStride,sStride,aStride,gStride,oStride,gated;};
struct Pitch{uint32_t packed,cacheScore,liveScore;};
struct Controls{uint32_t first[4]{},count[4]{};};
constexpr size_t guard=16;constexpr uint32_t poison=0x7fc12345;
void need(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
float word(uint32_t u){float f;std::memcpy(&f,&u,4);return f;}
uint32_t bits(float f){uint32_t u;std::memcpy(&u,&f,4);return u;}
float value(size_t i,uint32_t salt){uint32_t x=uint32_t(i)^salt;x^=x>>16;x*=0x7feb352dU;x^=x>>15;x*=0x846ca68bU;x^=x>>16;return float(int(x%20001)-10000)*.00003f;}
void save(const fs::path& p,const void* a,size_t n){std::ofstream f(p,std::ios::binary);f.write((const char*)a,n);need(bool(f),"Cannot save payload");}
std::vector<float> load(const fs::path& p){std::ifstream f(p,std::ios::binary|std::ios::ate);need(bool(f),"Missing retained input");size_t n=f.tellg();need(n&&n%4==0,"Bad input size");std::vector<float> a(n/4);f.seekg(0);f.read((char*)a.data(),n);need(bool(f),"Cannot read payload");return a;}
json readJson(const fs::path& p){json j;std::ifstream f(p);f>>j;return j;}
struct Pipelines{
 id<MTLDevice> device;id<MTLCommandQueue> queue;id<MTLLibrary> library;
 id<MTLComputePipelineState> pack,unpack,mergedUnpack;
 struct Bound{id<MTLComputePipelineState> mergedPack;std::array<id<MTLComputePipelineState>,3> softmax;};std::map<size_t,Bound> bounds;
 Pipelines(const char* path){device=MTLCreateSystemDefaultDevice();queue=[device newCommandQueue];need(device&&queue,"No Metal device/queue");NSError* error=nil;
  auto src=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:path] encoding:NSUTF8StringEncoding error:&error];need(src!=nil,"Missing source");auto opts=[MTLCompileOptions new];opts.languageVersion=MTLLanguageVersion4_0;opts.mathMode=MTLMathModeSafe;
  library=[device newLibraryWithSource:src options:opts error:&error];if(!library)throw std::runtime_error(error.localizedDescription.UTF8String);
  auto compile=[&](NSString* name,size_t bound){NSError* e=nil;id<MTLFunction> fn;
   if(bound){auto c=[MTLFunctionConstantValues new];uint32_t extent=uint32_t(bound);[c setConstantValue:&extent type:MTLDataTypeUInt atIndex:31];fn=[library newFunctionWithName:name constantValues:c error:&e];}else fn=[library newFunctionWithName:name];
   if(!fn)throw std::runtime_error(e?e.localizedDescription.UTF8String:"Missing Metal function");auto p=[device newComputePipelineStateWithFunction:fn error:&e];if(!p)throw std::runtime_error(e.localizedDescription.UTF8String);need(p.threadExecutionWidth==32&&p.maxTotalThreadsPerThreadgroup>=128,"Bad pipeline geometry");return p;};
  pack=compile(@"chunk_mps_pack",0);unpack=compile(@"chunk_mps_unpack",0);mergedUnpack=compile(@"chunk_mps_merged_unpack",0);
  for(size_t e:{128,256,2048,4096})bounds[e]={compile(@"chunk_mps_merged_pack",e),{compile(@"chunk_mps_softmax",e),compile(@"chunk_mps_softmax_contiguous",e),compile(@"chunk_mps_merged_softmax",e)}};
 }
};
struct Cell{
 Pipelines& gpu;size_t index,batch,chunk,capacity,extent,rows,queries,live;Params p;Pitch pitch;Controls controls;
 std::array<std::vector<float>,8> input;std::array<id<MTLBuffer>,8> buffers;
 id<MTLBuffer> qp,kp,vp,score,liveScore,prob,part,mk,mv,out;
 MPSMatrix *queryView,*keyView,*valueView,*cacheKey,*cacheValue,*scoreView,*liveScoreView,*probView,*partView,*mergedKey,*mergedValue;
 MPSMatrixMultiplication *qk,*ql,*pv,*pl;std::vector<float> blank;std::array<std::vector<float>,3> expected;
 size_t checkedWords=0,readonlyWords=0;bool gated;
 Cell(Pipelines& g,size_t ci,const json& shape,const fs::path& fixture):gpu(g),index(ci){batch=shape["batch"];chunk=shape["chunk"];capacity=shape["capacity"];extent=shape["extent"];gated=shape["gated"];rows=batch*chunk;queries=batch*32*chunk;live=batch*8*chunk;
  p={batch,chunk,32,8,128,capacity,4096+4,1024+4,1024+4,8*capacity*128,8*capacity*128,chunk+3,chunk+5,4096+4,4096+4,uint64_t(gated)};
  pitch={128,uint32_t([MPSMatrixDescriptor rowBytesFromColumns:capacity dataType:MPSDataTypeFloat32]/4),uint32_t([MPSMatrixDescriptor rowBytesFromColumns:chunk dataType:MPSDataTypeFloat32]/4)};
  const size_t lengths[]={rows*p.qStride,rows*p.kStride,rows*p.vStride,batch*p.kcStride,batch*p.vcStride,batch*p.sStride,batch*p.aStride,rows*p.gStride};
  for(size_t i=0;i<8;++i){if(fixture.empty()){
    input[i].assign(lengths[i]+2*guard,word(poison));
    if(i<3||i==7){size_t logical=i==0||i==7?4096:1024,stride=i==0?p.qStride:i==1?p.kStride:i==2?p.vStride:p.gStride;
     for(size_t r=0;r<rows;++r)for(size_t d=0;d<logical;++d)input[i][guard+r*stride+d]=value(r*logical+d,17+37*i);}
    if(i==3||i==4)for(size_t n=0;n<lengths[i];++n)input[i][guard+n]=value(n,17+37*i);
   }else input[i]=load(fixture/("case"+std::to_string(ci)+"-input"+std::to_string(i)+".f32"));need(input[i].size()==lengths[i]+2*guard,"Retained input dimensions differ");}
  if(fixture.empty())for(size_t b=0;b<batch;++b){size_t first=shape["first"][b],count=shape["counts"][b];for(size_t t=0;t<chunk;++t){input[5][guard+b*p.sStride+t]=float(first+t);input[6][guard+b*p.aStride+t]=float(t<count);}}
  const float* slots=input[5].data()+guard;const float* active=input[6].data()+guard;
  need(Garnet::ChunkMPSContiguousPrefix(batch,chunk,capacity,slots,p.sStride,active,p.aStride),"Noncontiguous/invalid controls admitted");
  for(size_t b=0;b<batch;++b){while(controls.count[b]<chunk&&active[b*p.aStride+controls.count[b]]!=0)++controls.count[b];if(controls.count[b])controls.first[b]=uint32_t(slots[b*p.sStride]);need(controls.first[b]+controls.count[b]<=extent,"Context bucket too short");}
  auto altered=input[5];altered[guard]=NAN;need(!Garnet::ChunkMPSContiguousPrefix(batch,chunk,capacity,altered.data()+guard,p.sStride,active,p.aStride),"Invalid slot not rejected");
  auto alloc=[&](size_t n){std::vector<float> f(n+2*guard,word(poison));auto b=[gpu.device newBufferWithBytes:f.data() length:f.size()*4 options:MTLResourceStorageModeShared];need(b!=nil,"Buffer allocation failed");return b;};
  for(size_t i=0;i<8;++i){buffers[i]=[gpu.device newBufferWithBytes:input[i].data() length:input[i].size()*4 options:MTLResourceStorageModeShared];need(buffers[i]!=nil,"Input allocation failed");readonlyWords+=input[i].size();}
  qp=alloc(queries*128);kp=alloc(live*128);vp=alloc(live*128);score=alloc(queries*pitch.cacheScore);liveScore=alloc(queries*pitch.liveScore);prob=alloc(queries*pitch.liveScore);part=alloc(queries*128);
  mk=alloc(batch*8*capacity*128);mv=alloc(batch*8*capacity*128);blank.assign(rows*p.oStride+2*guard,word(poison));out=alloc(rows*p.oStride);
  auto view=[&](id<MTLBuffer> b,size_t r,size_t c,size_t stride){auto d=[MPSMatrixDescriptor matrixDescriptorWithRows:r columns:c matrices:batch*8 rowBytes:stride*4 matrixBytes:r*stride*4 dataType:MPSDataTypeFloat32];auto m=[[MPSMatrix alloc] initWithBuffer:b offset:guard*4 descriptor:d];need(m!=nil,"MPS view failed");return m;};
  size_t qr=4*chunk;queryView=view(qp,qr,128,128);keyView=view(kp,chunk,128,128);valueView=view(vp,chunk,128,128);cacheKey=view(buffers[3],capacity,128,128);cacheValue=view(buffers[4],capacity,128,128);
  scoreView=view(score,qr,capacity,pitch.cacheScore);liveScoreView=view(liveScore,qr,chunk,pitch.liveScore);probView=view(prob,qr,chunk,pitch.liveScore);partView=view(part,qr,128,128);mergedKey=view(mk,capacity,128,128);mergedValue=view(mv,capacity,128,128);
  auto product=[&](size_t n,size_t k,bool transpose){auto op=[[MPSMatrixMultiplication alloc] initWithDevice:gpu.device transposeLeft:NO transposeRight:transpose resultRows:qr resultColumns:n interiorColumns:k alpha:1 beta:0];op.batchSize=batch*8;need(op!=nil,"MPS product failed");return op;};
  qk=product(extent,128,true);ql=product(chunk,128,true);pv=product(128,extent,false);pl=product(128,chunk,false);
  if(!fixture.empty())for(int v=0;v<3;++v)expected[v]=load(fixture/("case"+std::to_string(ci)+"-output"+std::to_string(v)+".f32"));
 }
 std::array<double,2> run(int variant){std::memcpy(out.contents,blank.data(),blank.size()*4);auto began=std::chrono::steady_clock::now();auto command=[gpu.queue commandBuffer];need(command!=nil,"No command buffer");
  auto encoder=[command computeCommandEncoder];auto bind=[&](id<MTLBuffer> b,size_t i){[encoder setBuffer:b offset:guard*4 atIndex:i];};
  if(variant<2){[encoder setComputePipelineState:gpu.pack];bind(buffers[0],0);bind(buffers[1],1);bind(buffers[2],2);bind(buffers[6],3);bind(qp,4);bind(kp,5);bind(vp,6);[encoder setBytes:&p length:sizeof(p) atIndex:7];[encoder setBytes:&pitch length:sizeof(pitch) atIndex:8];}
  else{[encoder setComputePipelineState:gpu.bounds.at(extent).mergedPack];for(size_t i=0;i<5;++i)bind(buffers[i],i);bind(buffers[6],5);bind(qp,6);bind(mk,7);bind(mv,8);[encoder setBytes:&p length:sizeof(p) atIndex:9];[encoder setBytes:&controls length:sizeof(controls) atIndex:10];}
  size_t threads=variant<2?queries*128:std::max(queries*128,batch*8*extent*128);[encoder dispatchThreads:MTLSizeMake(threads,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[encoder endEncoding];
  [qk encodeToCommandBuffer:command leftMatrix:queryView rightMatrix:variant<2?cacheKey:mergedKey resultMatrix:scoreView];
  if(variant<2)[ql encodeToCommandBuffer:command leftMatrix:queryView rightMatrix:keyView resultMatrix:liveScoreView];
  encoder=[command computeCommandEncoder];[encoder setComputePipelineState:gpu.bounds.at(extent).softmax[variant]];
  if(variant<2){bind(score,0);bind(liveScore,1);bind(prob,2);bind(buffers[5],3);bind(buffers[6],4);[encoder setBytes:&p length:sizeof(p) atIndex:5];[encoder setBytes:&pitch length:sizeof(pitch) atIndex:6];if(!variant)[encoder setThreadgroupMemoryLength:(extent*2+15)&~size_t(15) atIndex:1];}
  else{bind(score,0);bind(buffers[5],1);bind(buffers[6],2);[encoder setBytes:&p length:sizeof(p) atIndex:3];[encoder setBytes:&pitch length:sizeof(pitch) atIndex:4];}
  [encoder setThreadgroupMemoryLength:(extent*4+15)&~size_t(15) atIndex:0];[encoder dispatchThreadgroups:MTLSizeMake(queries,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[encoder endEncoding];
  [pv encodeToCommandBuffer:command leftMatrix:scoreView rightMatrix:variant<2?cacheValue:mergedValue resultMatrix:queryView];
  if(variant<2)[pl encodeToCommandBuffer:command leftMatrix:probView rightMatrix:valueView resultMatrix:partView];
  encoder=[command computeCommandEncoder];[encoder setComputePipelineState:variant<2?gpu.unpack:gpu.mergedUnpack];bind(qp,0);
  if(variant<2){bind(part,1);bind(buffers[7],2);bind(buffers[6],3);bind(out,4);[encoder setBytes:&p length:sizeof(p) atIndex:5];[encoder setBytes:&pitch length:sizeof(pitch) atIndex:6];}
  else{bind(buffers[7],1);bind(buffers[6],2);bind(out,3);[encoder setBytes:&p length:sizeof(p) atIndex:4];}
  [encoder dispatchThreads:MTLSizeMake(queries*128,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[encoder endEncoding];[command commit];[command waitUntilCompleted];
  double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-began).count(),ms=1000*(command.GPUEndTime-command.GPUStartTime);need(command.status==MTLCommandBufferStatusCompleted,"GPU command failed");need(std::isfinite(ms)&&ms>0,"Invalid GPU span");return {ms,wall};
 }
 void check(bool compare,int variant){const float* a=(const float*)out.contents;size_t guards=0,inactive=0;
  for(size_t i=0;i<blank.size();++i){bool logical=i>=guard&&i<guard+rows*p.oStride&&(i-guard)%p.oStride<4096;
   if(!logical){need(bits(a[i])==poison,"Output parent guard changed");++guards;}else{need(std::isfinite(a[i]),"Nonfinite output");size_t row=(i-guard)/p.oStride;if(input[6][guard+(row/chunk)*p.aStride+row%chunk]==0){need(a[i]==0,"Inactive output changed");++inactive;}}}
  if(compare)need(expected[variant].size()==blank.size()&&!std::memcmp(a,expected[variant].data(),blank.size()*4),"Timed output differs from retained variant");
  for(size_t i=0;i<8;++i)need(!std::memcmp(buffers[i].contents,input[i].data(),input[i].size()*4),"Readonly input parent changed");
  for(auto b:{mk,mv}){const float* a=(const float*)b.contents;for(size_t i=0;i<2*guard+batch*8*capacity*128;++i){bool logical=i>=guard&&i<guard+batch*8*capacity*128&&((i-guard)/128)%capacity<extent;if(!logical)need(bits(a[i])==poison,"Merged KV guard/tail changed");}}
  checkedWords+=blank.size()+readonlyWords;
 }
 json metadata(const json& shape){json j=shape;j["params"]=std::vector<uint64_t>((uint64_t*)&p,(uint64_t*)&p+16);j["pitches"]={pitch.packed,pitch.cacheScore,pitch.liveScore};j["readonly_parent_words"]=readonlyWords;j["output_parent_words"]=blank.size();j["temporary_merged_kv_bytes"]=2*(batch*8*capacity*128+2*guard)*4;return j;}
};
int main(int argc,char** argv){@autoreleasepool{try{
 need(argc==5,"Usage: helper combined.metal '-'|audited-fixtures fresh-output correctness|screen");fs::path fixture=std::string(argv[2])=="-"?fs::path():fs::path(argv[2]),out=argv[3];bool screen=std::string(argv[4])=="screen";need(screen||std::string(argv[4])=="correctness","Bad mode");need(!fs::exists(out),"Fresh output required");fs::create_directories(out);need(screen==!fixture.empty(),"Screen needs saved audited fixtures");
 json shapes=json::array();auto add=[&](size_t b,size_t t,size_t c,size_t f,size_t e,bool gated,bool ragged){std::vector<size_t> first(b,f),counts(b,t);if(ragged){counts={t,t-3,1,0};first={f,f,0,f};}shapes.push_back({{"batch",b},{"chunk",t},{"capacity",c},{"extent",e},{"first",first},{"counts",counts},{"gated",gated}});};
 add(1,8,128,0,128,false,false);add(4,17,256,128,256,true,true);add(1,128,4096,1920,2048,false,false);add(4,128,4096,1920,2048,false,false);add(4,128,4096,3840,4096,true,true);add(4,128,4096,3840,4096,false,false);add(4,128,4096,0,128,false,false);
 if(screen){auto report=readJson(fixture/"correctness1.json");need(report["collected"]==true&&report["cases"].size()==shapes.size(),"Missing correctness collection");need(readJson(fixture/"audit1.json")["passed"]==true,"Independent correctness audit required");}
 Pipelines gpu(argv[1]);std::vector<std::unique_ptr<Cell>> cells;for(size_t i=0;i<shapes.size();++i)cells.push_back(std::make_unique<Cell>(gpu,i,shapes[i],fixture));
 json report={{"scope","Standalone three-variant full-row FP32 attention operator; original literal/control MPS composition versus read-only merged-KV2products. No native inference/serving admission."},{"cases",json::array()},{"device",gpu.device.name.UTF8String},{"all_prepared_before_measurement",true}};
 if(screen)for(auto& c:cells)for(int warm=0;warm<3;++warm)for(int v=0;v<3;++v){c->run(v);c->check(true,v);} // all shapes/pipelines warmed before first measurement
 for(size_t i=0;i<cells.size();++i){auto& c=*cells[i];json meta=c.metadata(shapes[i]);if(!screen){for(size_t k=0;k<8;++k)save(out/("case"+std::to_string(i)+"-input"+std::to_string(k)+".f32"),c.input[k].data(),c.input[k].size()*4);
   for(int v=0;v<3;++v){c.run(v);c.check(false,v);save(out/("case"+std::to_string(i)+"-output"+std::to_string(v)+".f32"),c.out.contents,c.blank.size()*4);}
   save(out/("case"+std::to_string(i)+"-merged-k.f32"),c.mk.contents,c.mk.length);
   save(out/("case"+std::to_string(i)+"-merged-v.f32"),c.mv.contents,c.mv.length);
  }else{meta["samples"]=json::array();for(int round=0;round<9;++round)for(int order=0;order<3;++order){int v=(round+order)%3;auto t=c.run(v); // verification/readback is after captured span
    const auto& expected=c.expected[v];need(!std::memcmp(c.out.contents,expected.data(),expected.size()*4),"Timed result changed");meta["samples"].push_back({{"round",round},{"order",order},{"variant",v},{"gpu_ms",t[0]},{"wall_ms",t[1]}});}
   for(int v=0;v<3;++v)c.check(false,v); // full read-only parent/guard audit after this cell's timing
  }
  meta["checked_parent_words"]=c.checkedWords;report["cases"].push_back(meta);std::cout<<"Completed cell "<<i<<' '<<(screen?"screen":"correctness")<<'\n'<<std::flush;
 }
 report["collected"]=true;report["native_integration"]=false;report["serving_speed_claim"]=false;report["goal_complete"]=false;std::ofstream(out/(screen?"screen1.json":"correctness1.json"))<<report.dump(2)<<'\n';return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}}
