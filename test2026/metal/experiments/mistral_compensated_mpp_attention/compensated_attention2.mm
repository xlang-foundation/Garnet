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
 id<MTLComputePipelineState> pack,unpack,mergedUnpack,queryOnly,layout;std::array<id<MTLComputePipelineState>,2> compensated;std::vector<uint32_t> layoutWords,coordinateWords,registerLayoutWords;
 struct Bound{id<MTLComputePipelineState> mergedPack;std::array<id<MTLComputePipelineState>,3> softmax;};std::map<size_t,Bound> bounds;
 Pipelines(const char* path){device=MTLCreateSystemDefaultDevice();queue=[device newCommandQueue];need(device&&queue,"No Metal device/queue");NSError* error=nil;
  auto src=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:path] encoding:NSUTF8StringEncoding error:&error];need(src!=nil,"Missing source");auto opts=[MTLCompileOptions new];opts.languageVersion=MTLLanguageVersion4_0;opts.mathMode=MTLMathModeSafe;
  library=[device newLibraryWithSource:src options:opts error:&error];if(!library)throw std::runtime_error(error.localizedDescription.UTF8String);
  auto compile=[&](NSString* name,size_t bound){NSError* e=nil;id<MTLFunction> fn;
   if(bound){auto c=[MTLFunctionConstantValues new];uint32_t extent=uint32_t(bound);[c setConstantValue:&extent type:MTLDataTypeUInt atIndex:31];fn=[library newFunctionWithName:name constantValues:c error:&e];}else fn=[library newFunctionWithName:name];
   if(!fn)throw std::runtime_error(e?e.localizedDescription.UTF8String:"Missing Metal function");auto p=[device newComputePipelineStateWithFunction:fn error:&e];if(!p)throw std::runtime_error(e.localizedDescription.UTF8String);need(p.threadExecutionWidth==32&&p.maxTotalThreadsPerThreadgroup>=128,"Bad pipeline geometry");return p;};
  pack=compile(@"chunk_mps_pack",0);unpack=compile(@"chunk_mps_unpack",0);mergedUnpack=compile(@"chunk_mps_merged_unpack",0);
  queryOnly=compile(@"strict_mpp_query_pack",0);compensated[0]=compile(@"compensated_mpp_three",0);compensated[1]=compile(@"compensated_mpp_four",0);layout=compile(@"compensated_mpp_capacities",0);
  auto lb=[device newBufferWithLength:192*4 options:MTLResourceStorageModeShared];need(lb!=nil,"Layout allocation failed");auto cb=[queue commandBuffer];auto enc=[cb computeCommandEncoder];[enc setComputePipelineState:layout];[enc setBuffer:lb offset:0 atIndex:0];[enc dispatchThreads:MTLSizeMake(32,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[enc endEncoding];[cb commit];[cb waitUntilCompleted];need(cb.status==MTLCommandBufferStatusCompleted,"Layout GPU command failed");layoutWords.assign((uint32_t*)lb.contents,(uint32_t*)lb.contents+192);
  auto coordinate=compile(@"compensated_mpp_coordinates",0);std::vector<uint32_t> init(2561+32,0);for(size_t i=0;i<16;++i)init[i]=init[init.size()-1-i]=poison;auto coords=[device newBufferWithBytes:init.data() length:init.size()*4 options:MTLResourceStorageModeShared];need(coords!=nil,"Coordinate allocation failed");cb=[queue commandBuffer];enc=[cb computeCommandEncoder];[enc setComputePipelineState:coordinate];[enc setBuffer:coords offset:16*4 atIndex:0];[enc dispatchThreads:MTLSizeMake(32,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[enc endEncoding];[cb commit];[cb waitUntilCompleted];need(cb.status==MTLCommandBufferStatusCompleted,"Coordinate GPU command failed");coordinateWords.assign((uint32_t*)coords.contents,(uint32_t*)coords.contents+init.size());for(size_t i=0;i<16;++i)need(coordinateWords[i]==poison&&coordinateWords[coordinateWords.size()-1-i]==poison,"Coordinate parent guard changed");need(coordinateWords[16+2560]==0,"Coordinate out of range");for(size_t i=0;i<2560;++i){uint32_t n=coordinateWords[16+i];need(n>=1,"Coordinate missing element");if((i>=768&&i<1280)||i>=2048)need(n==1,"Destination coordinate has duplicate writers");}
  auto registerLayout=compile(@"compensated_mpp_layout",0);auto rg=[device newBufferWithLength:32*4 options:MTLResourceStorageModeShared];need(rg!=nil,"Register guard allocation failed");cb=[queue commandBuffer];enc=[cb computeCommandEncoder];[enc setComputePipelineState:registerLayout];[enc setBuffer:rg offset:0 atIndex:0];[enc dispatchThreads:MTLSizeMake(32,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[enc endEncoding];[cb commit];[cb waitUntilCompleted];need(cb.status==MTLCommandBufferStatusCompleted,"Register-layout GPU command failed");for(size_t i=0;i<32;++i){registerLayoutWords.push_back(((uint32_t*)rg.contents)[i]);need(registerLayoutWords.back()==0,"Strict FP32 register mapping differs from qualified map");}
  for(size_t e:{128,256,2048,4096})bounds[e]={compile(@"chunk_mps_merged_pack",e),{compile(@"chunk_mps_softmax",e),compile(@"chunk_mps_softmax_contiguous",e),compile(@"chunk_mps_merged_softmax",e)}};
 }
};
struct Cell{
 Pipelines& gpu;size_t index,batch,chunk,capacity,extent,rows,queries,live;Params p;Pitch pitch;Controls controls;
 std::array<std::vector<float>,8> input;std::array<id<MTLBuffer>,8> buffers;
 id<MTLBuffer> qp,kp,vp,score,liveScore,prob,part,mk,mv,out;
 MPSMatrix *queryView,*keyView,*valueView,*cacheKey,*cacheValue,*scoreView,*liveScoreView,*probView,*partView,*mergedKey,*mergedValue;
 MPSMatrixMultiplication *qk,*ql,*pv,*pl;std::vector<float> blank;std::array<std::vector<float>,4> expected;std::array<std::vector<float>,2> committed;
 size_t checkedWords=0,readonlyWords=0;bool gated;
 Cell(Pipelines& g,size_t ci,const json& shape,const fs::path& fixture,bool expectedOutputs=true):gpu(g),index(ci){batch=shape["batch"];chunk=shape["chunk"];capacity=shape["capacity"];extent=shape["extent"];gated=shape["gated"];rows=batch*chunk;queries=batch*32*chunk;live=batch*8*chunk;
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
  if(fixture.empty()&&ci>=7){
   for(size_t row=0;row<rows;++row){size_t t=row%chunk;
    for(size_t d=0;d<4096;++d)input[0][guard+row*p.qStride+d]=(ci==7)?1.00048828125f:float(d%128==0);
    for(size_t d=0;d<1024;++d){input[1][guard+row*p.kStride+d]=(ci==7)?(t%2?1.0f:1.00048828125f):(d%128==0?-float((ci==8?20:22)*std::sqrt(128.0)):0);
     input[2][guard+row*p.vStride+d]=(ci==7)?(t%2?-30.0f:30.0f):float(ci==8?1024:2048);}}
   for(size_t b=0;b<batch;++b)for(size_t head=0;head<8;++head)for(size_t t=0;t<capacity;++t)for(size_t d=0;d<128;++d){size_t at=guard+((b*8+head)*capacity+t)*128+d;
    input[3][at]=(ci==7||t==0||d)?0:-float((ci==8?20:22)*std::sqrt(128.0));input[4][at]=(ci==7||t==0)?0:float(ci==8?1024:2048);}
  }
  // CPU research eligibility check; no unsupported-input fallback is claimed.
  for(size_t i=0;i<5;++i){size_t width=i==0?4096:(i<3?1024:size_t(p.kcStride)),stride=i==0?p.qStride:i==1?p.kStride:i==2?p.vStride:p.kcStride,nrows=i<3?rows:batch;
   for(size_t row=0;row<nrows;++row)for(size_t d=0;d<width;++d){float x=input[i][guard+row*stride+d];need(std::isfinite(x)&&std::abs(x)<=2048,"Compensated input outside explicit finite2048 prototype domain");}}
  const float* slots=input[5].data()+guard;const float* active=input[6].data()+guard;
  need(Garnet::ChunkMPSContiguousPrefix(batch,chunk,capacity,slots,p.sStride,active,p.aStride),"Noncontiguous/invalid controls admitted");
  for(size_t b=0;b<batch;++b){while(controls.count[b]<chunk&&active[b*p.aStride+controls.count[b]]!=0)++controls.count[b];if(controls.count[b])controls.first[b]=uint32_t(slots[b*p.sStride]);need(controls.first[b]+controls.count[b]<=extent,"Context bucket too short");}
  auto altered=input[5];altered[guard]=NAN;need(!Garnet::ChunkMPSContiguousPrefix(batch,chunk,capacity,altered.data()+guard,p.sStride,active,p.aStride),"Invalid slot not rejected");
  auto alloc=[&](size_t n){std::vector<float> f(n+2*guard,word(poison));auto b=[gpu.device newBufferWithBytes:f.data() length:f.size()*4 options:MTLResourceStorageModeShared];need(b!=nil,"Buffer allocation failed");return b;};
  for(size_t i=0;i<8;++i){buffers[i]=[gpu.device newBufferWithBytes:input[i].data() length:input[i].size()*4 options:MTLResourceStorageModeShared];need(buffers[i]!=nil,"Input allocation failed");readonlyWords+=input[i].size();}
  qp=alloc(queries*128);kp=alloc(live*128);vp=alloc(live*128);score=alloc(queries*pitch.cacheScore);liveScore=alloc(queries*pitch.liveScore);prob=alloc(queries*pitch.liveScore);part=alloc(queries*128);
  mk=alloc(batch*8*capacity*128);mv=alloc(batch*8*capacity*128);blank.assign(rows*p.oStride+2*guard,word(poison));out=alloc(rows*p.oStride);
  for(size_t part=0;part<2;++part){auto& a=committed[part];a.assign(batch*8*capacity*128+2*guard,word(poison));for(size_t b=0;b<batch;++b)for(size_t h=0;h<8;++h)for(size_t t=0;t<extent;++t)for(size_t d=0;d<128;++d){float x=0;size_t first=controls.first[b],count=controls.count[b];if(count&&t<first)x=input[3+part][guard+b*p.kcStride+(h*capacity+t)*128+d];else if(count&&t<first+count)x=input[1+part][guard+(b*chunk+t-first)*(part?p.vStride:p.kStride)+h*128+d];a[guard+((b*8+h)*capacity+t)*128+d]=x;}std::memcpy(part?mv.contents:mk.contents,a.data(),a.size()*4);}
  auto view=[&](id<MTLBuffer> b,size_t r,size_t c,size_t stride){auto d=[MPSMatrixDescriptor matrixDescriptorWithRows:r columns:c matrices:batch*8 rowBytes:stride*4 matrixBytes:r*stride*4 dataType:MPSDataTypeFloat32];auto m=[[MPSMatrix alloc] initWithBuffer:b offset:guard*4 descriptor:d];need(m!=nil,"MPS view failed");return m;};
  size_t qr=4*chunk;queryView=view(qp,qr,128,128);keyView=view(kp,chunk,128,128);valueView=view(vp,chunk,128,128);cacheKey=view(buffers[3],capacity,128,128);cacheValue=view(buffers[4],capacity,128,128);
  scoreView=view(score,qr,capacity,pitch.cacheScore);liveScoreView=view(liveScore,qr,chunk,pitch.liveScore);probView=view(prob,qr,chunk,pitch.liveScore);partView=view(part,qr,128,128);mergedKey=view(mk,capacity,128,128);mergedValue=view(mv,capacity,128,128);
  auto product=[&](size_t n,size_t k,bool transpose){auto op=[[MPSMatrixMultiplication alloc] initWithDevice:gpu.device transposeLeft:NO transposeRight:transpose resultRows:qr resultColumns:n interiorColumns:k alpha:1 beta:0];op.batchSize=batch*8;need(op!=nil,"MPS product failed");return op;};
  qk=product(extent,128,true);ql=product(chunk,128,true);pv=product(128,extent,false);pl=product(128,chunk,false);
  if(!fixture.empty()&&expectedOutputs)for(int v=0;v<4;++v)expected[v]=load(fixture/("case"+std::to_string(ci)+"-output"+std::to_string(v)+".f32"));
 }
 std::array<double,2> run(int variant){std::memcpy(out.contents,blank.data(),blank.size()*4);auto began=std::chrono::steady_clock::now();auto command=[gpu.queue commandBuffer];need(command!=nil,"No command buffer");
  if(variant>=2){auto enc=[command computeCommandEncoder];[enc setComputePipelineState:gpu.compensated[variant-2]];[enc setBuffer:buffers[0] offset:guard*4 atIndex:0];[enc setBuffer:mk offset:guard*4 atIndex:1];[enc setBuffer:mv offset:guard*4 atIndex:2];[enc setBuffer:buffers[7] offset:guard*4 atIndex:3];[enc setBuffer:out offset:guard*4 atIndex:4];[enc setBytes:&p length:sizeof(p) atIndex:5];[enc setBytes:&controls length:sizeof(controls) atIndex:6];[enc dispatchThreadgroups:MTLSizeMake(32,(chunk+63)/64,batch) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[enc endEncoding];[command commit];[command waitUntilCompleted];double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-began).count(),ms=1000*(command.GPUEndTime-command.GPUStartTime);need(command.status==MTLCommandBufferStatusCompleted,"Strict MPP GPU command failed");need(std::isfinite(ms)&&ms>0,"Invalid strict MPP GPU span");return {ms,wall};}
  auto encoder=[command computeCommandEncoder];auto bind=[&](id<MTLBuffer> b,size_t i){[encoder setBuffer:b offset:guard*4 atIndex:i];};
  if(variant==0){[encoder setComputePipelineState:gpu.pack];bind(buffers[0],0);bind(buffers[1],1);bind(buffers[2],2);bind(buffers[6],3);bind(qp,4);bind(kp,5);bind(vp,6);[encoder setBytes:&p length:sizeof(p) atIndex:7];[encoder setBytes:&pitch length:sizeof(pitch) atIndex:8];}
  else{[encoder setComputePipelineState:gpu.queryOnly];bind(buffers[0],0);bind(buffers[6],1);bind(qp,2);[encoder setBytes:&p length:sizeof(p) atIndex:3];}
  size_t threads=queries*128;[encoder dispatchThreads:MTLSizeMake(threads,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[encoder endEncoding];
  [qk encodeToCommandBuffer:command leftMatrix:queryView rightMatrix:variant==0?cacheKey:mergedKey resultMatrix:scoreView];
  if(variant==0)[ql encodeToCommandBuffer:command leftMatrix:queryView rightMatrix:keyView resultMatrix:liveScoreView];
  encoder=[command computeCommandEncoder];[encoder setComputePipelineState:gpu.bounds.at(extent).softmax[variant==0?0:2]];
  if(variant==0){bind(score,0);bind(liveScore,1);bind(prob,2);bind(buffers[5],3);bind(buffers[6],4);[encoder setBytes:&p length:sizeof(p) atIndex:5];[encoder setBytes:&pitch length:sizeof(pitch) atIndex:6];if(!variant)[encoder setThreadgroupMemoryLength:(extent*2+15)&~size_t(15) atIndex:1];}
  else{bind(score,0);bind(buffers[5],1);bind(buffers[6],2);[encoder setBytes:&p length:sizeof(p) atIndex:3];[encoder setBytes:&pitch length:sizeof(pitch) atIndex:4];}
  [encoder setThreadgroupMemoryLength:(extent*4+15)&~size_t(15) atIndex:0];[encoder dispatchThreadgroups:MTLSizeMake(queries,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[encoder endEncoding];
  [pv encodeToCommandBuffer:command leftMatrix:scoreView rightMatrix:variant==0?cacheValue:mergedValue resultMatrix:queryView];
  if(variant==0)[pl encodeToCommandBuffer:command leftMatrix:probView rightMatrix:valueView resultMatrix:partView];
  encoder=[command computeCommandEncoder];[encoder setComputePipelineState:variant==0?gpu.unpack:gpu.mergedUnpack];bind(qp,0);
  if(variant==0){bind(part,1);bind(buffers[7],2);bind(buffers[6],3);bind(out,4);[encoder setBytes:&p length:sizeof(p) atIndex:5];[encoder setBytes:&pitch length:sizeof(pitch) atIndex:6];}
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
  need(!std::memcmp(mk.contents,committed[0].data(),committed[0].size()*4)&&!std::memcmp(mv.contents,committed[1].data(),committed[1].size()*4),"Immutable committed cache changed");
  checkedWords+=blank.size()+readonlyWords;
 }
 json metadata(const json& shape){json j=shape;j["params"]=std::vector<uint64_t>((uint64_t*)&p,(uint64_t*)&p+16);j["pitches"]={pitch.packed,pitch.cacheScore,pitch.liveScore};j["readonly_parent_words"]=readonlyWords;j["output_parent_words"]=blank.size();j["temporary_merged_kv_bytes"]=2*(batch*8*capacity*128+2*guard)*4;return j;}
};
int main(int argc,char** argv){@autoreleasepool{try{
 need(argc==5,"Usage: timing-helper combined.metal audited-correctness fresh-output forward|reverse");
 fs::path fixture=argv[2],out=argv[3];std::string mode=argv[4];bool reverse=mode=="reverse";
 need(reverse||mode=="forward","Explicit timing order required");need(!fs::exists(out),"Fresh output required");
 auto correctness=readJson(fixture/"correctness1.json"),audit=readJson(fixture/"audit1.json");
 need(correctness["collected"]==true&&audit["passed"]==true,"Completed independently audited collection required");
 const std::vector<int> admitted={0,1,3};
 need(audit["qualified_variants"].size()==4,"Wrong variant gate count");
 need(correctness["variant_names"]==json({"original4mps","committed2mps","compensated_half3_fp32","compensated_half4_fp32"}),"Wrong numerical policies");
 need(audit["original_bound"]=="1e-5 + 2e-4*abs(FP64 expected)","Changed numerical bound");
 need(correctness["probability_scale"]==32768&&correctness["qkv_scale"]==16&&correctness["relaxed_precision"]==false&&correctness["math_mode"]=="Safe","Wrong precision policy");
 need(correctness["cases"].size()==audit["cases"].size(),"Case audit count differs");
 for(int v:admitted){need(audit["qualified_variants"][v]==true,"Unqualified numerical variant");
  for(auto& a:audit["cases"]){need(a["qualified_variants"][v]==true&&a["complete_control_bound_pass"][v]==true,"Unqualified case");need(a["fp64_bound_fraction_max"][v].get<double>()<=1,"FP64 bound violated");}}
 fs::create_directories(out);Pipelines gpu(argv[1]);std::vector<std::unique_ptr<Cell>> cells;
 for(size_t i=0;i<correctness["cases"].size();++i)cells.push_back(std::make_unique<Cell>(gpu,i,correctness["cases"][i],fixture,true));
 // Complete all compilation, allocation, input preparation and warmup before measurement.
 for(auto& c:cells)for(int warm=0;warm<3;++warm)for(int v:admitted){c->run(v);c->check(true,v);}
 json report={{"scope","Same trained/synthetic attention operands and committed-cache inputs; isolated component timing, not serving or quality qualification."},{"cases",json::array()},{"device",gpu.device.name.UTF8String},{"actual_trained_operands",correctness["actual_trained_operands"]},{"admitted_variants",admitted},{"excluded_variants",{2}},{"all_prepared_before_measurement",true},{"warmups_per_cell_variant",3},{"measured_rounds",9},{"reverse",reverse},{"compensated_half_layout_capacities",gpu.layoutWords},{"compensated_half_coordinate_parent_words",gpu.coordinateWords},{"register_layout_status",gpu.registerLayoutWords},{"probability_scale",32768},{"qkv_scale",16},{"relaxed_precision",false},{"math_mode","Safe"},{"variant_names",correctness["variant_names"]},{"native_integration",false},{"serving_speed_claim",false},{"goal_complete",false}};
 for(size_t i=0;i<cells.size();++i){auto& c=*cells[i];auto meta=c.metadata(correctness["cases"][i]);meta["samples"]=json::array();
  for(int round=0;round<9;++round)for(int order=0;order<3;++order){int slot=(round+order)%3,v=admitted[reverse?2-slot:slot];auto t=c.run(v);
   const auto& expected=c.expected[v];need(expected.size()==c.blank.size()&&!std::memcmp(c.out.contents,expected.data(),expected.size()*4),"Timed result changed");
   meta["samples"].push_back({{"round",round},{"order",order},{"variant",v},{"gpu_ms",t[0]},{"wall_ms",t[1]}});}
  // Full readonly parents and immutable committed cache checks stay outside timing.
  c.check(false,3);meta["checked_parent_words"]=c.checkedWords;report["cases"].push_back(meta);
  std::cout<<"Completed timing cell "<<i<<'\n'<<std::flush;
 }
 report["collected"]=true;std::ofstream(out/"screen1.json")<<report.dump(2)<<'\n';return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}}
