// SPDX-License-Identifier: Apache-2.0
// Isolated current packed projection descriptors. No model/serving claim.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "nlohmann/json.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <set>
#include <vector>
using JSON=nlohmann::json;namespace fs=std::filesystem;
void Need(bool v,const char* s){if(!v)throw std::runtime_error(s);}
template<class T>std::vector<T> Read(fs::path p,size_t n){Need(fs::file_size(p)==n*sizeof(T),"Fixture extent differs");std::vector<T> a(n);std::ifstream f(p,std::ios::binary);f.read((char*)a.data(),n*sizeof(T));Need(bool(f),"Fixture read failed");return a;}
template<class T>void Save(fs::path p,const std::vector<T>& a){std::ofstream f(p,std::ios::binary);f.write((const char*)a.data(),a.size()*sizeof(T));Need(bool(f),"Cannot save payload");}
void Report(fs::path p,const JSON& a){std::ofstream f(p);f<<a.dump(2)<<'\n';Need(bool(f),"Cannot save metadata");}
uint16_t HalfBits(float x){_Float16 h=(_Float16)x;uint16_t b;std::memcpy(&b,&h,2);return b;}
float HalfFloat(uint16_t b){_Float16 h;std::memcpy(&h,&b,2);return float(h);}
struct Fixture {
 size_t n,k;std::vector<float> input,gold,poison;std::vector<uint32_t> weights;std::vector<uint16_t> scales,biases;
 id<MTLBuffer> a,w,s,b,o,compact;std::array<id<MTLComputePipelineState>,3> pipelines{};
 size_t tile=0;
};
int main(int argc,char**argv){fs::path output;JSON report={{"completed",false},{"cases",JSON::array()}};
 try{@autoreleasepool{
 Need(argc==6,"correctness|timing SHADER CAPTURES FRESH_OUTPUT forward|reverse");std::string mode=argv[1],order=argv[5];Need((mode=="correctness"||mode=="timing")&&(order=="forward"||order=="reverse"),"Unknown mode/order");
 output=fs::absolute(argv[4]);Need(fs::create_directory(output),"Fresh output directory required");
 auto device=MTLCreateSystemDefaultDevice();Need(device&&[device supportsFamily:MTLGPUFamilyApple10],"Apple10/M5 required");auto queue=[device newCommandQueue];Need(queue!=nil,"Cannot create queue");NSError* error=nil;
 auto options=[MTLCompileOptions new];options.languageVersion=MTLLanguageVersion4_0;options.mathMode=MTLMathModeSafe;
 auto source=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[2]] encoding:NSUTF8StringEncoding error:&error];Need(source!=nil,"Cannot read exact shader source");
 auto library=[device newLibraryWithSource:source options:options error:&error];if(!library)throw std::runtime_error(error.localizedDescription.UTF8String);
 auto pipeline=[&](NSString* name,MTLFunctionConstantValues* constants=nil){auto fn=constants?[library newFunctionWithName:name constantValues:constants error:&error]:[library newFunctionWithName:name];Need(fn!=nil,"Missing compiled function");auto p=[device newComputePipelineStateWithFunction:fn error:&error];if(!p)throw std::runtime_error(error.localizedDescription.UTF8String);Need(p.threadExecutionWidth==32&&p.maxTotalThreadsPerThreadgroup>=32,"Unsupported pipeline geometry");return p;};
 auto finish=[&](id<MTLCommandBuffer> command){[command commit];[command waitUntilCompleted];Need(command.status==MTLCommandBufferStatusCompleted,"GPU command failed");};
 JSON guards=JSON::array();
 for(NSString* name in @[@"packed_affine_layout_guard",@"relaxed_layout_guard",@"fault_layout_guard",@"descriptor_coordinates"]){
  bool coordinates=[name isEqualToString:@"descriptor_coordinates"],fault=[name isEqualToString:@"fault_layout_guard"];size_t count=coordinates?1281:32;std::vector<uint32_t> words(count+32,0x7fc12345);std::fill(words.begin()+16,words.end()-16,coordinates?0u:UINT32_MAX);
  auto status=[device newBufferWithBytes:words.data() length:words.size()*4 options:MTLResourceStorageModeShared];Need(status!=nil,"Probe allocation failed");auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];[e setComputePipelineState:pipeline(name)];[e setBuffer:status offset:64 atIndex:0];[e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[e endEncoding];finish(cb);
  const auto* got=(const uint32_t*)status.contents;for(size_t i=0;i<16;++i)Need(got[i]==0x7fc12345&&got[16+count+i]==0x7fc12345,"Probe parent guard changed");
  for(size_t i=0;i<count;++i)Need(got[16+i]==(coordinates?uint32_t(i<1280):fault?1u:0u),"Cooperative layout/coverage guard failed");guards.push_back({{"kernel",std::string(name.UTF8String)},{"values",std::vector<uint32_t>(got+16,got+16+count)}});
 }
 auto gather=pipeline(@"packed_affine_gather");Need(gather.maxTotalThreadsPerThreadgroup>=256,"Gather geometry differs");
 const char* folders[]={"capture-up2","capture-down1","capture-qo1"};const size_t ns[]={14336,4096,4096},ks[]={4096,14336,4096};std::array<Fixture,3> fixtures;
 for(size_t fi=0;fi<3;++fi){auto& f=fixtures[fi];f.n=ns[fi];f.k=ks[fi];auto dir=fs::path(argv[3])/folders[fi];f.input=Read<float>(dir/"input-parent.f32",514*f.k);f.gold=Read<float>(dir/"reference-parent.f32",514*f.n);f.weights=Read<uint32_t>(dir/"weight.u32",f.n*f.k/8);f.scales=Read<uint16_t>(dir/"scales.f16",f.n*f.k/64);f.biases=Read<uint16_t>(dir/"biases.f16",f.n*f.k/64);
  f.poison=f.gold;std::fill(f.poison.begin()+f.n,f.poison.end()-f.n,NAN);
  auto upload=[&](const auto& a){auto buffer=[device newBufferWithBytes:a.data() length:a.size()*sizeof(a[0]) options:MTLResourceStorageModeShared];Need(buffer!=nil,"Fixture allocation failed");return buffer;};f.a=upload(f.input);f.w=upload(f.weights);f.s=upload(f.scales);f.b=upload(f.biases);f.o=upload(f.poison);f.compact=[device newBufferWithLength:14680064+128 options:MTLResourceStorageModeShared];Need(f.compact!=nil,"Scratch allocation failed");
 }
 // All six arithmetic and six writer specializations exist before any timing.
 std::array<std::array<std::array<id<MTLComputePipelineState>,3>,2>,3> plans;
 for(size_t fi=0;fi<3;++fi)for(size_t ti=0;ti<2;++ti){auto& f=fixtures[fi];auto constants=[MTLFunctionConstantValues new];uint32_t dims[]={uint32_t(f.k),uint32_t(f.n),uint32_t(f.k),uint32_t(f.n)};for(NSUInteger i=0;i<4;++i)[constants setConstantValue:dims+i type:MTLDataTypeUInt atIndex:10+i];
  for(size_t v=0;v<3;++v){NSString* name=[NSString stringWithFormat:@"%s_m%zuk128",v==0?"half":v==1?"relaxed":"writer",ti?256ul:128ul];auto p=pipeline(name,constants);Need(p.maxTotalThreadsPerThreadgroup>=256,"Projection threads exceed pipeline");if(v<2)Need(p.staticThreadgroupMemoryLength==17408,"Projection shared storage changed");plans[fi][ti][v]=p;}
 }
 uint64_t checked=0,samples=0;double maxFraction=0;
 auto execute=[&](size_t fi,size_t count,bool dense,bool reverse,size_t v,bool keep){@autoreleasepool{
  auto& f=fixtures[fi];std::vector<uint32_t> ids(count);for(size_t i=0;i<count;++i)ids[i]=dense?i:uint32_t((i*127+5)%512);if(reverse)std::reverse(ids.begin(),ids.end());
  std::vector<bool> active(512);for(auto i:ids)active[i]=true;size_t tile=count>=256&&(f.n==14336||(f.k==14336&&count>=384))?256:128;
  std::memcpy(f.o.contents,f.poison.data(),f.poison.size()*4);auto* out=(uint32_t*)f.o.contents;
  if(v==2)std::fill_n(out+f.n,512*f.n,0u);else for(size_t row=0;row<512;++row)if(!active[row])std::fill_n((float*)f.o.contents+(row+1)*f.n,f.n,0.f);
  auto* scratch=(uint16_t*)f.compact.contents;std::fill_n(scratch,f.compact.length/2,0x7e09);
  auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];Need(cb&&e,"Cannot encode gather");uint32_t shape[]={uint32_t(count),uint32_t(f.k),uint32_t(f.k),0};[e setComputePipelineState:gather];[e setBuffer:f.a offset:f.k*4 atIndex:0];[e setBuffer:f.compact offset:64 atIndex:1];[e setBytes:ids.data() length:ids.size()*4 atIndex:2];[e setBytes:shape length:sizeof(shape) atIndex:3];[e dispatchThreads:MTLSizeMake(count*f.k,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding];
  struct Params{uint64_t rows,inner,columns,aStride,oStride;};Params p{count,f.k,f.n,f.k,f.n};e=[cb computeCommandEncoder];[e setComputePipelineState:plans[fi][tile==256][v]];[e setBuffer:f.compact offset:64 atIndex:0];[e setBuffer:f.w offset:0 atIndex:1];[e setBuffer:f.s offset:0 atIndex:2];[e setBuffer:f.b offset:0 atIndex:3];[e setBuffer:f.o offset:f.n*4 atIndex:4];[e setBytes:&p length:sizeof(p) atIndex:5];[e setBytes:ids.data() length:ids.size()*4 atIndex:6];[e dispatchThreadgroups:MTLSizeMake(f.n/64,(count+tile-1)/tile,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding];finish(cb);
  double ms=(cb.GPUEndTime-cb.GPUStartTime)*1000;Need(std::isfinite(ms)&&ms>0,"Invalid GPU span");
  // Everything below is outside the GPU span. Preserve outputs before rejecting.
  std::vector<float> actual(f.gold.size());std::memcpy(actual.data(),f.o.contents,actual.size()*4);
  std::string label="case"+std::to_string(fi)+"-n"+std::to_string(count)+(dense?"-dense":"-indexed")+(reverse?"-reverse":"")+"-v"+std::to_string(v);
  if(keep)Save(output/(label+".f32"),actual);
  auto expected=f.gold;for(size_t row=0;row<512;++row)for(size_t col=0;col<f.n;++col){if(v==2){uint32_t bits=uint32_t(active[row]);std::memcpy(&expected[(row+1)*f.n+col],&bits,4);}else if(!active[row])expected[(row+1)*f.n+col]=0;}
  Need(!std::memcmp(actual.data(),expected.data(),actual.size()*4),v==2?"Actual output writer coverage differs":"Descriptor candidate/control differs from complete frozen native output");checked+=actual.size();
  std::vector<uint16_t> compact(f.compact.length/2,0x7e09);for(size_t i=0;i<count;++i)for(size_t j=0;j<f.k;++j)compact[32+i*f.k+j]=HalfBits(f.input[(ids[i]+1)*f.k+j]);Need(!std::memcmp(scratch,compact.data(),f.compact.length),"Complete compact scratch/guards/tails differ");
  Need(!std::memcmp(f.a.contents,f.input.data(),f.input.size()*4)&&!std::memcmp(f.w.contents,f.weights.data(),f.weights.size()*4)&&!std::memcmp(f.s.contents,f.scales.data(),f.scales.size()*2)&&!std::memcmp(f.b.contents,f.biases.data(),f.biases.size()*2),"Readonly parents changed");
  if(v<2)for(size_t pos:std::set<size_t>{0,count/2,count-1})for(size_t col:std::set<size_t>{0,1,15,16,31,32,63,64,f.n/2,f.n-1}){
   size_t row=ids[pos];double sum=0,absolute=0;for(size_t j=0;j<f.k;++j){size_t ci=col*(f.k/64)+j/64;float sc=HalfFloat(f.scales[ci]),bias=HalfFloat(f.biases[ci]);uint32_t code=(f.weights[col*(f.k/8)+j/8]>>((j%8)*4))&15;double weight=HalfFloat(HalfBits(std::fma(float(code),sc,bias))),input=HalfFloat(HalfBits(f.input[(row+1)*f.k+j]));double product=input*weight;sum+=product;absolute+=std::abs(product);}double fraction=std::abs(double(actual[(row+1)*f.n+col])-sum)/(2e-6+4e-6*absolute);Need(fraction<=1,"Original fixed FP64 gate failed");maxFraction=std::max(maxFraction,fraction);++samples;
  }
  return JSON{{"fixture",fi},{"count",count},{"dense",dense},{"reverse",reverse},{"variant",v},{"tile",tile},{"gpu_ms_including_gather",ms},{"output_file",keep?label+".f32":""}};
 }};
 for(size_t fi=0;fi<3;++fi){
  if(mode=="correctness"){
   std::vector<std::array<size_t,3>> cells;for(size_t count:{128ul,129ul,255ul,256ul,257ul,383ul,384ul,511ul,512ul})cells.push_back({count,0,0});cells.push_back({512,0,1});for(size_t count:{128ul,256ul,512ul})cells.push_back({count,1,0});
   for(auto cell:cells){for(size_t v=0;v<3;++v)report["cases"].push_back(execute(fi,cell[0],cell[1],cell[2],v,v>0));Report(output/"progress1.json",report);}
  }else{
   for(int round=-3;round<9;++round)for(size_t step=0;step<2;++step){size_t v=(step+size_t(round+3)+(order=="reverse"))%2;auto row=execute(fi,512,true,false,v,round==8);row["round"]=round;row["step"]=step;report["cases"].push_back(row);}
  }
  std::cout<<"packed descriptor fixture "<<fi<<" PASS\n"<<std::flush;
 }
 report.update({{"completed",true},{"mode",mode},{"order",order},{"math_mode","Safe"},{"relaxed_descriptor_only",true},{"guards",guards},{"checked_parent_words",checked},{"FP64_values",samples},{"maximum_FP64_fraction",maxFraction},{"timing_admitted",mode=="timing"},{"native_integration",false},{"quality_qualified",false},{"serving_speed_claim",false}});Report(output/"result1.json",report);return 0;
 }}catch(const std::exception& error){std::cerr<<error.what()<<'\n';if(!output.empty()&&fs::exists(output)){report["error"]=error.what();Report(output/"failure1.json",report);}return 1;}}
