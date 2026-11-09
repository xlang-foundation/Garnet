// SPDX-License-Identifier: Apache-2.0
// Prepared half-weight layouts with strict native MPP arithmetic; operator only.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <CommonCrypto/CommonDigest.h>
#include "nlohmann/json.hpp"
#include "src/model/float16_codec.h"
#include "src/backends/metal/packed_affine_layout_guard.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <vector>
using J=nlohmann::json;namespace fs=std::filesystem;
void Need(bool v,const char* why){if(!v)throw std::runtime_error(why);}
template<class T>std::vector<T> Read(fs::path p,size_t count){std::ifstream f(p,std::ios::binary|std::ios::ate);Need(bool(f)&&size_t(f.tellg())==count*sizeof(T),"Wrong fixture size");f.seekg(0);std::vector<T> v(count);f.read((char*)v.data(),count*sizeof(T));Need(bool(f),"Read failed");return v;}
std::string Sha(const void* p,size_t bytes){Need(bytes<=UINT32_MAX,"SHA size exceeds CC_LONG");unsigned char d[CC_SHA256_DIGEST_LENGTH];CC_SHA256(p,CC_LONG(bytes),d);std::ostringstream s;for(auto b:d)s<<std::hex<<std::setfill('0')<<std::setw(2)<<unsigned(b);return s.str();}
void Report(fs::path p,const J& j){auto temp=p;temp+=".tmp";std::ofstream f(temp);f<<j.dump(2)<<'\n';f.close();fs::rename(temp,p);}
void Save(fs::path p,const void* data,size_t bytes){std::ofstream f(p,std::ios::binary);f.write((const char*)data,bytes);Need(bool(f),"Save failed");}
struct Params{uint64_t rows,inner,columns,aStride,oStride;};
int main(int argc,char** argv){J report={{"completed",false},{"cases",J::array()},{"samples",J::array()},{"fp64",J::array()},{"prepared_weights",J::array()}};fs::path destination;
try{@autoreleasepool{
 Need(argc==6,"SHADER ASSETS CAPTURES FRESH_OUTPUT correctness|timing");bool timing=std::string(argv[5])=="timing";Need(timing||std::string(argv[5])=="correctness","Unknown mode");
 destination=argv[4];Need(!fs::exists(destination),"Never reuse output");fs::create_directory(destination);
 auto device=MTLCreateSystemDefaultDevice();Need(device&&[device supportsFamily:MTLGPUFamilyApple10],"Apple10 required");auto queue=[device newCommandQueue];Need(queue!=nil,"Queue failed");
 NSError* error=nil;auto text=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]] encoding:NSUTF8StringEncoding error:&error];Need(text!=nil,"Shader source missing");
 auto options=[MTLCompileOptions new];options.languageVersion=MTLLanguageVersion4_0;options.mathMode=MTLMathModeSafe;
 auto library=[device newLibraryWithSource:text options:options error:&error];if(!library)throw std::runtime_error(error.localizedDescription.UTF8String);
 auto pipe=[&](NSString* name,MTLFunctionConstantValues* constants=nil){auto f=constants?[library newFunctionWithName:name constantValues:constants error:&error]:[library newFunctionWithName:name];if(!f)throw std::runtime_error(error.localizedDescription.UTF8String);auto p=[device newComputePipelineStateWithFunction:f error:&error];if(!p)throw std::runtime_error(error.localizedDescription.UTF8String);Need(p.threadExecutionWidth==32&&p.maxTotalThreadsPerThreadgroup>=256,"Pipeline SIMD/thread mismatch");return p;};
 auto buffer=[&](size_t bytes){auto b=[device newBufferWithLength:bytes options:MTLResourceStorageModeShared];Need(b!=nil,"Allocation failed");return b;};
 auto finish=[&](id<MTLCommandBuffer> cb){[cb commit];[cb waitUntilCompleted];Need(cb.status==MTLCommandBufferStatusCompleted,"GPU command failed");};
 auto guardFunction=[library newFunctionWithName:@"packed_affine_layout_guard"];auto guard=[device newComputePipelineStateWithFunction:guardFunction error:&error];Need(guard!=nil,"ABI probe missing");auto abi=buffer(128);std::fill_n((uint32_t*)abi.contents,32,UINT32_MAX);auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];[e setComputePipelineState:guard];[e setBuffer:abi offset:0 atIndex:0];[e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[e endEncoding];finish(cb);Need(Garnet::PackedAffineLayoutMatches((uint32_t*)abi.contents,32),"Actual cooperative layout mismatch");
 auto prepPipe=[&](bool counted){auto c=[MTLFunctionConstantValues new];[c setConstantValue:&counted type:MTLDataTypeBool atIndex:23];return pipe(@"prepare_half_weights",c);};
 auto prepareNormal=prepPipe(false),prepareCounted=prepPipe(true);
 auto gather=pipe(@"packed_affine_gather");constexpr size_t M=512;constexpr uint32_t G=0x1937abef;constexpr uint16_t H=0x35a5;
 struct Shape{const char* name;size_t inner,columns,fullColumns;bool down;};
 const std::array<Shape,3> shapes{{{"gate",4096,14336,14336,false},{"down",14336,4096,4096,true},{"gate-prefix-proxy",4096,4096,14336,false}}};
 std::map<std::tuple<size_t,size_t,int,size_t,bool>,id<MTLComputePipelineState>> pipelines;
 report["pipeline_resources"]=J::array();
 for(const auto& shape:shapes)for(size_t tile:{128ul,256ul})for(int mode=-1;mode<4;mode++)for(bool counted:{false,true}){
  if(mode==-1&&counted)continue;
  auto constants=[MTLFunctionConstantValues new];uint32_t dims[]={uint32_t(shape.inner),uint32_t(shape.columns),uint32_t(shape.inner),uint32_t(shape.columns)};
  for(NSUInteger i=0;i<4;i++)[constants setConstantValue:dims+i type:MTLDataTypeUInt atIndex:10+i];[constants setConstantValue:&counted type:MTLDataTypeBool atIndex:22];
  std::string name=mode==-1?"half_m"+std::to_string(tile)+"k128":"cached_mode"+std::to_string(mode)+"_m"+std::to_string(tile);
  auto p=pipe([NSString stringWithUTF8String:name.c_str()],constants);Need(p.staticThreadgroupMemoryLength==17408,"Shared-memory profile changed");pipelines[{shape.inner,shape.columns,mode,tile,counted}]=p;
  report["pipeline_resources"].push_back({{"shape",shape.name},{"mode",mode},{"tile",tile},{"counted",counted},{"shared_bytes",p.staticThreadgroupMemoryLength},{"maximum_threads",p.maxTotalThreadsPerThreadgroup},{"simd_width",p.threadExecutionWidth}});
 }
 const std::vector<size_t> layers=timing?std::vector<size_t>{15}:std::vector<size_t>{0,15,31};size_t compared=0,writerWords=0;
 for(auto layer:layers)for(const auto& shape:shapes){@autoreleasepool{
  size_t k=shape.inner,n=shape.columns;auto asset=fs::path(argv[2])/("layer-"+std::to_string(layer)),capture=fs::path(argv[3])/("layer-"+std::to_string(layer));std::string stream=shape.down?"down":"gate";
  auto input=Read<float>(capture/(shape.down?"silu.f32":"norm-input.f32"),M*k);auto gold=Read<float>(capture/(shape.down?"down.f32":"gate.f32"),M*shape.fullColumns);
  auto weights=Read<uint32_t>(asset/(stream+"-weight.u32"),shape.fullColumns*k/8);auto scales=Read<uint16_t>(asset/(stream+"-scales.f16"),shape.fullColumns*k/64),biases=Read<uint16_t>(asset/(stream+"-biases.f16"),shape.fullColumns*k/64);
  weights.resize(n*k/8);scales.resize(n*k/64);biases.resize(n*k/64);
  auto upload=[&](const auto& v){auto b=[device newBufferWithBytes:v.data() length:v.size()*sizeof(v[0]) options:MTLResourceStorageModeShared];Need(b!=nil,"Weight upload failed");return b;};
  auto w=upload(weights),scale=upload(scales),bias=upload(biases);auto x=buffer((M*k+32)*4),compact=buffer((M*k+64)*2),out=buffer((M*n+32)*4),counts=buffer((M*n+32)*4);
  std::fill_n((uint32_t*)x.contents,M*k+32,G);std::memcpy((float*)x.contents+16,input.data(),input.size()*4);std::string inputSha=timing?"":Sha(x.contents,x.length);
  std::array<id<MTLBuffer>,2> prepared;
  size_t rowElements=n*k,tiledElements=n*k/128*136;
  size_t extra=(rowElements+tiledElements)*2+256+(timing?256:(rowElements+tiledElements+64)*4);
  Need(device.recommendedMaxWorkingSetSize>extra+256*1024*1024&&device.currentAllocatedSize<=device.recommendedMaxWorkingSetSize-extra-256*1024*1024,"Prepared cache exceeds device allocation envelope");
  for(size_t layout=0;layout<2;layout++){@autoreleasepool{
   auto began=std::chrono::steady_clock::now();size_t elements=layout?tiledElements:rowElements;
   prepared[layout]=buffer((elements+64)*2);auto target=prepared[layout];
   if(!timing)std::fill_n((uint16_t*)target.contents,elements+64,H);
   else{std::fill_n((uint16_t*)target.contents,32,H);std::fill_n((uint16_t*)target.contents+32+elements,32,H);}
   auto writers=buffer(timing?128:(elements+32)*4);
   if(!timing){std::fill_n((uint32_t*)writers.contents,elements+32,G);std::fill_n((uint32_t*)writers.contents+16,elements,0u);}
   auto command=[queue commandBuffer];auto enc=[command computeCommandEncoder];uint32_t dims[]={uint32_t(k),uint32_t(n),uint32_t(layout),0};
   [enc setComputePipelineState:timing?prepareNormal:prepareCounted];[enc setBuffer:w offset:0 atIndex:0];[enc setBuffer:scale offset:0 atIndex:1];[enc setBuffer:bias offset:0 atIndex:2];[enc setBuffer:target offset:64 atIndex:3];[enc setBytes:dims length:sizeof(dims) atIndex:4];[enc setBuffer:writers offset:64 atIndex:5];[enc dispatchThreads:MTLSizeMake(elements/4,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[enc endEncoding];finish(command);
   double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-began).count(),gpu=(command.GPUEndTime-command.GPUStartTime)*1000;
   Need(std::isfinite(gpu)&&gpu>0,"Missing preparation GPU clock");
   J metadata={{"layer",layer},{"shape",shape.name},{"inner",k},{"columns",n},{"layout",layout},{"elements",elements},{"payload_bytes",elements*2},{"allocation_bytes",target.length},{"counted",!timing},{"gpu_start_seconds",command.GPUStartTime},{"gpu_end_seconds",command.GPUEndTime},{"gpu_ms",gpu},{"allocation_encode_wait_wall_ms",wall}};
   if(!timing){
    for(size_t i=0;i<32;i++)Need(((uint16_t*)target.contents)[i]==H&&((uint16_t*)target.contents)[32+elements+i]==H,"Prepared weight guard changed");
    for(size_t i=0;i<16;i++)Need(((uint32_t*)writers.contents)[i]==G&&((uint32_t*)writers.contents)[16+elements+i]==G,"Preparation writer guard changed");
    for(size_t i=0;i<elements;i++)Need(((uint32_t*)writers.contents)[16+i]==1u,"Preparation did not write exactly once");
    metadata["half_sha256"]=Sha(target.contents,target.length);metadata["writers_sha256"]=Sha(writers.contents,writers.length);metadata["guards_exact"]=true;
    if(layer==0){std::string name="prepared-layer0-"+std::string(shape.name)+"-layout"+std::to_string(layout)+".u16";Save(destination/name,target.contents,target.length);metadata["raw_half"]=name;}
   }
   report["prepared_weights"].push_back(metadata);
  }}
  auto encode=[&](size_t rows,const std::vector<uint32_t>& ids,int mode,size_t tile,bool counted,bool zeroInactive){
   auto cb=[queue commandBuffer];if(zeroInactive&&rows<M){auto blit=[cb blitCommandEncoder];[blit fillBuffer:out range:NSMakeRange(64,M*n*4) value:0];[blit endEncoding];}
   auto e=[cb computeCommandEncoder];uint32_t shape[]={uint32_t(rows),uint32_t(k),uint32_t(k),0};[e setComputePipelineState:gather];[e setBuffer:x offset:64 atIndex:0];[e setBuffer:compact offset:64 atIndex:1];[e setBytes:ids.data() length:rows*4 atIndex:2];[e setBytes:shape length:sizeof(shape) atIndex:3];[e dispatchThreads:MTLSizeMake(rows*k,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding];
   auto p=pipelines.at({k,n,mode,tile,counted});Params params{rows,k,n,k,n};e=[cb computeCommandEncoder];[e setComputePipelineState:p];[e setBuffer:compact offset:64 atIndex:0];[e setBuffer:w offset:0 atIndex:1];[e setBuffer:scale offset:0 atIndex:2];[e setBuffer:bias offset:0 atIndex:3];[e setBuffer:out offset:64 atIndex:4];[e setBytes:&params length:sizeof(params) atIndex:5];[e setBytes:ids.data() length:rows*4 atIndex:6];if(mode!=-1)[e setBuffer:counts offset:64 atIndex:7];if(mode>0)[e setBuffer:prepared[mode==1?0:1] offset:64 atIndex:8];[e dispatchThreadgroups:MTLSizeMake(n/64,(rows+tile-1)/tile,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding];finish(cb);return cb;
  };
  if(!timing){
   for(size_t rr:{0ul,256ul,511ul})for(size_t col:{0ul,1ul,15ul,16ul,31ul,32ul,n/2,n-2,n-1}){
    double sum=0,sumabs=0;for(size_t j=0;j<k;j++){size_t ci=col*(k/64)+j/64;uint32_t code=(weights[col*(k/8)+j/8]>>((j%8)*4))&15u;float sw=Garnet::Float16ToFloat32(scales[ci]),bw=Garnet::Float16ToFloat32(biases[ci]);float weight=Garnet::Float16ToFloat32(Garnet::Float32ToFloat16RNE(std::fma(float(code),sw,bw)));float value=Garnet::Float16ToFloat32(Garnet::Float32ToFloat16RNE(input[rr*k+j]));double product=double(weight)*double(value);sum+=product;sumabs+=std::abs(product);}
    float actual=gold[rr*shape.fullColumns+col];double fraction=std::abs(double(actual)-sum)/(2e-6+4e-6*sumabs);Need(std::isfinite(actual)&&fraction<=1,"Fixed independent FP64 bound failed");report["fp64"].push_back({{"layer",layer},{"shape",shape.name},{"row",rr},{"column",col},{"actual",actual},{"sum",sum},{"sum_abs",sumabs},{"fraction",fraction}});
   }
   for(size_t rows:{127ul,128ul,129ul,255ul,256ul,257ul,383ul,384ul,511ul,512ul})for(size_t tile:{128ul,256ul}){
    std::vector<uint32_t> ids(rows);for(size_t i=0;i<rows;i++)ids[i]=uint32_t((511-i*257)%512);
    std::vector<uint32_t> expected(M*n+32,G),expectedCounts(M*n+32,G),zeroCounts(M*n+32,G);std::fill_n(expectedCounts.data()+16,M*n,0u);std::fill_n(zeroCounts.data()+16,M*n,0u);
    for(auto row:ids){std::memcpy(expected.data()+16+row*n,gold.data()+row*shape.fullColumns,n*4);std::fill_n(expectedCounts.data()+16+row*n,n,1u);}
    std::string expectedSha=Sha(expected.data(),expected.size()*4),oneSha=Sha(expectedCounts.data(),expectedCounts.size()*4),zeroSha=Sha(zeroCounts.data(),zeroCounts.size()*4);
    for(int mode=-1;mode<4;mode++)for(bool counted:{false,true}){
     if(mode==-1&&counted)continue;
     std::fill_n((uint32_t*)out.contents,M*n+32,G);std::fill_n((uint16_t*)compact.contents,M*k+64,H);std::memcpy(counts.contents,zeroCounts.data(),zeroCounts.size()*4);
     encode(rows,ids,mode,tile,counted,false);
     Need(!std::memcmp(out.contents,expected.data(),out.length),"Complete trained output/guard comparison failed");Need(!std::memcmp(counts.contents,(counted?expectedCounts:zeroCounts).data(),counts.length),"Complete writer/guard comparison failed");Need(Sha(x.contents,x.length)==inputSha,"Read-only input changed");
     for(size_t j=0;j<32;j++)Need(((uint16_t*)compact.contents)[j]==H&&((uint16_t*)compact.contents)[32+M*k+j]==H,"Gather guard changed");
     compared+=M*n+32;writerWords+=M*n+32;J row={{"layer",layer},{"shape",shape.name},{"rows",rows},{"tile",tile},{"mode",mode},{"counted",counted},{"output_sha256",Sha(out.contents,out.length)},{"expected_output_sha256",expectedSha},{"writers_sha256",Sha(counts.contents,counts.length)},{"expected_writers_sha256",counted?oneSha:zeroSha},{"complete_parent_words",M*n+32},{"guards_exact",true}};
     if(counted&&((layer==0&&std::string(shape.name)=="gate"&&rows==512&&tile==256)||(layer==31&&std::string(shape.name)=="down"&&rows==129&&tile==128))){
      std::string prefix="layer-"+std::to_string(layer)+"-"+shape.name+"-rows"+std::to_string(rows)+"-tile"+std::to_string(tile)+"-mode"+std::to_string(mode);Save(destination/(prefix+"-output.u32"),out.contents,out.length);Save(destination/(prefix+"-writers.u32"),counts.contents,counts.length);row["raw_output"]=prefix+"-output.u32";row["raw_writers"]=prefix+"-writers.u32";
     }
     report["cases"].push_back(row);
    }
   }
   std::cout<<J{{"layer",layer},{"shape",shape.name},{"cases",report["cases"].size()}}.dump()<<'\n'<<std::flush;
  }else{
   for(size_t rows:{128ul,256ul,512ul})for(bool sparse:{false,true}){
    size_t tile=(n==4096&&k==4096)?128:(rows>=(k==14336?384:256)?256:128);std::vector<uint32_t> ids(rows);for(size_t i=0;i<rows;i++)ids[i]=uint32_t(sparse?(511-i*257)%512:i);
    for(int round=-5;round<20;round++)for(int at=0;at<5;at++){
     int index=(at+(round+5)%5)%5;if((round+5)&1)index=4-index;int mode=index-1;auto cb=encode(rows,ids,mode,tile,false,true);double ms=(cb.GPUEndTime-cb.GPUStartTime)*1000;Need(std::isfinite(ms)&&ms>0,"Missing GPU clock");
     if(round>=0)report["samples"].push_back({{"layer",layer},{"shape",shape.name},{"rows",rows},{"tile",tile},{"sparse",sparse},{"round",round},{"position",at},{"mode",mode},{"gpu_ms",ms},{"gpu_start_seconds",cb.GPUStartTime},{"gpu_end_seconds",cb.GPUEndTime}});
    }
    std::cout<<J{{"timing_shape",shape.name},{"rows",rows},{"sparse",sparse},{"samples",report["samples"].size()}}.dump()<<'\n'<<std::flush;
   }
  }
  Report(destination/"result1.json",report);
 }}
 report.update({{"completed",true},{"timing",timing},{"cases_count",report["cases"].size()},{"samples_count",report["samples"].size()},{"complete_parent_word_comparisons",compared},{"writer_word_comparisons",writerWords},{"exact_native_captured_controls",!timing},{"no_runtime_integration",true},{"serving_speed_claim",false},{"goal_complete",false}});Report(destination/"result1.json",report);std::cout<<J{{"completed",true},{"timing",timing},{"cases",report["cases"].size()},{"samples",report["samples"].size()}}.dump()<<'\n';return 0;
}}catch(const std::exception& error){report["error"]=error.what();if(!destination.empty()&&fs::exists(destination))Report(destination/"result1.json",report);std::cerr<<error.what()<<'\n';return 1;}}
