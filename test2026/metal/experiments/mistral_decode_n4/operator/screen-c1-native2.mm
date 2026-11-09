// SPDX-License-Identifier: Apache-2.0
// Lossless N4 packed-word layout screen; no full-model or serving claim.
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
#include <type_traits>
#include <vector>
using J=nlohmann::json;namespace fs=std::filesystem;
void Need(bool v,const char* why){if(!v)throw std::runtime_error(why);}
template<class T>std::vector<T> Read(fs::path p,size_t count){std::ifstream f(p,std::ios::binary|std::ios::ate);Need(bool(f)&&size_t(f.tellg())==count*sizeof(T),"Wrong fixture size");f.seekg(0);std::vector<T> v(count);f.read((char*)v.data(),count*sizeof(T));Need(bool(f),"Fixture read failed");return v;}
std::string Sha(const void* p,size_t bytes){Need(bytes<=UINT32_MAX,"Hash size exceeds CC_LONG");unsigned char d[32];CC_SHA256(p,CC_LONG(bytes),d);std::ostringstream s;for(auto b:d)s<<std::hex<<std::setfill('0')<<std::setw(2)<<unsigned(b);return s.str();}
void Report(fs::path p,const J& j){auto t=p;t+=".tmp";std::ofstream f(t);f<<j.dump(2)<<'\n';f.close();fs::rename(t,p);}
void Save(fs::path p,const void* data,size_t bytes){std::ofstream f(p,std::ios::binary);f.write((const char*)data,bytes);Need(bool(f),"Output save failed");}
struct Params{uint64_t rows,inner,columns,aStride,oStride;};
int main(int argc,char** argv){J report={{"completed",false},{"cases",J::array()},{"fp64",J::array()},{"samples",J::array()},{"pipeline_resources",J::array()}};fs::path dest;
try{@autoreleasepool{
 Need(argc==9,"STUDY ASSETS CAPTURES FRESH_OUTPUT correctness|warm|pressure ORDER(0|1) ROUNDS");
 bool padded=std::string(argv[8])=="padded";Need(padded||std::string(argv[8])=="contiguous","Bad layout");std::string mode=argv[5];bool correctness=mode=="correctness",pressure=mode=="pressure";Need(correctness||pressure||mode=="warm","Unknown mode");int order=std::stoi(argv[6]),rounds=std::stoi(argv[7]);Need((order==0||order==1)&&rounds>=3&&rounds<=31,"Bad sample policy");
 dest=argv[4];Need(!fs::exists(dest),"Never overwrite output");fs::create_directory(dest);
 auto device=MTLCreateSystemDefaultDevice();Need(device!=nil,"Metal device absent");size_t baseline=device.currentAllocatedSize;
 report.update({{"device",device.name.UTF8String},{"mode",mode},{"order",order},{"rounds",rounds},{"warmups",5},{"pressure_bytes",pressure?256*1024*1024:0},{"baseline_allocated_bytes",baseline},{"native_math","MSL3.1 fastMathEnabled=NO, FP32 operands/FMA/SIMD reduction"}});
 {@autoreleasepool{
 auto queue=[device newCommandQueue];Need(queue!=nil,"Command queue absent");NSError* error=nil;
 std::array<id<MTLLibrary>,2> libs;
 auto options=[MTLCompileOptions new];options.languageVersion=MTLLanguageVersion3_1;options.fastMathEnabled=NO;
 for(int indexed=0;indexed<2;indexed++){
  auto path=fs::path(argv[1])/(indexed?"indexed1.metal":"ordinary1.metal");auto text=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:path.c_str()] encoding:NSUTF8StringEncoding error:&error];Need(text!=nil,"Missing shader");
  libs[indexed]=[device newLibraryWithSource:text options:options error:&error];if(!libs[indexed])throw std::runtime_error(error.localizedDescription.UTF8String);
 }
 auto buffer=[&](size_t bytes){auto b=[device newBufferWithLength:bytes options:MTLResourceStorageModeShared];Need(b!=nil,"Buffer allocation failed");return b;};
 auto finish=[&](id<MTLCommandBuffer> cb){[cb commit];[cb waitUntilCompleted];if(cb.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(cb.error.localizedDescription.UTF8String);};
 id<MTLBuffer> pressureData=nil;id<MTLComputePipelineState> pressurePipe=nil;
 if(pressure){pressureData=buffer(256*1024*1024);std::fill_n((uint32_t*)pressureData.contents,pressureData.length/4,37u);auto f=[libs[0] newFunctionWithName:@"geometry_pressure"];pressurePipe=[device newComputePipelineStateWithFunction:f error:&error];Need(pressurePipe!=nil,"Pressure pipeline absent");}
 struct Shape{const char* name;const char* stream;size_t k,n,full;bool down;};
 std::vector<Shape> shapes{{"gate","gate",4096,14336,14336,false},{"up","up",4096,14336,14336,false},{"down","down",14336,4096,4096,true}};
 if(correctness)for(size_t n:{1ul,7ul,17ul,33ul})shapes.push_back({"gate-tail","gate",4096,n,14336,false});
 using Key=std::tuple<bool,size_t,size_t,size_t,size_t,int>;
 std::map<Key,id<MTLComputePipelineState>> pipelines;bool preparationFinished=false;
 auto pipeline=[&](bool indexed,size_t k,size_t n,size_t aStride,size_t oStride,size_t active,int variant){
  Key key{indexed,k,n,aStride,active,variant};auto found=pipelines.find(key);if(found!=pipelines.end())return found->second;Need(!preparationFinished,"Pipeline creation after preparation barrier");
  auto c=[MTLFunctionConstantValues new];uint32_t count=uint32_t(active),dims[]={uint32_t(k),uint32_t(n),uint32_t(aStride),uint32_t(oStride)};[c setConstantValue:&count type:MTLDataTypeUInt atIndex:0];for(NSUInteger i=0;i<4;i++)[c setConstantValue:dims+i type:MTLDataTypeUInt atIndex:10+i];
  std::string name=variant==0?(indexed?"affine_q4_indexed_decode":"affine_q4_decode"):variant==4?"affine_q4_blocked":"affine_q4_interleaved4";
  auto f=[libs[indexed] newFunctionWithName:[NSString stringWithUTF8String:name.c_str()] constantValues:c error:&error];if(!f)throw std::runtime_error(error.localizedDescription.UTF8String);auto p=[device newComputePipelineStateWithFunction:f error:&error];if(!p)throw std::runtime_error(error.localizedDescription.UTF8String);
  Need(p.threadExecutionWidth==32&&p.maxTotalThreadsPerThreadgroup>=128&&p.staticThreadgroupMemoryLength<=device.maxThreadgroupMemoryLength,"Pipeline exceeds SIMD/thread/storage bounds");pipelines[key]=p;
  report["pipeline_resources"].push_back({{"indexed",indexed},{"inner",k},{"columns",n},{"input_stride",aStride},{"output_stride",oStride},{"active",active},{"variant",variant},{"maximum_threads",p.maxTotalThreadsPerThreadgroup},{"threadgroup_bytes",p.staticThreadgroupMemoryLength},{"simd_width",p.threadExecutionWidth}});return p;
 };
 for(auto shape:shapes)for(bool indexed:{false,true})for(size_t active:{1ul,2ul,3ul,4ul})for(int variant:{0,4,9})pipeline(indexed,shape.k,shape.n,shape.k+(padded?3:0),shape.n+(padded?5:0),active,variant);
 preparationFinished=true;report["all_pipelines_prepared_before_collection"]=true;report["prepared_pipeline_count"]=pipelines.size();
 size_t words=0,caseIndex=0;std::vector<size_t> layers=correctness?std::vector<size_t>{0,15,31}:std::vector<size_t>{15};
 for(size_t layer:layers)for(auto shape:shapes){@autoreleasepool{
  const size_t k=shape.k,n=shape.n,aStride=k+(padded?3:0),oStride=n+(padded?5:0);constexpr uint32_t G=0x1937abef;const size_t xWords=32+4*aStride,yWords=32+4*oStride;
  auto asset=fs::path(argv[2])/("layer-"+std::to_string(layer)),capture=fs::path(argv[3])/("layer-"+std::to_string(layer));
  auto input=Read<float>(capture/(shape.down?"silu.f32":"norm-input.f32"),512*k);auto w=Read<uint32_t>(asset/(std::string(shape.stream)+"-weight.u32"),shape.full*k/8);auto sc=Read<uint16_t>(asset/(std::string(shape.stream)+"-scales.f16"),shape.full*k/64),bi=Read<uint16_t>(asset/(std::string(shape.stream)+"-biases.f16"),shape.full*k/64);w.resize(n*k/8);sc.resize(n*k/64);bi.resize(n*k/64);
  auto pack=[&](const auto& source,size_t width){
   using V=std::decay_t<decltype(source)>;V result(((n+3)/4)*width*4,0);
   for(size_t col=0;col<n;col++)for(size_t word=0;word<width;word++)result[((col/4)*width+word)*4+col%4]=source[col*width+word];
   for(size_t col=0;col<n;col++)for(size_t word=0;word<width;word++)Need(result[((col/4)*width+word)*4+col%4]==source[col*width+word],"Packed word roundtrip failed");
   for(size_t col=n;col<(n+3)/4*4;col++)for(size_t word=0;word<width;word++)Need(result[((col/4)*width+word)*4+col%4]==0,"Packed tail padding nonzero");
   return result;
  };
  auto packedW=pack(w,k/8);auto packedS=pack(sc,k/64),packedB=pack(bi,k/64);
  auto upload=[&](const auto& v){auto b=[device newBufferWithBytes:v.data() length:v.size()*sizeof(v[0]) options:MTLResourceStorageModeShared];Need(b!=nil,"Fixture upload failed");return b;};auto weights=upload(w),scales=upload(sc),biases=upload(bi);auto x=buffer(xWords*4),y=buffer(yWords*4);
  auto guarded=[&](const auto& data){auto b=buffer(data.size()*sizeof(data[0])+128);std::fill_n((uint32_t*)b.contents,b.length/4,G);std::memcpy((char*)b.contents+64,data.data(),data.size()*sizeof(data[0]));return b;};
  auto packedWeights=guarded(packedW),packedScales=guarded(packedS),packedBiases=guarded(packedB);
  const std::string pwSha=correctness?Sha(packedWeights.contents,packedWeights.length):"",psSha=correctness?Sha(packedScales.contents,packedScales.length):"",pbSha=correctness?Sha(packedBiases.contents,packedBiases.length):"";
  if(correctness){
   auto metadata=[&](const auto& source,const auto& packed,id<MTLBuffer> b,const char* kind){
    std::string name="layer"+std::to_string(layer)+"-"+shape.name+"-n"+std::to_string(n)+"-"+kind+".bin";
    J row={{"layer",layer},{"shape",shape.name},{"inner",k},{"columns",n},{"padded_columns",(n+3)/4*4},{"kind",kind},{"element_bytes",sizeof(source[0])},{"source_elements",source.size()},{"packed_elements",packed.size()},{"source_sha256",Sha(source.data(),source.size()*sizeof(source[0]))},{"packed_sha256",Sha(packed.data(),packed.size()*sizeof(packed[0]))},{"allocated_bytes",b.length},{"roundtrip_exact",true},{"tail_padding_zero",true}};
    if(layer==0&&n==7){Save(dest/name,packed.data(),packed.size()*sizeof(packed[0]));row["raw_packed"]=name;}
    report["prepared_weights"].push_back(row);
   };metadata(w,packedW,packedWeights,"weight");metadata(sc,packedS,packedScales,"scale");metadata(bi,packedB,packedBiases,"bias");
  }
  std::fill_n((uint32_t*)x.contents,xWords,G);std::array<size_t,4> sourceRows{0,127,256,511};for(size_t row=0;row<4;row++)std::memcpy((float*)x.contents+16+row*aStride,input.data()+sourceRows[row]*k,k*4);
  const std::string xSha=correctness?Sha(x.contents,x.length):"",wSha=correctness?Sha(weights.contents,weights.length):"",sSha=correctness?Sha(scales.contents,scales.length):"",bSha=correctness?Sha(biases.contents,biases.length):"";
  struct Selection{size_t physical,mask;bool indexed;};std::vector<Selection> selections;
  if(correctness){for(size_t rows=1;rows<=4;rows++)selections.push_back({rows,(1ul<<rows)-1,false});for(size_t mask=0;mask<16;mask++)selections.push_back({4,mask,true});}
  else selections={{4,1,true},{4,8,true}};
  for(auto selection:selections){
   std::vector<uint32_t> ids;for(size_t row=0;row<selection.physical;row++)if(selection.mask&(1ul<<row))ids.push_back(uint32_t(row));
   // Compile all variants before the timing loops; no pipeline creation inside.
   for(int variant:{0,4,9})if(!ids.empty())pipeline(selection.indexed,k,n,aStride,oStride,ids.size(),variant);
   double pressureStart=0,pressureEnd=0;
   auto encode=[&](int variant){
    if(pressure){auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];[e setComputePipelineState:pressurePipe];[e setBuffer:pressureData offset:0 atIndex:0];[e dispatchThreads:MTLSizeMake(pressureData.length/4,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding];finish(cb);pressureStart=cb.GPUStartTime;pressureEnd=cb.GPUEndTime;Need(pressureEnd>pressureStart,"Invalid pressure clock");}
    auto cb=[queue commandBuffer];if(selection.indexed&&ids.size()<selection.physical){auto blit=[cb blitCommandEncoder];[blit fillBuffer:y range:NSMakeRange(64,((selection.physical-1)*oStride+n)*4) value:0];[blit endEncoding];}
    if(!ids.empty()){
     auto e=[cb computeCommandEncoder];auto p=pipelines.at({selection.indexed,k,n,aStride,ids.size(),variant});Params params{ids.size(),k,n,aStride,oStride};[e setComputePipelineState:p];[e setBuffer:x offset:64 atIndex:0];[e setBuffer:(variant==9?packedWeights:weights) offset:(variant==9?64:0) atIndex:1];[e setBuffer:(variant==9?packedScales:scales) offset:(variant==9?64:0) atIndex:2];[e setBuffer:(variant==9?packedBiases:biases) offset:(variant==9?64:0) atIndex:3];[e setBuffer:y offset:64 atIndex:4];[e setBytes:&params length:sizeof(params) atIndex:5];if(selection.indexed)[e setBytes:ids.data() length:ids.size()*4 atIndex:6];size_t width=variant?16:4,threads=128;[e dispatchThreadgroups:MTLSizeMake((n+width-1)/width,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];[e endEncoding];
    }finish(cb);return cb;
   };
   if(correctness){
    std::fill_n((uint32_t*)y.contents,yWords,G);encode(0);std::vector<uint32_t> gold(yWords);std::memcpy(gold.data(),y.contents,y.length);std::string goldSha=Sha(gold.data(),y.length);std::string allControl="case"+std::to_string(caseIndex)+"-legacy.u32";Save(dest/allControl,gold.data(),y.length);
    for(size_t i=0;i<yWords;i++){
     bool payload=i>=16&&i<16+selection.physical*oStride;size_t relative=i>=16?i-16:0,row=relative/oStride,col=relative%oStride;bool active=payload&&col<n&&(selection.mask&(1ul<<row));
     if(active){float value;std::memcpy(&value,gold.data()+i,4);Need(std::isfinite(value)&&gold[i]!=G,"Legacy control omitted/nonfinite output");}
     else{bool zero=selection.indexed&&ids.size()<selection.physical&&i>=16&&i<16+(selection.physical-1)*oStride+n;Need(gold[i]==(zero?0u:G),"Legacy control changed guard/inactive parent");}
    }
    for(auto row:ids)for(size_t col:std::array<size_t,3>{0,n/2,n-1}){
     double sum=0,absolute=0;for(size_t kk=0;kk<k;kk++){size_t ci=col*(k/64)+kk/64;uint32_t q=(w[col*(k/8)+kk/8]>>((kk%8)*4))&15;float sw=Garnet::Float16ToFloat32(sc[ci]),bw=Garnet::Float16ToFloat32(bi[ci]),weight=std::fma(float(q),sw,bw);double product=double(((float*)x.contents)[16+row*aStride+kk])*double(weight);sum+=product;absolute+=std::abs(product);}
     float actual;std::memcpy(&actual,gold.data()+16+row*oStride+col,4);double fraction=std::abs(double(actual)-sum)/(2e-6+4e-6*absolute);Need(fraction<=1,"Fixed FP64 bound exceeded");report["fp64"].push_back({{"case",caseIndex},{"layer",layer},{"shape",shape.name},{"row",row},{"column",col},{"actual",actual},{"sum",sum},{"sum_abs",absolute},{"fraction",fraction}});
    }
    for(int variant:{4,9}){
     std::fill_n((uint32_t*)y.contents,yWords,G);encode(variant);Need(!std::memcmp(gold.data(),y.contents,y.length),"Full parent differs from legacy FP32 control");Need(Sha(x.contents,x.length)==xSha&&Sha(weights.contents,weights.length)==wSha&&Sha(scales.contents,scales.length)==sSha&&Sha(biases.contents,biases.length)==bSha&&Sha(packedWeights.contents,packedWeights.length)==pwSha&&Sha(packedScales.contents,packedScales.length)==psSha&&Sha(packedBiases.contents,packedBiases.length)==pbSha,"Read-only input/weight parents or guards changed");words+=yWords;
     J cell={{"case",caseIndex},{"layer",layer},{"shape",shape.name},{"inner",k},{"columns",n},{"physical_rows",selection.physical},{"mask",selection.mask},{"indexed",selection.indexed},{"variant",variant},{"parent_words",yWords},{"output_sha256",Sha(y.contents,y.length)},{"expected_sha256",goldSha},{"guards_exact",true},{"read_only_exact",true},{"raw_control",allControl}};
     if(layer==0&&((std::string(shape.name)=="gate"&&selection.indexed&&selection.mask==8)||(std::string(shape.name)=="down"&&!selection.indexed&&selection.physical==4))){std::string name="case"+std::to_string(caseIndex)+"-variant"+std::to_string(variant)+".u32";Save(dest/name,y.contents,y.length);cell["raw_output"]=name;std::string control="case"+std::to_string(caseIndex)+"-legacy.u32";if(!fs::exists(dest/control))Save(dest/control,gold.data(),y.length);cell["raw_control"]=control;}
     report["cases"].push_back(cell);
    }
   }else{
    for(int round=-5;round<rounds;round++)for(int at=0;at<2;at++){
     int index=(at+(round+5)%2)%2;if(((round+5)+order)&1)index=1-index;int variant=std::array<int,2>{0,9}[index];auto began=std::chrono::steady_clock::now();auto cb=encode(variant);double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-began).count(),gpu=(cb.GPUEndTime-cb.GPUStartTime)*1000;Need(std::isfinite(gpu)&&gpu>0&&wall>=gpu,"Invalid GPU/wall clock");
     if(round>=0)report["samples"].push_back({{"layer",layer},{"shape",shape.name},{"inner",k},{"columns",n},{"physical_rows",selection.physical},{"mask",selection.mask},{"indexed",selection.indexed},{"variant",variant},{"round",round},{"position",at},{"gpu_ms",gpu},{"gpu_start_seconds",cb.GPUStartTime},{"gpu_end_seconds",cb.GPUEndTime},{"wall_including_pressure_ms",wall},{"pressure_gpu_start_seconds",pressureStart},{"pressure_gpu_end_seconds",pressureEnd}});
    }
   }++caseIndex;
  }
  Report(dest/"result1.json",report);std::cout<<J{{"layer",layer},{"shape",shape.name},{"columns",n},{"cases",report["cases"].size()},{"samples",report["samples"].size()}}.dump()<<'\n'<<std::flush;
 }}
 report.update({{"native_c1_control_variant",0},{"contiguous_layout",!padded},{"case_combinations",caseIndex},{"complete_parent_word_comparisons",words},{"maximum_device_allocated_bytes_before_close",device.currentAllocatedSize}});
 }}
 size_t after=device.currentAllocatedSize;Need(after<=baseline+4*1024*1024,"Metal buffers retained after close");report.update({{"completed",true},{"after_close_allocated_bytes",after},{"no_runtime_integration",true},{"serving_speed_claim",false},{"goal_complete",false}});Report(dest/"result1.json",report);std::cout<<J{{"completed",true},{"cases",report["cases"].size()},{"samples",report["samples"].size()},{"after_close_allocated_bytes",after}}.dump()<<'\n';return 0;
}}catch(const std::exception& e){report["error"]=e.what();if(!dest.empty()&&fs::exists(dest))Report(dest/"result1.json",report);std::cerr<<e.what()<<'\n';return 1;}}
