// SPDX-License-Identifier: Apache-2.0
// Standalone rejected research candidate; not a runtime kernel or CTest target.
// Build from the repository root:
// xcrun clang++ -std=c++17 -O3 -fobjc-arc -I. -Isrc/backends/metal \
//   -framework Foundation -framework Metal \
//   test2026/metal/experiments/mistral_thin_mpp_probe.mm -o /tmp/mistral-thin-mpp
// Run serially: /tmp/mistral-thin-mpp NEW_OUTPUT_DIRECTORY 15
// stdout is JSONL; every warmup/sample output is checked against full FP64 sums.
// A 256 MiB sweep is a cache pressure control, not a proof of cold caches.
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
#include <limits>
#include <sstream>
#include <vector>
#include "affine_int4_profile_shader.h"
#include "affine_int4_shader.h"
#include "affine_int4_blocked_decode_shader.h"
#include "src/model/float16_codec.h"
#include "third_party/nlohmann/json.hpp"
using json=nlohmann::json;
void Need(bool b,const char* why){if(!b)throw std::runtime_error(why);}
struct Params{uint64_t rows,inner,columns,aStride,oStride;};
struct Variant{std::string label;uint32_t values,outputs,threads;id<MTLComputePipelineState> pipeline;};
static const char* thin=R"METAL(
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
struct ThinParams{ulong rows,inner,columns,aStride,oStride;};
#define THIN_KERNEL(NAME,ROWS) \
kernel void NAME(device const float* a [[buffer(0)]],device const uint* packed [[buffer(1)]], \
 device const ushort* scales [[buffer(2)]],device const ushort* biases [[buffer(3)]],device float* out [[buffer(4)]], \
 constant ThinParams& runtime [[buffer(5)]],uint group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) { \
 GARNET_AFFINE_PARAMETERS(ThinParams); \
 constexpr auto desc=matmul2d_descriptor(ROWS,32,16,false,true,false,matmul2d_descriptor::mode::multiply_accumulate); \
 matmul2d<desc,execution_simdgroup> multiply; \
 auto left=multiply.get_left_input_cooperative_tensor<float,float,float>(); \
 auto right=multiply.get_right_input_cooperative_tensor<float,float,float>(); \
 auto accumulator=multiply.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>(); \
 for(ushort i=0;i<accumulator.get_capacity();++i)accumulator[i]=0; \
 const uint firstColumn=group*32; \
 for(uint block=0;block<p.inner;block+=16){ \
  for(ushort i=0;i<left.get_capacity();++i){auto coord=left.get_multidimensional_index(i);uint k=block+coord[0],row=coord[1];left[i]=row<p.rows?a[row*p.aStride+k]:0;} \
  for(ushort i=0;i<right.get_capacity();++i){auto coord=right.get_multidimensional_index(i);uint k=block+coord[0],n=firstColumn+coord[1];uint word=0;float scale=0,bias=0;if(n<p.columns){word=packed[n*(p.inner/8)+k/8];uint coefficient=n*(p.inner/64)+k/64;scale=garnet_affine_coefficient(scales[coefficient]);bias=garnet_affine_coefficient(biases[coefficient]);}right[i]=fma(float((word>>(4*(k%8)))&15u),scale,bias);} \
  multiply.run(left,right,accumulator); \
 } \
 for(ushort i=0;i<accumulator.get_capacity();++i){auto coord=accumulator.get_multidimensional_index(i);uint row=coord[1],n=firstColumn+coord[0];if(row<p.rows&&n<p.columns)out[row*p.oStride+n]=accumulator[i];} \
}

#define HOIST_KERNEL(NAME,ROWS,TILEK) \
kernel void NAME(device const float* a [[buffer(0)]],device const uint* packed [[buffer(1)]], \
 device const ushort* scales [[buffer(2)]],device const ushort* biases [[buffer(3)]],device float* out [[buffer(4)]], \
 constant ThinParams& runtime [[buffer(5)]],uint group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) { \
 GARNET_AFFINE_PARAMETERS(ThinParams); \
 constexpr auto desc=matmul2d_descriptor(ROWS,32,TILEK,false,true,false,matmul2d_descriptor::mode::multiply_accumulate); \
 matmul2d<desc,execution_simdgroup> multiply; \
 auto left=multiply.get_left_input_cooperative_tensor<float,float,float>(); \
 auto right=multiply.get_right_input_cooperative_tensor<float,float,float>(); \
 auto accumulator=multiply.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>(); \
 const ushort leftCount=left.get_capacity(),rightCount=right.get_capacity(),accCount=accumulator.get_capacity(); \
 if(leftCount>128 || rightCount>128 || accCount>128)return; \
 ushort2 leftCoords[128],rightCoords[128],accCoords[128]; \
 for(ushort i=0;i<leftCount;++i){auto c=left.get_multidimensional_index(i);leftCoords[i]=ushort2(c[0],c[1]);} \
 for(ushort i=0;i<rightCount;++i){auto c=right.get_multidimensional_index(i);rightCoords[i]=ushort2(c[0],c[1]);} \
 for(ushort i=0;i<accCount;++i){auto c=accumulator.get_multidimensional_index(i);accCoords[i]=ushort2(c[0],c[1]);accumulator[i]=0;} \
 const uint firstColumn=group*32; \
 for(uint block=0;block<p.inner;block+=TILEK){ \
  for(ushort i=0;i<leftCount;++i){auto coord=leftCoords[i];uint k=block+coord[0],row=coord[1];left[i]=row<p.rows?a[row*p.aStride+k]:0;} \
  for(ushort i=0;i<rightCount;++i){auto coord=rightCoords[i];uint k=block+coord[0],n=firstColumn+coord[1];uint word=0;float scale=0,bias=0;if(n<p.columns){word=packed[n*(p.inner/8)+k/8];uint coefficient=n*(p.inner/64)+k/64;scale=garnet_affine_coefficient(scales[coefficient]);bias=garnet_affine_coefficient(biases[coefficient]);}right[i]=fma(float((word>>(4*(k%8)))&15u),scale,bias);} \
  multiply.run(left,right,accumulator); \
 } \
 for(ushort i=0;i<accCount;++i){auto coord=accCoords[i];uint row=coord[1],n=firstColumn+coord[0];if(row<p.rows&&n<p.columns)out[row*p.oStride+n]=accumulator[i];} \
}

HOIST_KERNEL(affine_q4_thin16_cached,16,16)

HOIST_KERNEL(affine_q4_thin32_cached,16,32)

THIN_KERNEL(affine_q4_thin16,16)
)METAL";
float half(uint16_t value){return Garnet::Float16ToFloat32(value);}
int main(int argc,char** argv){@autoreleasepool{try{
 Need(argc==3,"Usage NEW_OUTPUT REPS");const auto root=std::filesystem::path(argv[1]);Need(std::filesystem::create_directory(root),"Output exists");const int reps=std::stoi(argv[2]);Need(reps>=3&&reps<=51,"Invalid reps");
 auto device=MTLCreateSystemDefaultDevice();auto queue=[device newCommandQueue];NSError* error=nil;
 std::string source="#define GARNET_AFFINE_FP16_COEFFICIENTS 1\n#define GARNET_AFFINE_PROFILE 1\n#define GARNET_AFFINE_INDEX32 1\n";
 source+=Garnet::kAffineInt4ProfilePrelude;source+=Garnet::kAffineInt4Shader;
 source+="\n#define GARNET_BLOCKED_ROWS affine_decode_rows\n#define GARNET_BLOCKED_COLUMNS 4\n";source+=Garnet::kAffineInt4BlockedDecodeShader;source+=thin;
 auto options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;options.languageVersion=MTLLanguageVersion4_0;
 auto lib=[device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()] options:options error:&error];Need(lib!=nil,error?error.localizedDescription.UTF8String:"compile error");
 NSString* flushSource=@"#include <metal_stdlib>\nusing namespace metal;kernel void sweep(device const float* a [[buffer(0)]],device float* b [[buffer(1)]],uint id [[thread_position_in_grid]]){float sum=0;for(uint j=0;j<256;++j)sum+=a[id*256+j];b[id]=sum;}";
 auto flib=[device newLibraryWithSource:flushSource options:nil error:&error];auto flush=[device newComputePipelineStateWithFunction:[flib newFunctionWithName:@"sweep"] error:&error];Need(flush!=nil,"sweep compile failed");
 auto trash=[device newBufferWithLength:256ull*1024*1024 options:MTLResourceStorageModeShared];auto sink=[device newBufferWithLength:1024*1024 options:MTLResourceStorageModeShared];std::fill_n((float*)trash.contents,64*1024*1024,.125f);
 auto sweep=[&](){auto command=[queue commandBuffer];auto enc=[command computeCommandEncoder];[enc setComputePipelineState:flush];[enc setBuffer:trash offset:0 atIndex:0];[enc setBuffer:sink offset:0 atIndex:1];[enc dispatchThreads:MTLSizeMake(256*1024,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[enc endEncoding];[command commit];[command waitUntilCompleted];Need(command.status==MTLCommandBufferStatusCompleted,"sweep failed");};
 std::cout<<json{{"kind","metadata"},{"device",device.name.UTF8String},{"warmups",3},{"reps",reps},{"scope","Synthetic thin FP32 MPP four-row Mistral projections; changes reduction order. Independent full-output FP64 bound. Sweep is not proof of cold cache. No model speed qualification."}}.dump()<<'\n'<<std::flush;
 size_t compared=0,refWords=0,index=0;
 for(auto shape:std::vector<std::array<uint32_t,2>>{{4096,4096},{1024,4096},{14336,4096},{4096,14336},{32768,4096}}){@autoreleasepool{
  const uint32_t n=shape[0],k=shape[1];Params p{4,k,n,k,n};auto buffer=[&](size_t bytes){auto b=[device newBufferWithLength:bytes options:MTLResourceStorageModeShared];Need(b!=nil,"allocation failed");return b;};
  auto a=buffer(4*k*4),w=buffer(size_t(n)*k/2),s=buffer(size_t(n)*k/32),b=buffer(size_t(n)*k/32),out=buffer(4*n*4);
  uint32_t seed=37;for(uint32_t i=0;i<4*k;++i){seed=1664525u*seed+1013904223u;((float*)a.contents)[i]=float(int((seed>>8)%65521)-32760)*0x1p-18f;}
  for(size_t i=0;i<size_t(n)*k/8;++i)((uint32_t*)w.contents)[i]=uint32_t(i)*2654435761u+131u;
  for(size_t i=0;i<size_t(n)*k/64;++i){((uint16_t*)s.contents)[i]=uint16_t(0x2c00+i%8);((uint16_t*)b.contents)[i]=uint16_t(0xb000+i%7);}
  std::vector<Variant> variants{{"ordinary",0,1,128,nil},{"blocked4",0,4,128,nil},{"mpp16",16,32,32,nil},{"mpp16_cached",17,32,32,nil},{"mpp32_cached",18,32,32,nil}};
  for(auto& variant:variants){auto constants=[MTLFunctionConstantValues new];const uint32_t dims[]{k,n,k,n};for(NSUInteger i=0;i<4;++i)[constants setConstantValue:dims+i type:MTLDataTypeUInt atIndex:10+i];uint32_t row=4;[constants setConstantValue:&row type:MTLDataTypeUInt atIndex:0];
   NSString* name=variant.values?(variant.values==19?@"affine_q4_thin64_cached":variant.values==18?@"affine_q4_thin32_cached":variant.values==17?@"affine_q4_thin16_cached":@"affine_q4_thin16"):variant.outputs==1?@"affine_q4_decode":@"affine_q4_blocked";
   auto fn=[lib newFunctionWithName:name constantValues:constants error:&error];Need(fn!=nil,error?error.localizedDescription.UTF8String:"specialization failed");variant.pipeline=[device newComputePipelineStateWithFunction:fn error:&error];Need(variant.pipeline!=nil,"pipeline failed");Need(variant.pipeline.maxTotalThreadsPerThreadgroup>=variant.threads,"pipeline limit");
  }
  auto run=[&](const Variant& variant){const auto start=std::chrono::steady_clock::now();auto command=[queue commandBuffer];auto enc=[command computeCommandEncoder];[enc setComputePipelineState:variant.pipeline];[enc setBuffer:a offset:0 atIndex:0];[enc setBuffer:w offset:0 atIndex:1];[enc setBuffer:s offset:0 atIndex:2];[enc setBuffer:b offset:0 atIndex:3];[enc setBuffer:out offset:0 atIndex:4];[enc setBytes:&p length:sizeof(p) atIndex:5];size_t tile=(variant.threads/32)*variant.outputs;[enc dispatchThreadgroups:MTLSizeMake((n+tile-1)/tile,1,1) threadsPerThreadgroup:MTLSizeMake(variant.threads,1,1)];[enc endEncoding];[command commit];[command waitUntilCompleted];Need(command.status==MTLCommandBufferStatusCompleted,"projection failed");return std::array<double,2>{(command.GPUEndTime-command.GPUStartTime)*1000.,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()};};
  // Compute full FP64 bounds once, outside all timed GPU spans.
  std::vector<double> expected(4*n),limits(4*n);
  for(uint32_t col=0;col<n;++col){std::array<double,4> sum{},absolute{};for(uint32_t j=0;j<k;++j){const size_t c=size_t(col)*(k/64)+j/64;uint32_t q=(((uint32_t*)w.contents)[size_t(col)*(k/8)+j/8]>>(4*(j%8)))&15;float weight=std::fma(float(q),half(((uint16_t*)s.contents)[c]),half(((uint16_t*)b.contents)[c]));for(uint32_t row=0;row<4;++row){double product=double(((float*)a.contents)[row*k+j])*weight;sum[row]+=product;absolute[row]+=std::abs(product);}}for(uint32_t row=0;row<4;++row){expected[row*n+col]=sum[row];limits[row*n+col]=2e-6+4e-6*absolute[row];}}refWords+=4*n;
  std::vector<float> reference(4*n);run(variants[0]);std::memcpy(reference.data(),out.contents,4*n*4);
  for(bool swept:{false,true}){std::vector<std::vector<double>> gpu(variants.size()),wall(variants.size());std::vector<double> maxError(variants.size()),maxDifference(variants.size());std::vector<size_t> unequal(variants.size());
   for(int rep=-3;rep<reps;++rep)for(size_t order=0;order<variants.size();++order){size_t v=(order+size_t(rep+3))%variants.size();if((rep+3)&1)v=variants.size()-1-v;if(swept)sweep();auto timing=run(variants[v]);auto actual=(float*)out.contents;
    for(uint32_t c=0;c<4*n;++c){Need(std::isfinite(actual[c]),"nonfinite result");double error=std::abs(double(actual[c])-expected[c])/limits[c];if(error>1.){std::cerr<<"bad variant="<<variants[v].label<<" N="<<n<<" K="<<k<<" index="<<c<<" actual="<<actual[c]<<" expected="<<expected[c]<<" bound="<<limits[c]<<" fraction="<<error<<"\n";}Need(error<=1.,"original FP64 tolerance exceeded");maxError[v]=std::max(maxError[v],error);maxDifference[v]=std::max(maxDifference[v],std::abs(double(actual[c])-reference[c]));if(rep==0&&std::memcmp(actual+c,reference.data()+c,4))++unequal[v];}compared+=4*n;
    if(rep>=0){gpu[v].push_back(timing[0]);wall[v].push_back(timing[1]);}
   }
   for(size_t v=0;v<variants.size();++v){run(variants[v]);const auto file="case-"+std::to_string(index)+"-"+(swept?"swept":"hot")+"-"+variants[v].label+".bin";std::ofstream f(root/file,std::ios::binary);f.write((char*)out.contents,4*n*4);Need(bool(f),"output write failed");
    std::cout<<json{{"kind","cell"},{"index",index},{"outputs",n},{"inner",k},{"rows",4},{"variant",variants[v].label},{"swept",swept},{"max_error_fraction",maxError[v]},{"maximum_difference_from_control",maxDifference[v]},{"nonidentical_words",unequal[v]},{"gpu_ms",gpu[v]},{"wall_ms",wall[v]},{"output",file}}.dump()<<'\n'<<std::flush;
   }
  }++index;
 }}
 std::cout<<json{{"kind","closed"},{"cases",index},{"checked_words",compared},{"independent_fp64_outputs",refWords}}.dump()<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}}
