// SPDX-License-Identifier: Apache-2.0
#include "metal_context.h"
#include "../../src/backends/metal/silu_profile_policy.h"
#include <array>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
using namespace Garnet;
void Need(bool v,const char* s){if(!v)throw std::runtime_error(s);}
template<class F>void Reject(F f){bool rejected=false;try{f();}catch(const std::exception&){rejected=true;}Need(rejected,"Expected guard rejection");}
int main(){try{
 Need(SiluProfile32Supported(512,14336,14344,14352,14336),"Padded up-to512 profile rejected");
 for(auto p:std::vector<std::array<uint64_t,5>>{{0,14336,14336,14336,14336},{513,14336,14336,14336,14336},{4,14335,14336,14336,14336},{4,14336,0,14336,14336},{4,14336,14336,14335,14336},{4,14336,14336,14336,0x100000000ull},{512,14336,0xffffffffull,14336,14336}})Need(!SiluProfile32Supported(p[0],p[1],p[2],p[3],p[4]),"Invalid uint32 extent admitted");
 size_t cases=0,words=0,samples=0;
 for(bool edges:{false,true})for(auto shape:std::vector<std::array<size_t,2>>{{1,14336},{4,14336},{128,14336},{256,14336},{512,14336},{1,4096},{4,4096},{128,4096},{256,4096},{512,4096},{128,14335},{513,4096}}){
  const size_t rows=shape[0],cols=shape[1],count=rows*cols;std::vector<float> a(count),b(count);uint32_t seed=37;
  for(size_t i=0;i<count;++i){seed=1664525u*seed+1013904223u;a[i]=float(int((seed>>8)%65521)-32760)*0x1p-10f;seed=1664525u*seed+1013904223u;b[i]=float(int((seed>>8)%65521)-32760)*0x1p-12f;}
  const uint32_t special[]{0,0x80000000u,1,0x80000001u,0x00800000u,0x80800000u,0x3f800000u,0xbf800000u,0x42b00000u,0xc2b00000u,0x42c80000u,0xc2c80000u,0x7f800000u,0xff800000u,0x7fc01234u,0x7fa01234u};
  if(edges)for(size_t i=0;i<count;i+=257){uint32_t value=special[(i/257)%16];std::memcpy(a.data()+i,&value,4);if((i/257)%3==0){value=special[(i/257+7)%16];std::memcpy(b.data()+i,&value,4);}}
  MetalContext gpu;auto ma=gpu.Upload(a,rows,cols),mb=gpu.Upload(b,rows,cols),mc=gpu.Allocate(rows,cols);auto owners=MetalContext::ProfileOwnerStats();
  // No setter: compile support does not enable this policy by default.
  gpu.Begin();gpu.EncodeSiluMul(ma,mb,mc);gpu.SubmitAndWait();Need(gpu.LastSiluProfile32Dispatches()==0,"SiLU32 default enabled");auto original=gpu.Download(mc);
  gpu.SetSiluProfile32Enabled(true);gpu.Begin();gpu.EncodeSiluMul(ma,mb,mc);gpu.SubmitAndWait();auto actual=gpu.Download(mc);bool eligible=rows<=512&&(cols==4096||cols==14336);Need(gpu.LastSiluProfile32Dispatches()==size_t(eligible),"Wrong native profile/fallback");Need(!std::memcmp(original.data(),actual.data(),count*4),"Native SiLU32 output bits differ");words+=count;
  for(size_t j=0;j<128;++j){size_t i=(j*104729+19)%count;if(!std::isfinite(a[i])||!std::isfinite(b[i]))continue;double expected=(double(a[i])/(1.+std::exp(-double(a[i]))))*double(b[i]);Need(std::abs(double(actual[i])-expected)<=2e-6+4e-6*std::abs(expected),"Finite-input FP64 bound");++samples;}
  auto after=MetalContext::ProfileOwnerStats();Need(after.matrices==owners.matrices&&after.pipelineSets==owners.pipelineSets,"Request-time native allocation");++cases;
 }
 {
  MetalContext gpu;const size_t rows=4,cols=4096;auto a=gpu.Upload(std::vector<float>(rows*cols,.25f),rows,cols),b=gpu.Upload(std::vector<float>(rows*cols,.5f),rows,cols),out=gpu.Upload(std::vector<float>(rows*cols,17.f),rows,cols);auto original=gpu.Download(out);gpu.SetSiluProfile32Enabled(true);
  auto unready=gpu.Allocate(rows,cols);gpu.Begin();Reject([&]{gpu.EncodeSiluMul(unready,b,out);});Reject([&]{gpu.SetSiluProfile32Enabled(false);});gpu.Abort();Need(gpu.Download(out)==original,"Unready input wrote output");
  gpu.Begin();Reject([&]{gpu.EncodeSiluMul(a,b,a);});gpu.Abort();Need(gpu.Download(out)==original,"Alias error wrote output");
  auto wrong=gpu.Upload(std::vector<float>(rows*3,1.f),rows,3);gpu.Begin();Reject([&]{gpu.EncodeSiluMul(a,wrong,out);});gpu.Abort();Need(gpu.Download(out)==original,"Bad shape wrote output");
  gpu.Begin();gpu.EncodeSiluMul(a,b,out);gpu.Abort();Reject([&]{gpu.Download(out);});Need(gpu.LastSiluProfile32Dispatches()==0,"Abort did not reset observer");
  gpu.Begin();gpu.EncodeSiluMul(a,b,out);gpu.SubmitAndWait();Need(gpu.LastSiluProfile32Dispatches()==1,"Recovery missed profile");
  gpu.Begin();gpu.EncodeSiluMul(a,b,out);gpu.Flush();Reject([&]{gpu.SetSiluProfile32Enabled(false);});gpu.SubmitAndWait();
  gpu.Begin();gpu.EncodeAdd(a,b,out);gpu.SubmitAndWait();Need(gpu.LastSiluProfile32Dispatches()==0,"Add selected SiLU profile");
  auto weight=gpu.Upload(std::vector<float>(cols,1.f),1,cols);gpu.Begin();gpu.EncodeRMSNorm(a,weight,out,1e-6f,0);gpu.SubmitAndWait();Need(gpu.LastSiluProfile32Dispatches()==0,"Norm selected SiLU profile");
 }
 auto owners=MetalContext::ProfileOwnerStats();Need(!owners.matrices&&!owners.pipelineSets&&!owners.affineWeights,"Native owners retained");Need(cases==24,"Missing profile cells");std::cout<<"Native SiLU32 PASS cases="<<cases<<" exact_words="<<words<<" finite_FP64_samples="<<samples<<" defaults/fallback/index/alias/unready/abort/flush/policy guards and owners checked\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
