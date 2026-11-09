// SPDX-License-Identifier: Apache-2.0
#include "metal_context.h"
#include "../../src/backends/metal/value_profile_policy.h"
#include <vector>
#include <limits>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
using namespace Garnet;
void Need(bool b,const char* m){if(!b)throw std::runtime_error(m);}
int main(){try{
 uint64_t p[]{4,32,8,128,4096,131072,4194304,4096,4,4096,0};Need(ValueProfile32Supported(p),"Valid value profile rejected");Need(!ValueProfile32Supported(nullptr),"Null value profile accepted");
 for(size_t field=0;field<11;++field){auto saved=p[field];p[field]=(field>=5&&field<=9)?0x100000000ull:saved+1;Need(!ValueProfile32Supported(p),"Invalid shape/index/gate admitted");p[field]=saved;}
 MetalContext gpu;size_t cases=0,words=0;
 for(size_t batch:{1u,2u,4u})for(size_t cap:{2053u,4096u})for(size_t length:std::vector<size_t>{1,512,2048,cap})for(bool gated:{false,true})for(bool invalid:{false,true}){
  std::vector<float> prob(batch*32*cap),value(batch*8*cap*128),gate(batch*4096),slots(batch,float(length-1));uint32_t seed=37;auto next=[&](){seed=1664525u*seed+1013904223u;return float(int((seed>>8)%65521)-32760)*0x1p-14f;};for(auto& x:value)x=next();for(auto& x:gate)x=next();
  if(invalid)for(size_t row=0;row<batch;++row)slots[row]=std::vector<float>{std::numeric_limits<float>::quiet_NaN(),-1.f,std::numeric_limits<float>::infinity(),float(cap)}[row];
  for(size_t row=0;row<batch;++row)for(size_t h=0;h<32;++h)for(size_t j=0;j<(invalid?cap:length);++j)prob[(row*32+h)*cap+j]=(row==1?0.f:float(1.0/(invalid?cap:length)));
  auto pm=gpu.Upload(prob,batch,32*cap),vm=gpu.Upload(value,batch,8*cap*128),gm=gpu.Upload(gate,batch,4096),sm=gpu.Upload(slots,batch,1),out=gpu.Allocate(batch,4096);
  gpu.SetValueProfile32Enabled(false);gpu.Begin();gpu.EncodeAttentionContext(pm,vm,gm,sm,out,32,8,128,cap,gated);gpu.SubmitAndWait();auto expected=gpu.Download(out);Need(gpu.LastValueProfile32Dispatches()==0,"Disabled value profile selected");
  gpu.SetValueProfile32Enabled(true);gpu.Begin();gpu.EncodeAttentionContext(pm,vm,gm,sm,out,32,8,128,cap,gated);gpu.SubmitAndWait();auto actual=gpu.Download(out);Need(!std::memcmp(actual.data(),expected.data(),expected.size()*4),"Native value-profile output bits differ");Need(gpu.LastValueProfile32Dispatches()==size_t(cap==4096&&!gated),"Wrong native value-profile dispatch/fallback");
  gpu.Begin();bool rejected=false;try{gpu.SetValueProfile32Enabled(false);}catch(const std::logic_error&){rejected=true;}Need(rejected,"Value policy changed during open batch");gpu.Abort();++cases;words+=actual.size();
 }
 Need(cases==96,"Missing native value cases");std::cout<<"Native value profile:96 cases exact, C2053/gated fallback, invalid positions, zero-probability rows, index guards and policy lock PASS "<<words<<" words\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
