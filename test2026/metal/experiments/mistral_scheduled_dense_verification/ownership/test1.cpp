// SPDX-License-Identifier: Apache-2.0
#include "metal_context.h"
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
using namespace Garnet;
size_t checks=0,words=0,rejections=0,splitDispatches=0;
void Need(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
std::vector<float> Values(size_t n,uint32_t seed){std::vector<float> a(n);for(auto& x:a){seed=1664525u*seed+1013904223u;x=float(int((seed>>8)%65521)-32760)*.00001f;}return a;}
void Equal(const std::vector<float>& a,const std::vector<float>& b,const char* why){Need(a.size()==b.size()&&!std::memcmp(a.data(),b.data(),a.size()*4),why);words+=a.size();}
template<class F> void Reject(F fn,const char* why){bool rejected=false;try{fn();}catch(const std::logic_error&){rejected=true;}Need(rejected,why);++rejections;}
void Route(MetalContext& gpu,bool on){
#ifdef GARNET_METAL_MISTRAL_VERIFY_SPLIT_SCORES
 gpu.VerificationSetSplitScoresEnabled(on);
#else
 (void)gpu;(void)on;
#endif
}
int main(){try{{MetalContext gpu,foreign;
#ifdef GARNET_METAL_MISTRAL_VERIFY_SPLIT_SCORES
 constexpr size_t expected=1;
#else
 constexpr size_t expected=0;
 Reject([&]{gpu.VerificationSetSplitScoresEnabled(true);},"OFF build accepted split route");
#endif
 constexpr size_t qw=4096,kw=1024,cw=8*4096*128,pw=32*4096;
 auto q=Values(4*qw,13),k=Values(4*kw,37),v=Values(4*kw,97),kc=Values(cw,31),vc=Values(cw,71),gate=Values(4*qw,131);
 for(auto item:std::array<std::array<size_t,2>,6>{{{2048,1},{2048,2},{2048,3},{2048,4},{4093,3},{4095,1}}})for(bool flush:{false,true}){
  auto first=item[0],length=item[1];std::vector<float> slots(4,std::numeric_limits<float>::quiet_NaN()),active(4,0);for(size_t t=0;t<length;++t){slots[t]=float(first+t);active[t]=1;}
  std::vector<MetalMatrix> inputs{gpu.Upload(q,4,qw),gpu.Upload(k,4,kw),gpu.Upload(v,4,kw),gpu.Upload(kc,1,cw),gpu.Upload(vc,1,cw),gpu.Upload(slots,1,4),gpu.Upload(active,1,4),gpu.Upload(gate,4,qw)};
  std::vector<std::vector<float>> before;for(auto& x:inputs)before.push_back(gpu.Download(x));
  auto probabilities=gpu.Allocate(4,pw),output=gpu.Allocate(4,qw);
  auto run=[&](MetalContext& c){c.EncodeVerificationAttention(inputs,probabilities,output,32,8,128,4096,4,false);};
  Route(gpu,false);gpu.Begin();run(gpu);gpu.SubmitAndWait();auto expectedP=gpu.Download(probabilities),expectedO=gpu.Download(output);
  Need(gpu.LastVerificationSplitScoreDispatches()==0,"Disabled control used split scores");
  Route(gpu,true);const auto ownerBefore=MetalContext::ProfileOwnerStats();gpu.Begin();run(gpu);
  Need(gpu.LastVerificationSplitScoreDispatches()==expected&&gpu.LastVerificationAttentionDispatches()==1,"New B1/T4 route missing");splitDispatches+=expected;
  auto ownerAfter=MetalContext::ProfileOwnerStats();Need(ownerBefore.matrices==ownerAfter.matrices,"Split dispatch allocated matrix owner");
  Reject([&]{gpu.Download(probabilities);},"Pending probabilities readable");Reject([&]{gpu.Download(output);},"Pending context readable");
  Reject([&]{gpu.Begin();},"Duplicate Begin accepted");Need(gpu.LastVerificationSplitScoreDispatches()==expected,"Rejected Begin reset split counter");
  foreign.Begin();Reject([&]{run(foreign);},"Foreign pending writer accepted");Need(foreign.LastVerificationAttentionDispatches()==0,"Foreign rejection changed foreign dispatch");foreign.Abort();
  if(flush){gpu.Flush();Need(gpu.LastPartialSubmissions()==1,"New split work not partially submitted");foreign.Begin();Reject([&]{run(foreign);},"Foreign partially submitted writer accepted");foreign.Abort();}
#ifdef GARNET_METAL_MISTRAL_VERIFY_SPLIT_SCORES
  Reject([&]{gpu.VerificationSetSplitScoresEnabled(false);},"Open/partial batch changed split policy");
#endif
  gpu.Abort();Need(gpu.LastVerificationSplitScoreDispatches()==0&&gpu.LastVerificationAttentionDispatches()==0,"Abort retained split counters");
  Reject([&]{gpu.Download(probabilities);},"Aborted probabilities readable");Reject([&]{gpu.Download(output);},"Aborted context readable");
  for(size_t i=0;i<inputs.size();++i)Equal(gpu.Download(inputs[i]),before[i],"Abort mutated readonly source");
  Route(gpu,true);gpu.Begin();run(gpu);gpu.SubmitAndWait();Need(gpu.LastVerificationSplitScoreDispatches()==expected,"Recovery did not use expected split route");splitDispatches+=expected;
  Equal(gpu.Download(probabilities),expectedP,"Recovery changed complete probabilities");Equal(gpu.Download(output),expectedO,"Recovery changed complete context");
  for(size_t i=0;i<inputs.size();++i)Equal(gpu.Download(inputs[i]),before[i],"Recovery mutated readonly source");
  ++checks;
 }
 }const auto owners=MetalContext::ProfileOwnerStats();Need(!owners.matrices&&!owners.affineWeights&&!owners.bf16Weights&&!owners.pipelineSets,"Ownership test retained native owners");
 std::cout<<"split_ownership PASS cases="<<checks<<" fp32_parent_words="<<words<<" rejections="<<rejections<<" split_dispatches="<<splitDispatches<<" owners=0\n";Need(checks==12,"Missing ownership cases");return 0;
}catch(const std::exception& e){std::cerr<<"split_ownership FAILED "<<e.what()<<'\n';return 1;}}
