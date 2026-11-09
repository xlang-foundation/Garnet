// SPDX-License-Identifier: Apache-2.0
#include "metal_context.h"
#include "../../src/backends/metal/split_score_policy.h"
#include <vector>
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <limits>
#include <algorithm>
using namespace Garnet;
void Need(bool yes,const char* why){if(!yes)throw std::runtime_error(why);}
int main(){try{
 MetalContext gpu;size_t cases=0,words=0;
#ifdef GARNET_METAL_SPLIT_SCORE_PROFILE32
 uint64_t profile[]{4,32,8,128,4096,4096,4194304,4,4,131072};
 Need(Profile32ScoreSupported(profile,2048),"Valid profile32 rejected");
 for(size_t field:{5u,6u,7u,8u,9u}){uint64_t saved=profile[field];profile[field]=0x100000000ull;Need(!Profile32ScoreSupported(profile,2048),"Profile32 index overflow admitted");profile[field]=saved;}
 for(size_t field:{5u,6u,9u}){++profile[field];Need(!Profile32ScoreSupported(profile,2048),"Wrong profile32 layout admitted");--profile[field];}
 Need(!Profile32ScoreSupported(nullptr,2048),"Null profile32 admitted");
#endif
 Need(!SplitScoreProfileSupported(3,32,8,128,4096,2048)&&!SplitScoreProfileSupported(4,16,8,128,4096,2048)&&!SplitScoreProfileSupported(4,32,4,128,4096,2048)&&!SplitScoreProfileSupported(4,32,8,64,4096,2048)&&!SplitScoreProfileSupported(4,32,8,128,4097,2048)&&!SplitScoreProfileSupported(4,32,8,128,4096,511)&&!SplitScoreProfileSupported(4,32,8,128,4096,4097),"Unsupported split-score policy admitted");
 for(size_t cap:{2053u,4096u})for(size_t batch:{1u,2u,4u})for(size_t visible:{127u,512u,2048u})for(bool mixed:{false,true}){
  constexpr size_t heads=32,kv=8,dim=128;std::vector<float> q(batch*heads*dim),k(batch*kv*cap*dim),slots(batch,float(visible-1)),flags(batch,1.f);uint32_t seed=37;auto next=[&](){seed=1664525u*seed+1013904223u;return float(int((seed>>8)%65521)-32760)*.000004f;};for(auto& x:q)x=next();for(auto& x:k)x=next();if(mixed&&batch>1){flags[1]=0;slots[1]=std::numeric_limits<float>::quiet_NaN();}
  auto query=gpu.Upload(q,batch,heads*dim),key=gpu.Upload(k,batch,kv*cap*dim),slot=gpu.Upload(slots,batch,1),active=gpu.Upload(flags,batch,1),out=gpu.Allocate(batch,heads*cap);
  gpu.SetSplitScoresEnabled(false);gpu.Begin();gpu.EncodeAttentionScores(query,key,slot,active,out,heads,kv,dim,cap);gpu.SubmitAndWait();auto reference=gpu.Download(out);Need(gpu.LastSplitScoreDispatches()==0,"Disabled route used split scores");
  gpu.SetSplitScoresEnabled(true);gpu.Begin();gpu.EncodeAttentionScores(query,key,slot,active,out,heads,kv,dim,cap);gpu.SubmitAndWait();auto candidate=gpu.Download(out);Need(reference.size()==candidate.size()&&!std::memcmp(reference.data(),candidate.data(),reference.size()*4),"Native split-score probability bits differ");Need(gpu.LastSplitScoreDispatches()==size_t(visible>=512),"Native split-score policy/dispatch differs");
#ifdef GARNET_METAL_SPLIT_SCORE_PROFILE32
  gpu.SetProfile32ScoresEnabled(true);gpu.Begin();gpu.EncodeAttentionScores(query,key,slot,active,out,heads,kv,dim,cap);gpu.SubmitAndWait();auto profiled=gpu.Download(out);Need(!std::memcmp(reference.data(),profiled.data(),reference.size()*4),"Native profile32 probability bits differ");Need(gpu.LastProfile32ScoreDispatches()==size_t(visible>=512&&cap==4096),"Native profile32 dispatch differs");gpu.SetProfile32ScoresEnabled(false);
#endif
  auto zero=gpu.Upload(std::vector<float>(batch,0),batch,1),pending=gpu.Allocate(batch,1);gpu.Begin();gpu.EncodeAdd(slot,zero,pending);gpu.EncodeAttentionScores(query,key,pending,active,out,heads,kv,dim,cap);gpu.SubmitAndWait();Need(gpu.LastSplitScoreDispatches()==0,"GPU-pending controls incorrectly used host split policy");auto fallback=gpu.Download(out);Need(!std::memcmp(reference.data(),fallback.data(),reference.size()*4),"Pending-control fallback changed output bits");
  gpu.Begin();bool rejected=false;try{gpu.SetSplitScoresEnabled(false);}catch(const std::logic_error&){rejected=true;}Need(rejected,"Split-score policy changed in open batch");gpu.Abort();++cases;words+=reference.size();
 }
 std::cout<<"Native split scores:36 ordinary/mixed cases, exact probabilities, pending-control fallback and policy guards PASS "<<words<<" words\n";Need(cases==36,"Missing native split cases");return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
