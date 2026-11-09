// SPDX-License-Identifier: Apache-2.0
#include "../../src/backends/metal/kv_commit_hoist_policy.h"
#include <iostream>
#include <stdexcept>
using namespace Garnet;
int main(){try{
 const std::array<size_t,6> ids{0,1,2,3,4,5};const std::array<std::vector<size_t>,2> outputs{{{10},{11}}};
 std::vector<KVHoistNode> baseline;
 for(size_t out:{size_t(8),size_t(6),size_t(7)})baseline.push_back({KVHoistNode::Other,true,{0},{out},{},{}});
 baseline.push_back({KVHoistNode::Attention,true,{8,6,7,1,2,4,5},{9},{8,6,7,1,2,4,5,8},{32,8,128,4096,128,0}});
 baseline.push_back({KVHoistNode::Commit,true,{6,1,4,5,3},{10},{6,1,4,5,3},{8,128,4096,128}});
 baseline.push_back({KVHoistNode::Commit,true,{7,2,4,5,3},{11},{7,2,4,5,3},{8,128,4096,128}});
 baseline.push_back({KVHoistNode::Other,true,{9},{14},{},{}});
 auto require=[](bool b){if(!b)throw std::runtime_error("KV hoist policy failure");};
 auto valid=[&](const auto& n){return ProveKVCommitHoist(n,ids,outputs,4,128).Valid();};
 auto plan=ProveKVCommitHoist(baseline,ids,outputs,4,128);require(plan.Valid()&&plan.attention==3&&plan.commits==std::array<size_t,2>{4,5});size_t rejected=0;
 auto reject=[&](auto change){auto n=baseline;change(n);require(!valid(n));++rejected;};
 reject([](auto& n){n[4].resident=false;});reject([](auto& n){n[5].kind=KVHoistNode::Other;});
 reject([](auto& n){n.push_back(n[3]);});reject([](auto& n){n[3].dims[5]=1;});
 for(size_t at=0;at<6;++at)reject([&](auto& n){++n[3].dims[at];});
 for(size_t at=0;at<5;++at)reject([&](auto& n){n[4].args[at]=99;});
 for(size_t at=0;at<4;++at)reject([&](auto& n){++n[5].dims[at];});
 reject([](auto& n){n[4].writes={10,20};});reject([](auto& n){n[5].writes={10};});
 reject([](auto& n){n[6].reads.push_back(1);});reject([](auto& n){n[6].reads.push_back(2);});
 reject([](auto& n){n.insert(n.begin()+4,{KVHoistNode::Other,true,{1},{20},{},{}});});
 reject([](auto& n){n.insert(n.begin()+4,{KVHoistNode::Other,true,{11},{20},{},{}});});
 reject([](auto& n){n.insert(n.begin()+4,{KVHoistNode::Other,true,{0},{6},{},{}});});
 reject([](auto& n){n.insert(n.begin()+4,{KVHoistNode::Other,true,{0},{4},{},{}});});
 reject([](auto& n){n.push_back({KVHoistNode::Other,true,{0},{1},{},{}});});
 reject([](auto& n){n.push_back(n[4]);});reject([](auto& n){n[3].args[1]=8;});
 reject([](auto& n){n[0].resident=false;});reject([](auto& n){n.push_back(n[1]);});
 for(size_t b:{size_t(0),size_t(5)}){require(!ProveKVCommitHoist(baseline,ids,outputs,b,128).Valid());++rejected;}
 for(size_t t:{size_t(0),size_t(1),size_t(129),size_t(512)}){require(!ProveKVCommitHoist(baseline,ids,outputs,4,t).Valid());++rejected;}
 auto other=outputs;other[1]={10};require(!ProveKVCommitHoist(baseline,ids,other,4,128).Valid());++rejected;
 auto duplicate=ids;duplicate[2]=duplicate[1];require(!ProveKVCommitHoist(baseline,duplicate,outputs,4,128).Valid());++rejected;
 for(const auto& shape:std::vector<std::pair<size_t,size_t>>{{1,256},{2,256},{1,512}}){
  auto n=baseline;const auto b=shape.first,t=shape.second;n[3].dims[4]=t;n[4].dims[3]=n[5].dims[3]=t;
  require(!ProveKVCommitHoist(n,ids,outputs,b,t).Valid());
  require(ProveKVCommitHoist(n,ids,outputs,b,t,512).Valid());
  n[6].reads.push_back(1);require(!ProveKVCommitHoist(n,ids,outputs,b,t,512).Valid());++rejected;
 }
 for(size_t maximum:{size_t(0),size_t(129),size_t(513),SIZE_MAX}){require(!ProveKVCommitHoist(baseline,ids,outputs,4,128,maximum).Valid());++rejected;}
 auto tooMany=baseline;tooMany[3].dims[4]=512;tooMany[4].dims[3]=tooMany[5].dims[3]=512;
 require(!ProveKVCommitHoist(tooMany,ids,outputs,2,512,512).Valid());++rejected;
 auto earlier=baseline;earlier.insert(earlier.begin(),{KVHoistNode::Other,true,{1,2},{20},{},{}});require(valid(earlier));
 auto updated=baseline;updated.push_back({KVHoistNode::Other,true,{10,11},{20},{},{}});require(valid(updated));
 std::cout<<"Immutable KV hoist proof: baseline/prior-reader/updated-reader accepted; "<<rejected<<" invalid alias/consumer/producer/order/shape/layout cases rejected\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
