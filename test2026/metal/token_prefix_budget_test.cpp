// SPDX-License-Identifier: Apache-2.0
#include "token_prefix_profile.h"
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace Garnet;
void Need(bool b){if(!b)throw std::runtime_error("prefix budget arithmetic mismatch");}
int main(){try{
 constexpr uint64_t M=1024*1024,R=256*M,H=kTokenPrefixHostLimitBytes;
 size_t checks=0;
 for(unsigned copies:{1u,2u}){
  const uint64_t used=8ull*1024*M,need=used+R+copies*H;
  Need(TokenPrefixHostMemoryFits(used,0,need,H,copies));checks++;
  Need(TokenPrefixHostMemoryFits(0,used,need,H,copies));checks++;
  Need(!TokenPrefixHostMemoryFits(used,0,need-1,H,copies));checks++;
  Need(!TokenPrefixHostMemoryFits(0,used,need-1,H,copies));checks++;
 }
 Need(!TokenPrefixHostMemoryFits(0,0,UINT64_MAX,H+1,1));checks++;
 Need(!TokenPrefixHostMemoryFits(0,0,UINT64_MAX,0,1));checks++;
 Need(!TokenPrefixHostMemoryFits(0,0,UINT64_MAX,H,0));checks++;
 Need(!TokenPrefixHostMemoryFits(0,0,UINT64_MAX,H,3));checks++;
 Need(!TokenPrefixHostMemoryFits(UINT64_MAX,0,UINT64_MAX,H,1));checks++;
 Need(!TokenPrefixHostMemoryFits(0,UINT64_MAX,UINT64_MAX,H,1));checks++;
 Need(!TokenPrefixHostMemoryFits(0,0,R,H,1));checks++;
 Need(!TokenPrefixHostMemoryFits(0,0,0,H,1));checks++;
 for(auto pair:{std::pair<uint64_t,uint64_t>{64,64},{128,2048},{1024,2051},{2048,2048},{2048,2112}}){Need(TokenPrefixIntermediateUseful(pair.first,pair.second));checks++;}
 for(auto pair:{std::pair<uint64_t,uint64_t>{0,2048},{63,2048},{65,2048},{2048,2049},{2048,2051},{2048,2111},{2048,1024}}){Need(!TokenPrefixIntermediateUseful(pair.first,pair.second));checks++;}
 std::cout<<"prefix host budget bounds PASS cases="<<checks<<" compiled_limit_bytes="<<H<<" exact_boundary=true physical_device_max=true overflow_safe=true\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
