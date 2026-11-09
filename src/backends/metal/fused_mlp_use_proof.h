// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <array>
#include <vector>
#include <algorithm>
#include <cstddef>
#include <cstdint>
namespace Garnet {
struct FusedMlpUseNode {std::vector<size_t> reads,writes;};
// IDs are immutable logical alias roots, before scratch lifetime rebinding.
inline bool ProveFusedMlpUses(const std::vector<FusedMlpUseNode>& nodes,size_t begin,
 const std::array<size_t,3>& intermediates,const std::vector<size_t>& observable) {
 if(begin>=nodes.size()||nodes.size()-begin<4)return false;
 for(size_t part=0;part<3;++part){
  size_t id=intermediates[part];for(size_t other=0;other<part;++other)if(id==intermediates[other])return false;
  if(std::find(observable.begin(),observable.end(),id)!=observable.end())return false;
  size_t writers=0,readers=0;
  for(size_t i=0;i<nodes.size();++i){
   for(auto write:nodes[i].writes)if(write==id){if(i!=begin+part)return false;++writers;}
   for(auto read:nodes[i].reads)if(read==id){if(i!=begin+(part==2?3:2))return false;++readers;}
  }
  if(writers!=1||readers!=1)return false;
 }
 return true;
}
}
