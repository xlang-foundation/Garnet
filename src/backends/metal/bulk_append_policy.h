// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <array>
#include <vector>
#include <cstdint>
#include <stdexcept>
namespace Garnet {
// Pure preflight: complete before any device write or tracked-history commit.
class BulkAppendValidator {
 std::array<size_t,4> starts_,counts_{};
 std::array<bool,4> valid_,ended_{};
 std::array<uint64_t,4> epochs_;
 size_t capacity_,vocabulary_;bool any_=false;
public:
 BulkAppendValidator(std::array<size_t,4> starts,std::array<bool,4> valid,std::array<uint64_t,4> epochs,size_t capacity,size_t vocabulary):starts_(starts),valid_(valid),epochs_(epochs),capacity_(capacity),vocabulary_(vocabulary) {
  if(!capacity||!vocabulary)throw std::invalid_argument("Missing native bulk token domain");
 }
 void Step(size_t row,float active,float slot,const float* pos,int64_t token) {
  if(row>=4||token<0||uint64_t(token)>=vocabulary_||(active!=0&&active!=1))throw std::invalid_argument("Invalid bulk active/token domain");
  if(active==0){ended_[row]=true;if(slot!=0||pos[0]!=0||pos[1]!=0||pos[2]!=0)throw std::invalid_argument("Inactive bulk controls must be zero");return;}
  if(!epochs_[row]||!valid_[row]||ended_[row]||starts_[row]>capacity_||counts_[row]>=capacity_-starts_[row])throw std::invalid_argument("Bulk row has no contiguous tracked lease history");
  const float next=float(starts_[row]+counts_[row]);
  if(slot!=next||pos[0]!=next||pos[1]!=next||pos[2]!=next)throw std::invalid_argument("Bulk append positions do not follow native token history");
  ++counts_[row];any_=true;
 }
 const std::array<size_t,4>& Counts() const{return counts_;}
 void Finish(bool lastChunkActive)const{if(!any_||!lastChunkActive)throw std::invalid_argument("Bulk request has no live trailing chunk");}
};
}
