// SPDX-License-Identifier: Apache-2.0
#include "prefix_precision_history.h"
#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>
using Garnet::PrefixPrecisionHistory;
void Need(bool v){if(!v)throw std::runtime_error("Native prefix producer-history reference failed");}
template<class F>void Reject(F f){bool failed=false;try{f();}catch(const std::exception&){failed=true;}Need(failed);}
void Check(const PrefixPrecisionHistory& h,const std::vector<bool>& phases){
 size_t prefix=0;while(prefix<phases.size()&&phases[prefix])++prefix;
 Need(h.TotalTokens()==phases.size()&&h.PrefillTokens()==prefix&&h.CaptureAllowed(phases.size())==(!phases.empty()&&prefix==phases.size()));
}
int main(){try{size_t checked=0;
 for(size_t count=0;count<=10;++count)for(size_t mask=0;mask<(size_t(1)<<count);++mask){
  PrefixPrecisionHistory h;std::vector<bool> phases;
  for(size_t i=0;i<count;++i){bool prefill=(mask>>i)&1;h.Append(phases.size(),1,prefill);phases.push_back(prefill);Check(h,phases);++checked;}
  for(size_t kept=0;kept<=count;++kept){auto copy=h;auto reference=phases;copy.Truncate(kept);reference.resize(kept);Check(copy,reference);copy.Append(kept,3,true);reference.insert(reference.end(),3,true);Check(copy,reference);++checked;}
  auto before=h;Reject([&]{h.Append(count+1,1,true);});Need(h.TotalTokens()==before.TotalTokens()&&h.PrefillTokens()==before.PrefillTokens());Reject([&]{h.Truncate(count+1);});Check(h,phases);
 }
 std::array<PrefixPrecisionHistory,4> rows;for(size_t row=0;row<4;++row)rows[row].Append(0,128+row,true);rows[0].Append(128,7,false);rows[0].Append(135,10,true);Need(rows[0].PrefillTokens()==128&&!rows[0].CaptureAllowed(145));for(size_t row=1;row<4;++row)Need(rows[row].CaptureAllowed(128+row));rows[0].Reset();Need(!rows[0].CaptureAllowed(0));rows[0].RestorePure(128);rows[0].Append(128,128,true);Need(rows[0].CaptureAllowed(256));rows[0].Append(256,8,false);rows[0].Truncate(128);rows[0].Append(128,2,true);Need(rows[0].CaptureAllowed(130));
 PrefixPrecisionHistory bound;bound.Append(0,4096,true);Need(bound.CaptureAllowed(4096));Reject([&]{bound.Append(4096,1,false);});Reject([&]{bound.RestorePure(4097);});Need(bound.CaptureAllowed(4096));bound.Truncate(0);Need(!bound.CaptureAllowed(0));
 std::cout<<"Native prefix producer-history PASS, reference checkpoints="<<checked<<"\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
