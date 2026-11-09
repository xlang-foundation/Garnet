// SPDX-License-Identifier: Apache-2.0
#include "metal_context.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
using namespace Garnet;
void Need(bool b,const char*m){if(!b)throw std::runtime_error(m);}
int main(){try{
 size_t cases=0,words=0;
 {MetalContext gpu;
 for(size_t batch:{1ul,2ul,4ul})for(size_t capacity:{256ul,4096ul})for(bool gated:{false,true}){
  std::vector<float> p(batch*32*capacity,NAN),v(batch*8*capacity*128,NAN),gate(batch*4096),slots(batch);
  std::vector<double>wanted(batch*4096),absolute(wanted.size());
  for(size_t row=0;row<batch;row++){
   size_t visible=std::max(size_t(1),capacity-row*capacity/4);slots[row]=float(visible-1);
   for(size_t kv=0;kv<8;kv++)for(size_t token=0;token<visible;token++)for(size_t col=0;col<128;col++)v[row*8*capacity*128+(kv*capacity+token)*128+col]=float(int((row*31+kv*19+token*13+col*7)%257)-128)*0x1p-7f;
   for(size_t h=0;h<32;h++){
    double sum=0;for(size_t token=0;token<visible;token++)sum+=1+(token*17+h*13+row*7)%251;
    for(size_t token=0;token<visible;token++)p[(row*32+h)*capacity+token]=row==3?0:float((1+(token*17+h*13+row*7)%251)/sum);
    for(size_t col=0;col<128;col++){
     size_t z=(row*32+h)*128+col;gate[z]=float(int(z%127)-63)*0x1p-4f;
     for(size_t token=0;token<visible;token++){double term=double(p[(row*32+h)*capacity+token])*double(v[row*8*capacity*128+((h/4)*capacity+token)*128+col]);wanted[z]+=term;absolute[z]+=std::abs(term);}
     if(gated){double denom=1+std::exp(-double(gate[z]));wanted[z]/=denom;absolute[z]/=denom;}
    }
   }
  }
  auto pg=gpu.Upload(p,batch,32*capacity),vg=gpu.Upload(v,batch,8*capacity*128),gg=gpu.Upload(gate,batch,4096),sg=gpu.Upload(slots,batch,1),out=gpu.Upload(std::vector<float>(batch*4096,NAN),batch,4096);
  gpu.Begin();gpu.EncodeAttentionContext(pg,vg,gg,sg,out,32,8,128,capacity,gated);bool rejected=false;try{gpu.Download(out);}catch(const std::logic_error&){rejected=true;}Need(rejected,"pending context readable");gpu.SubmitAndWait();auto actual=gpu.Download(out);
  for(size_t i=0;i<wanted.size();i++)Need(std::isfinite(actual[i])&&std::abs(double(actual[i])-wanted[i])<=2e-6+4e-6*absolute[i],"context FP64 mismatch");
  // A partially encoded operation must not leave its output readable after abort.
  gpu.Begin();gpu.EncodeAttentionContext(pg,vg,gg,sg,out,32,8,128,capacity,gated);gpu.Abort();rejected=false;try{gpu.Download(out);}catch(const std::logic_error&){rejected=true;}Need(rejected,"aborted context readable");
  gpu.Begin();gpu.EncodeAttentionContext(pg,vg,gg,sg,out,32,8,128,capacity,gated);gpu.SubmitAndWait();auto recovered=gpu.Download(out);Need(recovered==actual,"abort recovery changed output");
  cases++;words+=wanted.size();
 }}
 auto owners=MetalContext::ProfileOwnerStats();Need(!owners.matrices&&!owners.affineWeights&&!owners.bf16Weights&&!owners.pipelineSets,"owners remain");
 std::cout<<"grouped value context reference PASS cases="<<cases<<" fp32_words="<<words<<" ragged_poisoned_tail=true inactive_row=true gated_fallback=true partial_batch_fallback=true pending_abort_recovery=true owners=0\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
