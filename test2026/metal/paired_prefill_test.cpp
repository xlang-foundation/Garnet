// SPDX-License-Identifier: Apache-2.0
#include "metal_context.h"
#include "../../src/model/affine_int4_codec.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <vector>
using namespace Garnet;
void Need(bool b,const char* m){if(!b)throw std::runtime_error(m);}
template<class F> void Reject(F f){bool rejected=false;try{f();}catch(const std::exception&){rejected=true;}Need(rejected,"Expected rejection");}
int main(){try{
 size_t cases=0,words=0,fp64=0;
 {
 MetalContext gpu;const size_t m=512,n=14336,k=4096;auto layout=AffineInt4Layout::Create(n,k,AffineInt4CoefficientType::FP16);
 std::vector<float> data(m*k),poison(m*n,std::numeric_limits<float>::quiet_NaN());uint32_t seed=37;
 for(auto& x:data){seed=1664525u*seed+1013904223u;x=float(int((seed>>8)%65521)-32760)*0x1p-18f;}
 std::array<std::vector<uint32_t>,2> packed;std::array<std::vector<uint16_t>,2> scale,bias;
 std::array<MetalAffineInt4Weight,2> weights;std::array<MetalMatrix,2> out;
 auto input=gpu.Upload(data,m,k);
 for(size_t stream=0;stream<2;++stream){packed[stream].resize(layout.weightBytes/4);scale[stream].resize(layout.coefficientBytes/2);bias[stream].resize(scale[stream].size());
  for(size_t i=0;i<packed[stream].size();++i)packed[stream][i]=uint32_t(i)*(stream?2246822519u:2654435761u)+(stream?927u:131u);
  for(size_t i=0;i<scale[stream].size();++i){scale[stream][i]=Float32ToFloat16RNE(float(int((i+stream)%13)-6)*.00171f);bias[stream][i]=Float32ToFloat16RNE(float(int((i+3*stream)%17)-8)*.00313f);}
  weights[stream]=gpu.UploadAffineInt4(packed[stream].data(),layout.weightBytes,scale[stream].data(),layout.coefficientBytes,bias[stream].data(),layout.coefficientBytes,n,k,true);out[stream]=gpu.Upload(poison,m,n);
  Need(gpu.PrepareAffineInt4Profile(input,weights[stream],out[stream],512),"Prepare original profile");Need(gpu.PreparePackedAffineProfile(input,weights[stream],out[stream],512),"Prepare packed profile");
 }
 Need(gpu.PackedAffinePlanCount()==1,"Pair preparation duplicates shape plan");auto owners=MetalContext::ProfileOwnerStats();auto scratch=MetalContext::MatrixBufferInfo(gpu.PackedAffineScratch());Need(scratch.lengthBytes==14680064,"Private scratch changed");
 auto run=[&](const std::vector<size_t>& ids,bool dense,bool on){
  gpu.SetPairedPackedPrefillEnabled(on);for(auto& o:out)gpu.Update(o,poison);gpu.Begin();
  const auto* selected=dense?nullptr:&ids;bool paired=gpu.TryEncodePairedPackedPrefill(input,weights[0],weights[1],out[0],out[1],selected);
  Need(paired==bool(on&&ids.size()>=128),"Wrong paired route");
  if(!paired)for(size_t stream=0;stream<2;++stream){if(dense)gpu.EncodeAffineInt4MatMul(input,weights[stream],out[stream]);else gpu.EncodeAffineInt4IndexedMatMul(input,weights[stream],out[stream],ids);}
  gpu.SubmitAndWait();Need(gpu.LastPairedPackedPrefillDispatches()==size_t(paired),"Wrong pair count");Need(gpu.LastPackedAffineDispatches()==size_t(ids.size()>=128?2:0),"Logical packed count differs");Need(gpu.LastAffineProfileMisses()==0,"Profile miss");
  std::array<std::vector<float>,2> values;for(size_t stream=0;stream<2;++stream)values[stream]=gpu.Download(out[stream]);return values;
 };
 // Runtime policy starts disabled, even when extension compiled.
 std::vector<size_t> all(m);std::iota(all.begin(),all.end(),0);gpu.Begin();Need(!gpu.TryEncodePairedPackedPrefill(input,weights[0],weights[1],out[0],out[1]),"Policy default enabled");gpu.Abort();
 for(size_t count:{0u,4u,127u,128u,129u,255u,256u,257u,511u,512u})for(bool reverse:{false,true}){
  std::vector<size_t> ids(count);std::vector<bool> active(m);for(size_t i=0;i<count;++i){ids[i]=(127*i+5)%m;active[ids[i]]=true;}if(reverse)std::reverse(ids.begin(),ids.end());
  auto control=run(ids,false,false),candidate=run(ids,false,true);
  for(size_t stream=0;stream<2;++stream){Need(!std::memcmp(control[stream].data(),candidate[stream].data(),m*n*4),"Pair output bits differ");
   for(size_t i=0;i<m*n;++i){Need(std::isfinite(candidate[stream][i]),"Missing pair output");if(!active[i/n])Need(candidate[stream][i]==0,"Inactive pair output differs");}
   AffineInt4View view(layout,packed[stream].data(),layout.weightBytes,scale[stream].data(),layout.coefficientBytes,bias[stream].data(),layout.coefficientBytes);std::vector<float> decoded(k);
   if(count)for(size_t col:std::set<size_t>{0,1,15,16,31,32,n/2,n-1}){view.DecodeRow(col,decoded.data(),k);for(size_t index:std::set<size_t>{0,count/2,count-1}){
    double sum=0,absolute=0;for(size_t j=0;j<k;++j){double x=Float16ToFloat32(Float32ToFloat16RNE(data[ids[index]*k+j])),y=Float16ToFloat32(Float32ToFloat16RNE(decoded[j]));sum+=x*y;absolute+=std::abs(x*y);}
    Need(std::abs(double(candidate[stream][ids[index]*n+col])-sum)<=2e-6+4e-6*absolute,"Original FP64 bound");++fp64;
   }}words+=2*m*n;
  }++cases;
 }
 {auto a=run(all,true,false),b=run(all,true,true);for(size_t stream=0;stream<2;++stream)Need(!std::memcmp(a[stream].data(),b[stream].data(),m*n*4),"Dense pair bits differ");++cases;}
 auto unchanged=[&](){for(size_t stream=0;stream<2;++stream)gpu.Update(out[stream],std::vector<float>(m*n,17.f));};
 auto noWrite=[&](){for(size_t stream=0;stream<2;++stream){auto v=gpu.Download(out[stream]);Need(std::all_of(v.begin(),v.end(),[](float x){return x==17.f;}),"Rejected call wrote output");}};
 unchanged();gpu.SetPairedPackedPrefillEnabled(true);auto bad=all;bad[1]=bad[0];gpu.Begin();Reject([&]{gpu.TryEncodePairedPackedPrefill(input,weights[0],weights[1],out[0],out[1],&bad);});Reject([&]{gpu.SetPairedPackedPrefillEnabled(false);});gpu.Abort();noWrite();
 bad=all;bad[1]=m;gpu.Begin();Reject([&]{gpu.TryEncodePairedPackedPrefill(input,weights[0],weights[1],out[0],out[1],&bad);});gpu.Abort();noWrite();
 gpu.Begin();Need(!gpu.TryEncodePairedPackedPrefill(input,weights[0],weights[1],out[0],out[0]),"Output alias joined");gpu.Abort();noWrite();
 auto unready=gpu.Allocate(m,k);gpu.Begin();Reject([&]{gpu.TryEncodePairedPackedPrefill(unready,weights[0],weights[1],out[0],out[1]);});gpu.Abort();noWrite();
 auto wrong=gpu.Allocate(m,4096);gpu.Begin();Need(!gpu.TryEncodePairedPackedPrefill(input,weights[0],weights[1],out[0],wrong),"Unsupported output shape joined");gpu.Abort();noWrite();
 gpu.Begin();Need(gpu.TryEncodePairedPackedPrefill(input,weights[0],weights[1],out[0],out[1]),"Abort pair missing");gpu.Abort();for(auto& o:out)Reject([&]{gpu.Download(o);});
 auto recovered=run(all,true,true);Need(!recovered[0].empty(),"Recovery failed");
 gpu.Begin();Need(gpu.TryEncodePairedPackedPrefill(input,weights[0],weights[1],out[0],out[1]),"Flush pair missing");gpu.Flush();Reject([&]{gpu.SetPairedPackedPrefillEnabled(false);});gpu.SubmitAndWait();
 gpu.SetPackedAffineEnabled(false);gpu.Begin();Need(!gpu.TryEncodePairedPackedPrefill(input,weights[0],weights[1],out[0],out[1]),"Packed-off pair selected");gpu.Abort();gpu.SetPackedAffineEnabled(true);
 auto after=MetalContext::ProfileOwnerStats();Need(after.affineWeights==owners.affineWeights&&after.pipelineSets==owners.pipelineSets&&after.matrices==owners.matrices+2,"Unexpected request-time allocation");
 Need(gpu.PackedAffinePlanCount()==1&&MetalContext::MatrixBufferInfo(gpu.PackedAffineScratch()).identity==scratch.identity,"Pair plan/scratch mutated");
 }
 auto owners=MetalContext::ProfileOwnerStats();Need(!owners.matrices&&!owners.affineWeights&&!owners.pipelineSets,"Pair owners retained");Need(cases==21,"Missing native pair cases");std::cout<<"Native paired prefill PASS "<<cases<<" cases, "<<words<<" exact words, "<<fp64<<" FP64 checks; row guards, alias/unready rejection, policy lock, abort/recovery and bounded scratch\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
