// SPDX-License-Identifier: Apache-2.0
#include "metal_context.h"
#include "../../src/model/affine_int4_codec.h"
#include "../../src/model/float16_codec.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>
using namespace Garnet;namespace fs=std::filesystem;
namespace {
size_t checks=0,words=0;
void Need(bool value,const char* reason){++checks;if(!value)throw std::runtime_error(reason);}
template<class F>void Reject(F f){bool rejected=false;try{f();}catch(const std::exception&){rejected=true;}Need(rejected,"Invalid native operation accepted");}
void Same(const std::vector<float>& a,const std::vector<float>& b,const char* reason){Need(a.size()==b.size()&&!std::memcmp(a.data(),b.data(),a.size()*4),reason);words+=a.size();}
template<class T>std::vector<T> Read(fs::path p,size_t count){std::ifstream f(p,std::ios::binary|std::ios::ate);Need(bool(f)&&size_t(f.tellg())==count*sizeof(T),"Trained fixture size differs");f.seekg(0);std::vector<T> v(count);f.read((char*)v.data(),count*sizeof(T));Need(bool(f),"Trained fixture read failed");return v;}
void Run(MetalContext& gpu,const fs::path& assets,const fs::path& captures,int layer) {
 const size_t k=4096,n=14336;constexpr float guard=19.375f;
 auto layout=AffineInt4Layout::Create(n,k,AffineInt4CoefficientType::FP16);
 std::array<MetalAffineInt4Weight,2> weights;std::vector<float> input(4*k);
 if(layer>=0){auto all=Read<float>(captures/("layer-"+std::to_string(layer))/"norm-input.f32",512*k);size_t from[]={0,127,256,511};for(size_t row=0;row<4;++row)std::copy_n(all.data()+from[row]*k,k,input.data()+row*k);}
 else {uint32_t seed=37;for(auto& v:input){seed=1664525u*seed+1013904223u;v=float(int((seed>>8)%65521)-32760)*0x1p-18f;}}
 for(size_t role=0;role<2;++role){std::vector<uint32_t> packed;std::vector<uint16_t> scale,bias;
  if(layer>=0){auto p=assets/("layer-"+std::to_string(layer));std::string name=role?"up":"gate";packed=Read<uint32_t>(p/(name+"-weight.u32"),layout.weightBytes/4);scale=Read<uint16_t>(p/(name+"-scales.f16"),layout.coefficientBytes/2);bias=Read<uint16_t>(p/(name+"-biases.f16"),layout.coefficientBytes/2);}
  else {packed.resize(layout.weightBytes/4);scale.resize(layout.coefficientBytes/2);bias.resize(scale.size());for(size_t i=0;i<packed.size();++i)packed[i]=uint32_t(i)*(role?2246822519u:2654435761u)+(role?927u:131u);for(size_t i=0;i<scale.size();++i){scale[i]=Float32ToFloat16RNE(float(int((i+role)%13)-6)*.00171f);bias[i]=Float32ToFloat16RNE(float(int((i+role*3)%17)-8)*.00313f);if(i%97==0)scale[i]=uint16_t(i%2?0x8000:0x0001);}}
  weights[role]=gpu.UploadAffineInt4(packed.data(),layout.weightBytes,scale.data(),layout.coefficientBytes,bias.data(),layout.coefficientBytes,n,k,true);
 }
 std::vector<float> inputParent(32+4*k,guard),outParent(32+4*n,guard);std::copy(input.begin(),input.end(),inputParent.begin()+16);
 auto ip=gpu.Upload(inputParent,1,inputParent.size()),op=gpu.Upload(outParent,1,outParent.size());
 auto source=gpu.PrepareDiagnosticDenseView(ip,16,4,k),output=gpu.PrepareDiagnosticDenseView(op,16,4,n);
 auto gate=gpu.Allocate(4,n),up=gpu.Allocate(4,n);
 for(auto& w:weights)Need(gpu.PrepareAffineInt4Profile(source,w,gate,4),"Native affine preparation failed");
 auto run=[&](const std::vector<size_t>* ids,bool fused){
  gpu.Update(op,outParent);gpu.Begin();bool used=fused&&gpu.TryEncodeFusedMistralDecode(source,weights[0],weights[1],output,ids);
  if(!used){if(ids){gpu.EncodeAffineInt4IndexedDecode(source,weights[0],gate,*ids);gpu.EncodeAffineInt4IndexedDecode(source,weights[1],up,*ids);}else{gpu.EncodeAffineInt4MatMul(source,weights[0],gate);gpu.EncodeAffineInt4MatMul(source,weights[1],up);}gpu.EncodeSiluMul(gate,up,output);}
  gpu.SubmitAndWait();Need(gpu.LastFusedMistralDecodeBlocks()==size_t(used),"Wrong fused native count");Need(gpu.LastAffineProfileDispatches()==size_t(!ids||!ids->empty()?2:0)&&gpu.LastAffineProfileMisses()==0,"Logical projection count/profile changed");return gpu.Download(op);
 };
 // Original control executes before fusion profile preparation.
 auto full=run(nullptr,false);gpu.Begin();Need(!gpu.TryEncodeFusedMistralDecode(source,weights[0],weights[1],output),"Unprepared native profile executed");gpu.Abort();
 Need(gpu.PrepareFusedMistralDecode(source,weights[0],weights[1],output),"Native fused preparation failed");
 auto scratch=gpu.FusedMistralDecodeScratch();auto info=MetalContext::MatrixBufferInfo(scratch);Need(info.lengthBytes==65536,"Private FP32 input size differs");const auto plans=gpu.FusedMistralDecodeProfileCount();Need(plans>=1&&plans<=2,"Unexpected native profile multiplicity");
 auto before=MetalContext::ProfileOwnerStats();
 Same(run(nullptr,true),full,"Dense fused product/parent differs");Need(gpu.LastFusedMistralDecodeInputCopies()==0,"Separate parents copied input");
 for(size_t mask=0;mask<16;++mask){std::vector<size_t> ids;for(size_t row=0;row<4;++row)if(mask&(1ul<<row))ids.push_back(row);
  auto expected=run(&ids,false),actual=run(&ids,true);Same(actual,expected,"Selected-row product/parent differs");Need(gpu.LastFusedMistralDecodeBlocks()==size_t(mask==15),"Narrow mask selected four-row fusion");}
 std::vector<size_t> reverse{3,2,1,0};Same(run(&reverse,true),full,"Physical row permutation changed result");
 Same(gpu.Download(ip),inputParent,"Input parent changed");
 // The compiler may reuse one arena for normalized input and SiLU output.
 auto aliasData=outParent;std::copy(input.begin(),input.end(),aliasData.begin()+16);auto aliasOwner=gpu.Upload(aliasData,1,aliasData.size());
 auto aliasInput=gpu.PrepareDiagnosticDenseView(aliasOwner,16,4,k),aliasOutput=gpu.PrepareDiagnosticDenseView(aliasOwner,16,4,n);
 Need(gpu.PrepareFusedMistralDecode(aliasInput,weights[0],weights[1],aliasOutput),"Alias profile preparation failed");
 auto aliasRun=[&](){gpu.Update(aliasOwner,aliasData);gpu.Begin();Need(gpu.TryEncodeFusedMistralDecode(aliasInput,weights[0],weights[1],aliasOutput),"Physical alias declined");gpu.SubmitAndWait();Need(gpu.LastFusedMistralDecodeInputCopies()==1,"Overlapping input not copied first");Same(gpu.Download(aliasOwner),full,"Overlapping FP32 input/output changed product/guards");};aliasRun();
 gpu.Begin();Need(gpu.TryEncodeFusedMistralDecode(aliasInput,weights[0],weights[1],aliasOutput),"Abort candidate missing");gpu.Abort();Reject([&]{gpu.Download(aliasOwner);});aliasRun();
 gpu.Update(aliasOwner,aliasData);gpu.Begin();Need(gpu.TryEncodeFusedMistralDecode(aliasInput,weights[0],weights[1],aliasOutput),"Partial-submit candidate missing");gpu.Flush();gpu.Abort();Reject([&]{gpu.Download(aliasOwner);});aliasRun();
 gpu.Update(op,outParent);auto bad=reverse;bad[1]=bad[0];gpu.Begin();Reject([&]{gpu.TryEncodeFusedMistralDecode(source,weights[0],weights[1],output,&bad);});gpu.Abort();Same(gpu.Download(op),outParent,"Rejected duplicate wrote parent");
 bad={4};gpu.Begin();Reject([&]{gpu.TryEncodeFusedMistralDecode(source,weights[0],weights[1],output,&bad);});gpu.Abort();Same(gpu.Download(op),outParent,"Rejected range wrote parent");
 auto unready=gpu.Allocate(4,k);gpu.Begin();Reject([&]{gpu.TryEncodeFusedMistralDecode(unready,weights[0],weights[1],output);});gpu.Abort();Same(gpu.Download(op),outParent,"Rejected unready input wrote parent");
 auto wrong=gpu.Allocate(4,k);gpu.Begin();Need(!gpu.TryEncodeFusedMistralDecode(source,weights[0],weights[1],wrong),"Wrong width fused");Reject([&]{gpu.PrepareFusedMistralDecode(source,weights[0],weights[1],output);});gpu.Abort();
 gpu.Begin();Reject([&]{gpu.TryEncodeFusedMistralDecode(scratch,weights[0],weights[1],output);});gpu.Abort();
#ifdef GARNET_METAL_SILU_PROFILE32
 gpu.SetSiluProfile32Enabled(true);gpu.Begin();Need(!gpu.TryEncodeFusedMistralDecode(source,weights[0],weights[1],output),"Different SiLU policy fused");gpu.Abort();gpu.SetSiluProfile32Enabled(false);
#endif
 Need(gpu.FusedMistralDecodeProfileCount()==plans&&MetalContext::MatrixBufferInfo(gpu.FusedMistralDecodeScratch()).identity==info.identity,"Execution changed profile/scratch identity");
 auto after=MetalContext::ProfileOwnerStats();Need(after.affineWeights==before.affineWeights&&after.pipelineSets==before.pipelineSets&&after.matrices==before.matrices+5,"Unexpected execution-time owner growth");
 std::cout<<"fused_q4_decode native PASS layer="<<layer<<" masks=16 alias_copy=true abort_partial_recovery=true profiles="<<plans<<" words="<<words<<'\n';
}
}
int main(int argc,char** argv){try{
 if(argc==3){for(int layer:{0,15,31}){MetalContext gpu;Run(gpu,argv[1],argv[2],layer);}}
 else{Need(argc==1,"Expected optional ASSETS CAPTURES");MetalContext gpu;Run(gpu,{},{},-1);}
 auto owners=MetalContext::ProfileOwnerStats();Need(!owners.matrices&&!owners.affineWeights&&!owners.bf16Weights&&!owners.pipelineSets,"Native fused decode owners retained");
 std::cout<<"fused_q4_decode PASS checks="<<checks<<" fp32_parent_words="<<words<<" owners=0\n";return 0;
}catch(const std::exception& error){std::cerr<<"fused_q4_decode FAILED: "<<error.what()<<'\n';return 1;}}
