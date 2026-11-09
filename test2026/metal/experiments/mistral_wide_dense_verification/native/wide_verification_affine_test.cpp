// SPDX-License-Identifier: Apache-2.0
#include "metal_context.h"
#include "../../src/model/affine_int4_codec.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>
using namespace Garnet;
namespace {
size_t pairs=0,cases=0,rejections=0;
void Need(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void Same(const std::vector<float>& actual,const std::vector<float>& expected){Need(actual.size()==expected.size(),"Output size");Need(!std::memcmp(actual.data(),expected.data(),actual.size()*4),"Wide output differs from ordinary FP32 decode");pairs+=actual.size();++cases;}
void Case(MetalContext& gpu,size_t n,size_t k){
 const auto layout=AffineInt4Layout::Create(n,k,AffineInt4CoefficientType::FP16);std::vector<uint32_t> packed(layout.weightBytes/4);for(size_t i=0;i<packed.size();++i)packed[i]=uint32_t(i)*2654435761u+131u;
 std::vector<uint16_t> scales(layout.coefficientBytes/2),biases(scales.size());for(size_t i=0;i<scales.size();++i){scales[i]=uint16_t(0x2980+i%8);biases[i]=uint16_t(0xa900+i%7);}
 auto weight=gpu.UploadAffineInt4(packed.data(),layout.weightBytes,scales.data(),layout.coefficientBytes,biases.data(),layout.coefficientBytes,n,k,true);
 std::vector<float> input(16*k),reference(16*n);uint32_t seed=37;for(auto& v:input){seed=1664525u*seed+1013904223u;v=float(int((seed>>8)%65521)-32760)*.000004f;}
 for(size_t row=0;row<16;++row){auto x=gpu.Upload(std::vector<float>(input.begin()+row*k,input.begin()+(row+1)*k),1,k),y=gpu.Allocate(1,n);Need(gpu.PrepareAffineInt4Profile(x,weight,y,4),"Control profile");gpu.Begin();gpu.EncodeAffineInt4MatMul(x,weight,y);gpu.SubmitAndWait();Need(gpu.LastMPPAffineDispatches()==0&&gpu.LastAffineProfileMisses()==0,"Control route");auto v=gpu.Download(y);std::copy(v.begin(),v.end(),reference.begin()+row*n);}
 auto source=gpu.Upload(input,16,k),output=gpu.Allocate(16,n);Need(gpu.PrepareAffineInt4Profile(source,weight,output,16),"Wide profile");const auto prepared=gpu.AffineProfilePipelineCount();
 std::vector<size_t> all(16);std::iota(all.begin(),all.end(),0);std::vector<std::vector<size_t>> masks;for(size_t count=0;count<=16;++count)masks.emplace_back(all.begin(),all.begin()+count);masks.push_back({15,0,7,3,12,5,10,1,14});std::reverse(all.begin(),all.end());masks.push_back(all);masks.push_back({15});masks.push_back({0,15});masks.push_back({3,7,11,15});
 for(const auto& selected:masks){std::vector<float> expected(16*n,0);for(auto row:selected)std::copy_n(reference.data()+row*n,n,expected.data()+row*n);gpu.Update(output,std::vector<float>(16*n,-99));gpu.Begin();gpu.EncodeAffineInt4IndexedDecode(source,weight,output,selected);gpu.SubmitAndWait();Need(gpu.LastMPPAffineDispatches()==0&&gpu.LastAffineProfileMisses()==0&&gpu.LastIndexedDecodeAffineDispatches()==size_t(!selected.empty()),"Wide route used MPP or missed prepared profile");Need(gpu.AffineProfilePipelineCount()==prepared,"Request-time compilation");Same(gpu.Download(output),expected);}
 for(const auto& invalid:std::vector<std::vector<size_t>>{{0,0},{16},{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}}){gpu.Update(output,std::vector<float>(16*n,-99));gpu.Begin();bool rejected=false;try{gpu.EncodeAffineInt4IndexedDecode(source,weight,output,invalid);}catch(const std::invalid_argument&){rejected=true;}Need(rejected,"Invalid selected rows accepted");gpu.SubmitAndWait();Same(gpu.Download(output),std::vector<float>(16*n,-99));++rejections;}
 Same(gpu.Download(source),input);std::cout<<"wide affine PASS K="<<k<<" N="<<n<<" physical=16 selected=0..16 sparse_permuted=true no_mpp=true\n"<<std::flush;
}
}
int main(){try{
 {MetalContext gpu;for(auto shape:std::vector<std::pair<size_t,size_t>>{{17,64},{33,128},{1024,4096},{4096,4096},{14336,4096},{4096,14336},{32768,4096}})Case(gpu,shape.first,shape.second);}
 // A distinct unprepared shape must fail before even zeroing inactive output.
 {MetalContext gpu;const size_t k=192,n=17;auto layout=AffineInt4Layout::Create(n,k,AffineInt4CoefficientType::FP16);std::vector<uint32_t> packed(layout.weightBytes/4,0x76543210);std::vector<uint16_t> coeff(layout.coefficientBytes/2,0x2980);auto w=gpu.UploadAffineInt4(packed.data(),layout.weightBytes,coeff.data(),layout.coefficientBytes,coeff.data(),layout.coefficientBytes,n,k,true);auto x=gpu.Upload(std::vector<float>(16*k,.125f),16,k),y=gpu.Upload(std::vector<float>(16*n,-99),16,n);gpu.Begin();bool rejected=false;try{gpu.EncodeAffineInt4IndexedDecode(x,w,y,{0,1,2,3,15});}catch(const std::logic_error&){rejected=true;}Need(rejected,"Unprepared wide profile accepted");gpu.SubmitAndWait();Same(gpu.Download(y),std::vector<float>(16*n,-99));++rejections;}
 auto owners=MetalContext::ProfileOwnerStats();Need(!owners.matrices&&!owners.affineWeights&&!owners.bf16Weights&&!owners.pipelineSets,"Wide owners retained");std::cout<<"wide verification affine PASS cases="<<cases<<" fp32_word_pairs="<<pairs<<" rejections="<<rejections<<" owners=0\n";return 0;
}catch(const std::exception& e){std::cerr<<"wide verification affine FAILED: "<<e.what()<<'\n';return 1;}}
