// SPDX-License-Identifier: Apache-2.0
#include "metal_context.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
using namespace Garnet;
namespace {
size_t words=0,cases=0,rejections=0;
void Need(bool value,const char* why){if(!value)throw std::runtime_error(why);}
template<class F>void Reject(F fn){bool bad=false;try{fn();}catch(const std::exception&){bad=true;}Need(bad,"Invalid confidence access accepted");++rejections;}
void Check(MetalContext& gpu,size_t rows,size_t columns,int mode) {
 std::vector<float> input(rows*columns);
 for(size_t row=0;row<rows;++row) {
  for(size_t j=0;j<columns;++j)input[row*columns+j]=float(int((j*37+row*17)%8191)-4095)*0x1p-12f;
  if(mode==1)std::fill_n(input.data()+row*columns,columns,-0.f);
  if(mode==2){input[row*columns+columns-1]=2.f;if(columns>1)input[row*columns+std::min(size_t(127),columns-2)]=1.9f;}
  if(mode==3){input[row*columns]=std::numeric_limits<float>::max();if(columns>1)input[row*columns+columns-1]=-std::numeric_limits<float>::max();}
  if(mode>=4)input[row*columns+columns/2]=mode==4?std::numeric_limits<float>::quiet_NaN():mode==5?std::numeric_limits<float>::infinity():-std::numeric_limits<float>::infinity();
 }
 auto a=gpu.Upload(input,rows,columns),original=gpu.Allocate(rows,2),out=gpu.Allocate(rows,4);
 gpu.Update(original,std::vector<float>(rows*2,std::numeric_limits<float>::quiet_NaN()));
 gpu.Update(out,std::vector<float>(rows*4,std::numeric_limits<float>::quiet_NaN()));
 gpu.Begin();gpu.EncodeArgmax(a,original);gpu.EncodePredictionConfidence(a,out);gpu.SubmitAndWait();auto base=gpu.Download(original),actual=gpu.Download(out);
 for(size_t row=0;row<rows;++row) {
  Need(!std::memcmp(base.data()+2*row,actual.data()+4*row,2*sizeof(float)),"Existing argmax ID/error ABI changed");
  if(mode<4) {
   std::array<size_t,2> indices{0,columns>1?1u:0u};
   if(columns>1&&(input[row*columns+indices[1]]>input[row*columns+indices[0]]))std::swap(indices[0],indices[1]);
   for(size_t j=2;j<columns;++j) {
    float value=input[row*columns+j];
    if(value>input[row*columns+indices[0]]){indices[1]=indices[0];indices[0]=j;}
    else if(value>input[row*columns+indices[1]])indices[1]=j;
   }
   Need(actual[row*4]==float(indices[0]),"Lowest-index winner changed");
   Need(actual[row*4+2]==input[row*columns+indices[0]],"First confidence score differs from CPU");
   if(columns==1)Need(actual[row*4+3]==-std::numeric_limits<float>::infinity(),"Single-column second score must be negative infinity");
   else Need(actual[row*4+3]==input[row*columns+indices[1]],"Distinct-index second score differs from CPU");
  }else Need(actual[row*4+1]==1.f,"Nonfinite score not rejected");
 }
 words+=input.size();++cases;
}
}
int main(){try{
 {
  MetalContext gpu;
  for(size_t rows:{1,2,4})for(size_t columns:{1,2,3,31,32,33,127,128,129,151936})for(int mode=0;mode<7;++mode)Check(gpu,rows,columns,mode);
  Check(gpu,1,16777216,2);
  auto input=gpu.Upload(std::vector<float>(34,1.f),2,17),out=gpu.Allocate(2,4),bad=gpu.Allocate(2,3),empty=gpu.Allocate(2,17);
  gpu.Begin();Reject([&]{gpu.EncodePredictionConfidence(empty,out);});Reject([&]{gpu.EncodePredictionConfidence(input,bad);});gpu.Abort();
  auto alias=gpu.Upload(std::vector<float>(8,1.f),2,4);gpu.Begin();Reject([&]{gpu.EncodePredictionConfidence(alias,alias);});gpu.Abort();
  auto tooMany=gpu.Upload(std::vector<float>(8,1.f),8,1),manyOut=gpu.Allocate(8,4);gpu.Begin();Reject([&]{gpu.EncodePredictionConfidence(tooMany,manyOut);});gpu.Abort();
  Reject([&]{gpu.EncodePredictionConfidence(input,out);});
  gpu.Begin();gpu.EncodePredictionConfidence(input,out);gpu.Abort();gpu.Begin();gpu.EncodePredictionConfidence(input,out);gpu.SubmitAndWait();
  Need(gpu.Download(out)==std::vector<float>({0,0,1,1,0,0,1,1}),"Confidence did not recover after abort");
  MetalContext other;gpu.Begin();gpu.EncodePredictionConfidence(input,out);other.Begin();Reject([&]{other.EncodePredictionConfidence(input,out);});other.Abort();gpu.Abort();
 }
 auto owners=MetalContext::ProfileOwnerStats();Need(!owners.matrices&&!owners.affineWeights&&!owners.bf16Weights&&!owners.pipelineSets,"Confidence native owners remain");
 std::cout<<"Prediction confidence CPU top2/exact old ID-error, ties, nonfinite, bounds, ownership and abort PASS cases="<<cases<<" words="<<words<<" rejections="<<rejections<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
