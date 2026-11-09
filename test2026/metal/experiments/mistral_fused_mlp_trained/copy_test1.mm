// Research observer copies must preserve bytes, guards and command ownership.
#include "metal_context.h"
#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
using namespace Garnet;
void Need(bool v,const char* text){if(!v)throw std::runtime_error(text);}
float Bits(uint32_t word){float x;std::memcpy(&x,&word,4);return x;}
int main(){try{
 size_t words=0,rejected=0;
 auto reject=[&](auto action){bool failed=false;try{action();}catch(const std::exception&){failed=true;}Need(failed,"Invalid observer operation accepted");++rejected;};
 for(size_t columns:{4096ul,14336ul}){
  MetalContext gpu,other;std::vector<float> original(514*columns),poison(514*columns,Bits(0x1937abef));
  const uint32_t corpus[]={0,0x80000000,0x3f800000,0xbf800000,1,0x80000001,0x7f800000,0xff800000,0x7fc01234,0x7f7fffff};
  for(size_t i=0;i<original.size();++i)original[i]=Bits(corpus[i%10]);
  auto parent=gpu.Upload(original,514,columns),destination=gpu.Upload(poison,514,columns);
  auto source=gpu.PrepareDiagnosticDenseView(parent,columns,512,columns),out=gpu.PrepareDiagnosticDenseView(destination,columns,512,columns);
  reject([&]{gpu.EncodeDiagnosticMlpCopy(source,out);});
  gpu.Begin();reject([&]{gpu.EncodeDiagnosticMlpCopy(source,source);});auto wrong=gpu.Allocate(511,columns);reject([&]{gpu.EncodeDiagnosticMlpCopy(source,wrong);});gpu.Abort();
  auto uninitialized=gpu.Allocate(512,columns);gpu.Begin();reject([&]{gpu.EncodeDiagnosticMlpCopy(uninitialized,out);});gpu.Abort();
  gpu.Begin();gpu.EncodeDiagnosticMlpCopy(source,out);reject([&]{gpu.Download(out);});
  other.Begin();reject([&]{other.EncodeDiagnosticMlpCopy(source,out);});reject([&]{other.EncodeDiagnosticMlpCopy(out,uninitialized);});other.Abort();
  gpu.SubmitAndWait();auto actual=gpu.Download(destination);auto expected=poison;
  std::memcpy(expected.data()+columns,original.data()+columns,512*columns*4);
  Need(actual.size()==expected.size()&&!std::memcmp(actual.data(),expected.data(),expected.size()*4),"Snapshot values, NaN payloads or guard rows changed");words+=expected.size();
  auto unchanged=gpu.Download(parent);Need(!std::memcmp(unchanged.data(),original.data(),original.size()*4),"Observer mutated producer");
  gpu.Begin();gpu.EncodeDiagnosticMlpCopy(source,out);gpu.Abort();reject([&]{gpu.Download(out);});reject([&]{gpu.Download(destination);});
  gpu.Update(destination,poison);gpu.Begin();gpu.EncodeDiagnosticMlpCopy(source,out);gpu.SubmitAndWait();actual=gpu.Download(destination);Need(!std::memcmp(actual.data(),expected.data(),expected.size()*4),"Observer failed Abort recovery");words+=expected.size();
 }
 auto owners=MetalContext::ProfileOwnerStats();Need(!owners.matrices&&!owners.affineWeights&&!owners.bf16Weights&&!owners.pipelineSets,"Observer unit retained native owners");
 std::cout<<"{\"passed\":true,\"complete_parent_word_comparisons\":"<<words<<",\"invalid_operations_rejected\":"<<rejected<<",\"all_native_owners_zero\":true}\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
