// SPDX-License-Identifier: Apache-2.0
#include "metal_context.h"
#include "../../src/backends/metal/bulk_silu_half_policy.h"
#include "../../src/model/affine_int4_codec.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>
using namespace Garnet;using JSON=nlohmann::json;
void Need(bool v,const char* m){if(!v)throw std::runtime_error(m);}
template<class F>void Reject(F f){bool yes=false;try{f();}catch(const std::exception&){yes=true;}Need(yes,"Missing SiLU-half guard");}
template<class T>void Save(const std::filesystem::path& p,const std::vector<T>& v){std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(T));Need(bool(f),"Cannot save raw half fixture");}
std::vector<uint16_t> Halves(MetalContext& gpu,const MetalMatrix& carrier,size_t count){auto values=gpu.Download(carrier);Need(values.size()*2==count,"Half carrier extent differs");std::vector<uint16_t> out(count);std::memcpy(out.data(),values.data(),count*2);return out;}
int main(int argc,char** argv){try{Need(argc==1||argc==2,"Optional fresh output directory");std::filesystem::path root;if(argc==2){root=argv[1];Need(std::filesystem::create_directory(root),"Fresh output directory required");}
 Need(BulkSiluHalfSupported(8192,14336,14344,14352,14368),"Valid padded uint32 profile rejected");for(auto p:std::vector<std::array<uint64_t,5>>{{0,14336,14336,14336,14336},{8193,14336,14336,14336,14336},{1,14335,14336,14336,14336},{8192,14336,0xffffffffull,14336,14336},{4,14336,14336,0,14336},{4,14336,14336,14336,0x100000000ull}})Need(!BulkSiluHalfSupported(p[0],p[1],p[2],p[3],p[4]),"Invalid half profile admitted");
 JSON report={{"scope","Actual native FP32 SiLU plus existing bulk GPU cast versus same-context fused SiLU-to-half, exact complete half words. No model/performance admission."},{"cases",JSON::array()}};size_t cases=0,total=0,finiteSamples=0;
 for(bool edges:{false,true})for(auto shape:std::vector<std::array<size_t,2>>{{1,14336},{4,14336},{128,14336},{129,14336},{257,14336},{511,14336},{512,14336},{8192,14336},{512,4096},{8192,4096}}){
  size_t m=shape[0],k=shape[1],count=m*k;std::vector<float> g(count),u(count);uint32_t seed=37;for(size_t i=0;i<count;++i){seed=1664525u*seed+1013904223u;g[i]=float(int((seed>>8)%65521)-32760)*0x1p-10f;seed=1664525u*seed+1013904223u;u[i]=float(int((seed>>8)%65521)-32760)*0x1p-12f;}
  uint32_t special[]{0,0x80000000u,1,0x80000001u,0x00800000u,0x80800000u,0x3f800000u,0xbf800000u,0x42b00000u,0xc2b00000u,0x42c80000u,0xc2c80000u,0x477fe000u,0x47800000u,0x7f800000u,0xff800000u,0x7fc01234u,0x7fa01234u};if(edges)for(size_t i=0;i<count;i+=257){auto bits=special[(i/257)%18];std::memcpy(g.data()+i,&bits,4);if((i/257)%3==0){bits=special[(i/257+7)%18];std::memcpy(u.data()+i,&bits,4);}}
  MetalContext gpu;auto a=gpu.Upload(g,m,k),b=gpu.Upload(u,m,k),f32=gpu.Allocate(m,k);uint32_t poison=0x7eab7eab;float pf;std::memcpy(&pf,&poison,4);std::vector<float> poisoned(count/2,pf);auto reference=gpu.Upload(poisoned,m,k/2),candidate=gpu.Upload(poisoned,m,k/2);
  auto layout=AffineInt4Layout::Create(4096,k,AffineInt4CoefficientType::FP16);std::vector<uint32_t> words(layout.weightBytes/4,0);std::vector<uint16_t> scales(layout.coefficientBytes/2,0x3c00u),biases(scales.size(),0);auto weight=gpu.UploadAffineInt4(words.data(),layout.weightBytes,scales.data(),layout.coefficientBytes,biases.data(),layout.coefficientBytes,4096,k,true);auto out=gpu.Allocate(m,4096);Need(gpu.PrepareDiagnosticBulkAffine(f32,weight,out),"Cannot prepare actual native cast reference");auto owners=MetalContext::ProfileOwnerStats();
  gpu.Begin();gpu.EncodeSiluMul(a,b,f32);gpu.EncodeDiagnosticHalfCastReference(f32,reference);gpu.SubmitAndWait();Need(gpu.LastDiagnosticSiluHalfDispatches()==0,"Reference selected fusion");auto expected=Halves(gpu,reference,count);auto original=gpu.Download(f32);
  gpu.Begin();gpu.EncodeDiagnosticSiluHalf(a,b,candidate);gpu.SubmitAndWait();Need(gpu.LastDiagnosticSiluHalfDispatches()==1,"Fusion not selected");auto actual=Halves(gpu,candidate,count);
  if(!root.empty()){auto path=root/("case-"+std::to_string(cases));Need(std::filesystem::create_directory(path),"Fresh case directory required");Save(path/"gate.f32",g);Save(path/"up.f32",u);Save(path/"original-silu.f32",original);Save(path/"reference.f16",expected);Save(path/"candidate.f16",actual);}
Need(actual==expected,"Fused half words differ from actual native SiLU plus GPU cast");
  for(size_t j=0;j<128;++j){size_t i=(j*104729+19)%count;if(!std::isfinite(g[i])||!std::isfinite(u[i]))continue;double value=(double(g[i])/(1.+std::exp(-double(g[i]))))*double(u[i]);if(!std::isfinite(original[i]))continue;Need(std::abs(double(original[i])-value)<=2e-6+4e-6*std::abs(value),"Original finite FP32 oracle bound");++finiteSamples;}
  auto after=MetalContext::ProfileOwnerStats();Need(after.matrices==owners.matrices&&after.pipelineSets==owners.pipelineSets&&after.affineWeights==owners.affineWeights,"SiLU request-time owners changed");
  report["cases"].push_back(JSON{{"index",cases},{"edge",edges},{"rows",m},{"columns",k},{"checked_half_words",count},{"equal",true}});total+=count;++cases;std::cout<<"bulk_silu_half "<<cases<<" PASS\n"<<std::flush;
 }
 {
  MetalContext gpu;const size_t rows=4,k=4096;auto a=gpu.Upload(std::vector<float>(rows*k,.25f),rows,k),b=gpu.Upload(std::vector<float>(rows*k,.5f),rows,k),unready=gpu.Allocate(rows,k);auto parent=gpu.Upload(std::vector<float>(rows*k,17.f),rows*2,k/2),view=gpu.PrepareDiagnosticDenseView(parent,0,rows,k/2);auto before=gpu.Download(parent);
  gpu.Begin();Reject([&]{gpu.EncodeDiagnosticSiluHalf(unready,b,view);});gpu.Abort();Need(gpu.Download(parent)==before,"Unready input wrote carrier");
  auto alias=gpu.PrepareDiagnosticDenseView(a,0,rows,k/2);auto original=gpu.Download(a);gpu.Begin();Reject([&]{gpu.EncodeDiagnosticSiluHalf(a,b,alias);});gpu.Abort();Need(gpu.Download(a)==original,"Alias guard wrote input");
  auto wrong=gpu.Upload(std::vector<float>((rows-1)*k,1.f),rows-1,k);gpu.Begin();Reject([&]{gpu.EncodeDiagnosticSiluHalf(a,wrong,view);});gpu.Abort();Need(gpu.Download(parent)==before,"Shape guard wrote output");
  gpu.Begin();gpu.EncodeDiagnosticSiluHalf(a,b,view);gpu.Abort();Need(gpu.LastDiagnosticSiluHalfDispatches()==0,"Abort retained half observer");Reject([&]{gpu.Download(parent);});Reject([&]{gpu.Update(view,std::vector<float>(rows*k/2,0.f));});gpu.Update(parent,before);gpu.Begin();gpu.EncodeDiagnosticSiluHalf(a,b,view);gpu.SubmitAndWait();Need(gpu.LastDiagnosticSiluHalfDispatches()==1,"Half recovery failed");auto restored=gpu.Download(parent);Need(std::equal(restored.begin()+rows*k/2,restored.end(),before.begin()+rows*k/2),"Partial half writer changed unselected parent tail");
 }
 auto owners=MetalContext::ProfileOwnerStats();Need(!owners.matrices&&!owners.affineWeights&&!owners.bf16Weights&&!owners.pipelineSets,"Half test retained native owners");Need(cases==20,"Missing half cases");report["passed"]=true;report["checked_half_words"]=total;report["finite_FP64_samples"]=finiteSamples;if(!root.empty()){std::ofstream f(root/"results.json");f<<report.dump(2)<<'\n';Need(bool(f),"Cannot save report");}std::cout<<"Actual native bulk SiLU-half PASS cases="<<cases<<" exact_half_words="<<total<<" finite_FP64_samples="<<finiteSamples<<"\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
