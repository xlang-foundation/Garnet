// SPDX-License-Identifier: Apache-2.0
// CPU header controls only; execute on the remote build host after GPU work closes.
#include "modelopt_fp8_m128_profile.h"
#include "modelopt_fp8_z128_reconstructed_profile.h"
#include "modelopt_fp8_kv128_profile.h"
#include "modelopt_fp8_q128_profile.h"
#include "modelopt_fp8_o128_profile.h"
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
int checks=0,rejections=0;
void Check(bool value){++checks;if(!value)throw std::runtime_error("B4 profile predicate failed");}
template<class F>void Reject(F call){bool rejected=false;try{call();}catch(const std::invalid_argument&){rejected=true;}Check(rejected);++rejections;}
using Select=std::function<bool(const char*,int,int,int64_t,int,int,const char*)>;
struct Family{int n,k;Select select;};
}
int main(){
 try{
  using namespace Garnet;
  const std::string p="model.language_model.layers.3.self_attn.";
  std::vector<Family> families={
   {5120,6144,[](const char* f,int b,int t,int64_t m,int n,int k,const char* b4){return ModelOptFP8M128GDNSelected(f,b,t,m,n,k,true,b4);}},
   {6144,5120,[](const char* f,int b,int t,int64_t m,int n,int k,const char* b4){return ModelOptFP8Z128GDNZSelected(f,b,t,m,n,k,true,b4);}},
   {6144,5120,[](const char* f,int b,int t,int64_t m,int n,int k,const char* b4){return ModelOptFP8Z128ReconstructedGDNZSelected(f,f,b,t,m,n,k,true,b4);}},
   {1024,5120,[&](const char* f,int b,int t,int64_t m,int n,int k,const char* b4){return ModelOptFP8KV128Selected(f,b,t,m,n,k,p+"k_proj.weight",b4);}},
   {12288,5120,[&](const char* f,int b,int t,int64_t m,int n,int k,const char* b4){return ModelOptFP8Q128Selected(f,b,t,m,n,k,p+"q_proj.weight",b4);}},
   {5120,6144,[&](const char* f,int b,int t,int64_t m,int n,int k,const char* b4){return ModelOptFP8O128Selected(f,b,t,m,n,k,p+"o_proj.weight",b4);}}
  };
  // Original function arity remains valid and does not opt into B4.
  Check(ModelOptFP8M128GDNSelected("1",1,128,128,5120,6144,true));
  Check(ModelOptFP8Z128GDNZSelected("1",1,128,128,6144,5120,true));
  Check(ModelOptFP8Z128ReconstructedGDNZSelected("1","1",1,128,128,6144,5120,true));
  Check(ModelOptFP8KV128Selected("1",1,128,128,1024,5120,p+"k_proj.weight"));
  Check(ModelOptFP8Q128Selected("1",1,128,128,12288,5120,p+"q_proj.weight"));
  Check(ModelOptFP8O128Selected("1",1,128,128,5120,6144,p+"o_proj.weight"));
  for(const auto& f:families){
   for(const char* alias:{static_cast<const char*>(nullptr),"","0"}){
    Check(!f.select("1",4,32,128,f.n,f.k,alias));
    Check(f.select("1",1,128,128,f.n,f.k,alias));
    Check(!f.select(alias,4,32,128,f.n,f.k,"1"));
   }
   Check(f.select("1",4,32,128,f.n,f.k,"1"));
   Check(f.select("1",1,128,128,f.n,f.k,"1"));
   Check(!f.select("1",2,64,128,f.n,f.k,"1"));
   Check(!f.select("1",8,16,128,f.n,f.k,"1"));
   Check(!f.select("1",4,31,124,f.n,f.k,"1"));
   Check(!f.select("1",4,4,16,f.n,f.k,"1"));
   Reject([&]{f.select("1",4,32,127,f.n,f.k,"1");});
   Reject([&]{f.select("1",4,32,128,f.n+16,f.k,"1");});
   Reject([&]{f.select("1",4,32,128,f.n,f.k+16,"1");});
   for(const char* bad:{"2","true"," 1","01","-1","1 "})Reject([&]{f.select("1",4,32,128,f.n,f.k,bad);});
   Reject([&]{f.select("2",4,32,128,f.n,f.k,"1");});
  }
  Reject([]{ModelOptFP8Z128ReconstructedGDNZSelected("1","0",4,32,128,6144,5120,true,"1");});
  Check(!ModelOptFP8M128GDNSelected("1",4,32,128,5120,6144,false,"1"));
  Check(!ModelOptFP8Z128GDNZSelected("1",4,32,128,6144,5120,false,"1"));
  for(int layer=3;layer<64;layer+=4){
   const auto prefix="model.language_model.layers."+std::to_string(layer)+".self_attn.";
   Check(ModelOptFP8KV128Selected("1",4,32,128,1024,5120,prefix+"k_proj.weight","1"));
   Check(ModelOptFP8KV128Selected("1",4,32,128,1024,5120,prefix+"v_proj.weight","1"));
   Check(ModelOptFP8Q128Selected("1",4,32,128,12288,5120,prefix+"q_proj.weight","1"));
   Check(ModelOptFP8O128Selected("1",4,32,128,5120,6144,prefix+"o_proj.weight","1"));
  }
  for(const std::string name:{"model.visual.blocks.3.attn.q_proj.weight","model.mtp.layers.3.self_attn.q_proj.weight","model.language_model.layers.0.self_attn.q_proj.weight","model.language_model.layers.3.self_attn.q_proj.weight.trailing"}){
   Check(!ModelOptFP8KV128Selected("1",4,32,128,1024,5120,name,"1"));
   Check(!ModelOptFP8Q128Selected("1",4,32,128,12288,5120,name,"1"));
   Check(!ModelOptFP8O128Selected("1",4,32,128,5120,6144,name,"1"));
  }
  std::cout<<"{\"status\":\"B4_T32_FP8_header_controls_passed\",\"checks\":"<<checks<<",\"rejections\":"<<rejections<<",\"actual_GEMM_or_GPU_executed\":false,\"SDK_model_MTP_quality_performance_pending\":true}\n";
  return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
