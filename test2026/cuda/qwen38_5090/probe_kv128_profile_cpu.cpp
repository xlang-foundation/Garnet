// Pure header contract: no CUDA, TensorRT, or Garnet libraries.
#include "modelopt_fp8_kv128_profile.h"
#include <cstdio>
#include <stdexcept>
using namespace Garnet;
void require(bool value){if(!value)throw std::runtime_error("KV128 profile contract failed");}
int main(){try{
    int cases=0,selected=0,rejected=0;
    for(int layer=0;layer<65;++layer)for(const char* projection:{"q","k","v","o"}){
        auto name="model.language_model.layers."+std::to_string(layer)+".self_attn."+projection+"_proj.weight";
        const bool expected=layer<64&&layer%4==3&&(std::string(projection)=="k"||std::string(projection)=="v");
        require(ModelOptFP8KV128Selected("1",1,128,128,1024,5120,name)==expected);++cases;selected+=expected;rejected+=!expected;
    }
    const std::string valid="model.language_model.layers.3.self_attn.k_proj.weight";
    for(const char* policy:{static_cast<const char*>(nullptr),"","0"}){require(!ModelOptFP8KV128Selected(policy,1,128,128,1024,5120,valid));++cases;++rejected;}
    for(const char* policy:{"true","01"," 1","1 ","2","-1"}){try{ModelOptFP8KV128Selected(policy,1,128,128,1024,5120,valid);throw std::logic_error("Policy accepted");}catch(const std::invalid_argument&){++cases;++rejected;}}
    for(auto name:{"model.visual.layers.3.self_attn.k_proj.weight","mtp.model.language_model.layers.3.self_attn.k_proj.weight","model.language_model.layers.03.self_attn.k_proj.weight","model.language_model.layers.3.self_attn.k_proj.weight.extra"}){require(!ModelOptFP8KV128Selected("1",1,128,128,1024,5120,name));++cases;++rejected;}
    for(int part=0;part<5;++part){int shape[]={1,128,128,1024,5120};++shape[part];try{bool found=ModelOptFP8KV128Selected("1",shape[0],shape[1],shape[2],shape[3],shape[4],valid);require(part<3&&!found);}catch(const std::invalid_argument&){require(part>=3);}++cases;++rejected;}
    require(cases==278&&selected==32&&rejected==246);
    std::printf("{\"status\":\"KV128_profile_CPU_contract_passed\",\"cases\":278,\"selected\":32,\"rejected\":246,\"GPU_or_model_executed\":false}\n");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
