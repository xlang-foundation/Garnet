// Pure header selector contract; no CUDA, TensorRT or Garnet runtime.
#include "modelopt_fp8_qk_m16_profile.h"
#include <cstdio>
#include <stdexcept>
using namespace Garnet;
void require(bool v){if(!v)throw std::runtime_error("B4T4 QK M16 selector contract failed");}
int main(){try{
    int cases=0,selected=0,rejected=0;
    for(int layer=0;layer<65;++layer)for(const char* scope:{"self_attn","linear_attn"})for(const char* projection:{"q_proj","k_proj","v_proj","o_proj","in_proj_qkv","in_proj_z","in_proj_a","in_proj_b","out_proj"}){
        const std::string name="model.language_model.layers."+std::to_string(layer)+"."+scope+"."+projection+".weight";
        const bool expected=layer<64&&layer%4==3&&std::string(scope)=="self_attn"&&(std::string(projection)=="q_proj"||std::string(projection)=="k_proj");
        require(ModelOptFP8QKM16Selected("1",4,4,16,std::string(projection)=="q_proj"?12288:1024,5120,name)==expected);++cases;selected+=expected;rejected+=!expected;
    }
    for(const char* projection:{"q_proj","k_proj"}){
        const int n=std::string(projection)=="q_proj"?12288:1024;
        const std::string valid="model.language_model.layers.3.self_attn."+std::string(projection)+".weight";
        for(const char* policy:{static_cast<const char*>(nullptr),"","0"}){require(!ModelOptFP8QKM16Selected(policy,4,4,16,n,5120,valid));++cases;++rejected;}
        for(const char* policy:{"true","01"," 1","1 ","2","-1"}){try{ModelOptFP8QKM16Selected(policy,4,4,16,n,5120,valid);throw std::logic_error("Policy accepted");}catch(const std::invalid_argument&){++cases;++rejected;}}
        for(const std::string name:{"model.visual.layers.3.self_attn."+std::string(projection)+".weight","mtp."+valid,"model.language_model.layers.03.self_attn."+std::string(projection)+".weight",valid+".extra","model.language_model.layers.-1.self_attn."+std::string(projection)+".weight",valid.substr(0,valid.size()-7)}){require(!ModelOptFP8QKM16Selected("1",4,4,16,n,5120,name));++cases;++rejected;}
        const int shapes[][3]={{1,4,4},{2,8,16},{8,2,16},{4,1,4},{4,32,128},{1,16,16},{4,4,15},{4,4,17}};
        for(const auto& s:shapes){require(!ModelOptFP8QKM16Selected("1",s[0],s[1],s[2],n,5120,valid));++cases;++rejected;}
        for(int part=0;part<2;++part)for(int bad:{0,-1,1,5121,65536,6144,10240,n==1024?12288:1024}){
            int dims[]={n,5120};dims[part]=bad;
            try{ModelOptFP8QKM16Selected("1",4,4,16,dims[0],dims[1],valid);throw std::logic_error("Bad geometry accepted");}catch(const std::invalid_argument&){++cases;++rejected;}
        }
    }
    require(cases==1248&&selected==32&&rejected==1216);
    std::printf("{\"status\":\"B4T4_QK_M16_profile_CPU_contract_passed\",\"cases\":%d,\"selected\":%d,\"rejected\":%d,\"GPU_or_model_executed\":false}\n",cases,selected,rejected);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
