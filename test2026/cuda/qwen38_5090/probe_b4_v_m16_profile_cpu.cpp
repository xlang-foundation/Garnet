// Pure header selector contract. No GPU, TensorRT, or Garnet DLL.
#include "modelopt_fp8_v_m16_profile.h"
#include <cstdio>
#include <stdexcept>
using namespace Garnet;
void require(bool v){if(!v)throw std::runtime_error("B4T4 V M16 profile contract failed");}
int main(){try{
    int cases=0,selected=0,rejected=0;
    for(int layer=0;layer<65;++layer)for(const char* scope:{"self_attn","linear_attn"})for(const char* projection:{"q_proj","k_proj","v_proj","o_proj","in_proj_qkv","in_proj_z","in_proj_a","in_proj_b","out_proj"}){
        const std::string name="model.language_model.layers."+std::to_string(layer)+"."+scope+"."+projection+".weight";
        const bool expected=layer<64&&layer%4==3&&std::string(scope)=="self_attn"&&std::string(projection)=="v_proj";
        require(ModelOptFP8VM16Selected("1",4,4,16,1024,5120,name)==expected);++cases;selected+=expected;rejected+=!expected;
    }
    const std::string valid="model.language_model.layers.3.self_attn.v_proj.weight";
    for(const char* policy:{static_cast<const char*>(nullptr),"","0"}){require(!ModelOptFP8VM16Selected(policy,4,4,16,1024,5120,valid));++cases;++rejected;}
    for(const char* policy:{"true","01"," 1","1 ","2","-1"}){try{ModelOptFP8VM16Selected(policy,4,4,16,1024,5120,valid);throw std::logic_error("Policy accepted");}catch(const std::invalid_argument&){++cases;++rejected;}}
    for(const char* name:{"model.visual.layers.3.self_attn.v_proj.weight","mtp.model.language_model.layers.3.self_attn.v_proj.weight","model.language_model.layers.03.self_attn.v_proj.weight","model.language_model.layers.3.self_attn.v_proj.weight.extra","model.language_model.layers.-1.self_attn.v_proj.weight","model.language_model.layers.3.self_attn.v_proj"}){require(!ModelOptFP8VM16Selected("1",4,4,16,1024,5120,name));++cases;++rejected;}
    const int shapes[][3]={{1,4,4},{2,8,16},{8,2,16},{4,1,4},{4,32,128},{1,16,16},{4,4,15},{4,4,17}};
    for(const auto& s:shapes){require(!ModelOptFP8VM16Selected("1",s[0],s[1],s[2],1024,5120,valid));++cases;++rejected;}
    for(int part=0;part<2;++part)for(int bad:{0,-1,1,5121,12288,65536,6144,10240}){
        int dims[]={1024,5120};dims[part]=bad;
        try{ModelOptFP8VM16Selected("1",4,4,16,dims[0],dims[1],valid);throw std::logic_error("Bad geometry accepted");}catch(const std::invalid_argument&){++cases;++rejected;}
    }
    require(cases==1209&&selected==16&&rejected==1193);
    std::printf("{\"status\":\"B4T4_V_M16_profile_CPU_contract_passed\",\"cases\":%d,\"selected\":%d,\"rejected\":%d,\"GPU_or_model_executed\":false}\n",cases,selected,rejected);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
