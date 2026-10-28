// Pure header contract; no CUDA, TensorRT, Garnet DLL, or model execution.
#include "modelopt_fp8_gdn_m16_profile.h"
#include <cstdio>
#include <stdexcept>
using namespace Garnet;
void require(bool v){if(!v)throw std::runtime_error("B4T4 GDN M16 profile contract failed");}
int main(){try{
    int cases=0,selected=0,rejected=0;
    for(int layer=0;layer<65;++layer)for(const char* projection:{"qkv","z","out","a","k","v"}){
        const std::string name="model.language_model.layers."+std::to_string(layer)+".linear_attn."+(std::string(projection)=="out"?"out_proj":"in_proj_"+std::string(projection))+".weight";
        const bool expected=layer<64&&layer%4!=3&&(std::string(projection)=="qkv"||std::string(projection)=="z");const int n=std::string(projection)=="qkv"?10240:6144;
        require(ModelOptFP8GDNM16Selected("1",4,4,16,n,5120,name)==expected);++cases;selected+=expected;rejected+=!expected;
    }
    const std::string valid="model.language_model.layers.0.linear_attn.in_proj_qkv.weight";
    for(const char* policy:{static_cast<const char*>(nullptr),"","0"}){require(!ModelOptFP8GDNM16Selected(policy,4,4,16,10240,5120,valid));++cases;++rejected;}
    for(const char* policy:{"true","01"," 1","1 ","2","-1"}){try{ModelOptFP8GDNM16Selected(policy,4,4,16,10240,5120,valid);throw std::logic_error("Policy accepted");}catch(const std::invalid_argument&){++cases;++rejected;}}
    for(const char* name:{"model.visual.layers.0.linear_attn.in_proj_qkv.weight","mtp.model.language_model.layers.0.linear_attn.in_proj_qkv.weight","model.language_model.layers.00.linear_attn.in_proj_qkv.weight","model.language_model.layers.0.linear_attn.in_proj_qkv.weight.extra","model.language_model.layers.-1.linear_attn.in_proj_qkv.weight","model.language_model.layers.3.self_attn.q_proj.weight"}){require(!ModelOptFP8GDNM16Selected("1",4,4,16,10240,5120,name));++cases;++rejected;}
    const int shapes[][3]={{1,4,4},{2,8,16},{8,2,16},{4,1,4},{4,32,128},{1,16,16},{4,4,15},{4,4,17}};
    for(const auto& s:shapes){require(!ModelOptFP8GDNM16Selected("1",s[0],s[1],s[2],10240,5120,valid));++cases;++rejected;}
    for(const char* projection:{"qkv","z"})for(int part=0;part<2;++part)for(int bad:{0,-1,1,5121,12288,65536}){
        const std::string name="model.language_model.layers.0.linear_attn.in_proj_"+std::string(projection)+".weight";int dims[]={std::string(projection)=="qkv"?10240:6144,5120};dims[part]=bad;
        try{ModelOptFP8GDNM16Selected("1",4,4,16,dims[0],dims[1],name);throw std::logic_error("Bad geometry accepted");}catch(const std::invalid_argument&){++cases;++rejected;}
    }
    require(cases==437&&selected==96&&rejected==341);
    std::printf("{\"status\":\"B4T4_GDN_M16_profile_CPU_contract_passed\",\"cases\":%d,\"selected\":%d,\"rejected\":%d,\"GPU_or_model_executed\":false}\n",cases,selected,rejected);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
