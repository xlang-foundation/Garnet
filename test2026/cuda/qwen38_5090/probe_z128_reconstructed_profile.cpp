// Header-only CPU admission tests. No TensorRT, CUDA, SDK or Garnet linkage.
#include "modelopt_fp8_z128_reconstructed_profile.h"
#include <cstdio>
#include <initializer_list>
#include <stdexcept>
using Garnet::ModelOptFP8Z128ReconstructedGDNZSelected;
struct Profile {int batch,tokens,rows,n,k;bool z;};
int main(){
    try{
        const Profile profiles[]={{1,128,128,6144,5120,true},{1,32,32,6144,5120,true},{4,32,128,6144,5120,true},{8,16,128,6144,5120,true},{1,128,127,6144,5120,true},{1,128,128,6144,5120,false}};
        int count=0,rejections=0,selected=0;
        auto call=[&](const char* r,const char* c,Profile p){++count;return ModelOptFP8Z128ReconstructedGDNZSelected(r,c,p.batch,p.tokens,p.rows,p.n,p.k,p.z);};
        auto require=[](bool v){if(!v)throw std::logic_error("Profile contract failed");};
        for(auto* r:{static_cast<const char*>(nullptr),"","0"})for(auto* c:{static_cast<const char*>(nullptr),"","0","1","bad"})for(auto p:profiles)require(!call(r,c,p));
        for(auto* c:{static_cast<const char*>(nullptr),"","0","bad"})for(auto p:profiles){try{call("1",c,p);throw std::logic_error("Missing canonical policy accepted");}catch(const std::invalid_argument&){++rejections;}}
        for(auto p:profiles){bool result=call("1","1",p);selected+=result;require(result==(p.batch==1&&p.tokens==128&&p.rows==128&&p.z));}
        for(auto p:{Profile{1,128,128,6143,5120,true},Profile{1,128,128,6144,5119,true}}){try{call("1","1",p);throw std::logic_error("Wrong geometry accepted");}catch(const std::invalid_argument&){++rejections;}}
        for(auto* r:{"true","01"," 1","1 ","2","-1"})for(auto p:profiles){try{call(r,"1",p);throw std::logic_error("Invalid flag accepted");}catch(const std::invalid_argument&){++rejections;}}
        require(count==158&&rejections==62&&selected==1);
        std::printf("{\"status\":\"reconstructed_Z128_profile_CPU_contract_passed\",\"cases\":%d,\"rejections\":%d,\"selected\":%d,\"GPU_or_model_executed\":false}\n",count,rejections,selected);return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
