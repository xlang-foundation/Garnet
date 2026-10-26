// Header-only CPU guard tests; no Garnet, TensorRT or CUDA dependency.
#include "qwen38_fa3_projection_witness_profile.h"
#include <cstdio>
#include <initializer_list>
#include <limits>
using namespace Garnet;
int main(){
    int count=0,rejected=0,selected=0;
    auto check=[&](const char* flag,std::string precision,std::vector<int> shape,std::string layer,double epsilon,bool external,bool f32,bool valid){
        ++count;
        try{RequireQwen38FA3ProjectionWitness(flag,precision,shape,layer,epsilon,external,f32);if(!valid)throw std::logic_error("Invalid FA3 profile admitted");++selected;}
        catch(const std::invalid_argument&){if(valid)throw;++rejected;}
    };
    try{
        for(const char* flag:{static_cast<const char*>(nullptr),"","0","true","01"," 1","1 ","-1","2","1"})for(int t:{32,128})
            check(flag,"modelopt_mixed",{1,t,5120},"model.language_model.layers.3",1.e-6,false,true,flag&&std::strcmp(flag,"1")==0);
        for(auto shape:{std::vector<int>{},{1,32},{1,32,5120,1},{0,32,5120},{2,32,5120},{4,32,5120},{8,32,5120},{1,0,5120},{1,16,5120},{1,31,5120},{1,129,5120},{1,32,5119},{1,32,-1}})
            check("1","modelopt_mixed",shape,"model.language_model.layers.3",1.e-6,false,true,false);
        for(const char* precision:{"bf16","ct_int4_g128",""})check("1",precision,{1,32,5120},"model.language_model.layers.3",1.e-6,false,true,false);
        for(const char* layer:{"","model.language_model.layers.2","model.language_model.layers.7"})check("1","modelopt_mixed",{1,32,5120},layer,1.e-6,false,true,false);
        for(double eps:{0.,-1.,1.e-5,std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity()})check("1","modelopt_mixed",{1,32,5120},"model.language_model.layers.3",eps,false,true,false);
        check("1","modelopt_mixed",{1,32,5120},"model.language_model.layers.3",1.e-6,true,true,false);
        check("1","modelopt_mixed",{1,32,5120},"model.language_model.layers.3",1.e-6,false,false,false);
        if(count!=47||rejected!=45||selected!=2)throw std::logic_error("FA3 profile accounting changed");
        std::printf("{\"status\":\"FA3_projection_profile_CPU_contract_passed\",\"cases\":%d,\"rejected\":%d,\"selected\":%d,\"GPU_or_model_executed\":false}\n",count,rejected,selected);return 0;
    }catch(const std::exception& error){std::fprintf(stderr,"%s\n",error.what());return 1;}
}
