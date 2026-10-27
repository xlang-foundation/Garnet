// Actual plugin initialize/clone/wire methods with CPU factories only, no GPU runtime calls.
#include "modelopt_fp8_plugin.h"
#include "modelopt_nvfp4_plugin.h"
#include "modelopt_fp8_m128_plugin.h"
#include "modelopt_fp8_z128_plugin.h"
#include "modelopt_fp8_z128_reconstructed_plugin.h"
#include "modelopt_plugin_init_trace.h"
#include "modelopt_fp8_kv128_plugin.h"
#include "modelopt_fp8_q128_plugin.h"
#include "modelopt_fp8_o128_plugin.h"
#include <NvInfer.h>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
namespace Garnet {void ModelOptInitCPUSetMode(int);int ModelOptInitCPUFactoryCalls();int ModelOptInitCPUOperatorCalls();int ModelOptInitCPUNewZCalls();int ModelOptInitCPUOldZCalls();int ModelOptInitCPUKVCalls();int ModelOptInitCPUQCalls();}
using namespace Garnet;
using Plugin=nvinfer1::IPluginV2DynamicExt;
struct Destroy {void operator()(Plugin* value)const noexcept{if(value)value->destroy();}};
using OwnedPlugin=std::unique_ptr<Plugin,Destroy>;
void require(bool value,const char* error){if(!value)throw std::runtime_error(error);}
OwnedPlugin plugin(int route){
    if(route==7)return OwnedPlugin(new ModelOptFP8O128Plugin(5120,6144,.5f,.25f));
    if(route==6)return OwnedPlugin(new ModelOptFP8Q128Plugin(12288,5120,.5f,.25f));
    if(route==5)return OwnedPlugin(new ModelOptFP8KV128Plugin(1024,5120,.5f,.25f));
    if(route==0)return OwnedPlugin(new ModelOptFP8Plugin(5120,6144,.5f,.25f,true));
    if(route==1)return OwnedPlugin(new ModelOptNVFP4Plugin(5120,17408,.5f,.25f,true,true));
    if(route==2)return OwnedPlugin(new ModelOptFP8M128Plugin(5120,6144,.5f,.25f));
    if(route==4)return OwnedPlugin(new ModelOptFP8Z128ReconstructedPlugin(6144,5120,.5f,.25f));
    return OwnedPlugin(new ModelOptFP8Z128Plugin(6144,5120,.5f,.25f));
}
int main(){
    try{
        int calls=0,factories=0,successes=0,failures=0,clones=0;
        for(int route=0;route<8;++route)for(int mode=0;mode<4;++mode){
            ModelOptInitCPUSetMode(mode);auto value=plugin(route);value->setPluginNamespace("init_trace_cpu");std::vector<unsigned char> before(value->getSerializationSize()),after(before.size());value->serialize(before.data());
            const int expected=mode==1?0:1;
            for(int repeat=0;repeat<2;++repeat){const int result=value->initialize();require(result==expected,"Initialize result changed");++calls;if(result)++failures;else ++successes;}
            require(ModelOptInitCPUFactoryCalls()==(mode==1?1:2),"Factory call count changed");
            if(mode==1){auto* copy=value->clone();require(copy,"Clone failed");require(copy->initialize()==0,"Cached clone initialization changed");++calls;++successes;++clones;require(ModelOptInitCPUFactoryCalls()==1,"Clone called factory");copy->serialize(after.data());require(before==after&&std::string(copy->getPluginNamespace())=="init_trace_cpu","Clone wire/namespace changed");copy->destroy();}
            require(ModelOptInitCPUNewZCalls()==(route==4?(mode==1?1:2):0),"Reconstructed factory route changed");
            require(ModelOptInitCPUOldZCalls()==(route==3?(mode==1?1:2):0),"Legacy factory route changed");
            require(ModelOptInitCPUKVCalls()==(route==5?(mode==1?1:2):0),"KV factory route changed");
            require(ModelOptInitCPUQCalls()==(route==6?(mode==1?1:2):0),"Q factory route changed");
            value->serialize(after.data());require(before==after,"Initialization changed wire");factories+=ModelOptInitCPUFactoryCalls();require(ModelOptInitCPUOperatorCalls()==0,"GPU/operator stub called");value->terminate();
        }
        require(calls==72&&factories==56&&successes==24&&failures==48&&clones==8,"CPU accounting changed");
        std::printf("{\"status\":\"actual_o128_plugin_initialize_cpu_controls_passed\",\"actual_initialize_calls\":%d,\"factory_calls\":%d,\"successes\":%d,\"failures\":%d,\"cached_clone_initializations\":%d,\"unexpected_GPU_operator_calls\":0,\"trace_enabled\":%s,\"model_or_performance_qualified\":false}\n",calls,factories,successes,failures,clones,ModelOptPluginInitTraceEnabled()?"true":"false");return 0;
    }catch(const std::exception& error){std::fprintf(stderr,"%s\n",error.what());return 1;}
}
