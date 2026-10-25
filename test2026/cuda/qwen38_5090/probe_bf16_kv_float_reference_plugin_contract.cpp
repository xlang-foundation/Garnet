// SPDX-License-Identifier: Apache-2.0
// CPU metadata only: never enqueue a valid kernel or construct a model/engine.
#include "qwen38_bf16_kv_float_reference_plugin.h"
#include <NvInfer.h>
#include <NvInferVersion.h>
#include <array>
#include <climits>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
using namespace Garnet;
using namespace nvinfer1;
namespace {
void require(bool okay,const char* message){if(!okay)throw std::runtime_error(message);}
PluginTensorDesc desc(std::initializer_list<int64_t> shape){
    PluginTensorDesc d{};d.type=DataType::kFLOAT;d.format=TensorFormat::kLINEAR;d.dims.nbDims=int(shape.size());
    int i=0;for(auto value:shape)d.dims.d[i++]=value;return d;
}
}
int main(){
    try{
        require(getInferLibVersion()==NV_TENSORRT_VERSION,"TensorRT header/runtime version differs");
        char loaded[MAX_PATH]{};const auto length=GetModuleFileNameA(GetModuleHandleA("nvinfer_10.dll"),loaded,MAX_PATH);
        require(length>0&&length<MAX_PATH,"Cannot identify loaded TensorRT DLL");for(DWORD i=0;i<length;++i)if(loaded[i]=='\\')loaded[i]='/';
        require(EnsureBF16KVFloatReferencePluginRegistered(),"Reference creator registration failed");
        auto* registry=getPluginRegistry();require(registry,"No TensorRT registry");
        auto* creator=registry->getPluginCreator("GarnetQwen38BF16KVFloatReference","1","");require(creator,"Reference creator missing");
        int profiles=0,clones=0,roundtrips=0,badWire=0,badConfig=0,badFormat=0,badDesc=0,badEnqueue=0;
        for(int tokens:{4,32}){
            Qwen35AttentionCoreConfig config{4,tokens,24,4,4096,256};BF16KVFloatReferencePlugin plugin(config);
            std::array<int64_t,6> expected{4,tokens,24,4,4096,256},wire{};
            require(plugin.initialize()==0&&plugin.getNbOutputs()==3&&std::string(plugin.getPluginType())=="GarnetQwen38BF16KVFloatReference"&&std::string(plugin.getPluginVersion())=="1","Wrong plugin identity");
            require(plugin.getSerializationSize()==48,"Reference wire must be48bytes");plugin.serialize(wire.data());require(wire==expected,"Reference serialized fields differ");
            plugin.setPluginNamespace("reference_cpu");auto* clone=plugin.clone();require(clone,"Clone failed");
            std::array<int64_t,6> back{};clone->serialize(back.data());require(back==wire&&std::string(clone->getPluginNamespace())=="reference_cpu","Clone lost fields/namespace");clone->destroy();++clones;
            creator->setPluginNamespace("reference_restored");auto* restored=creator->deserializePlugin("cpu",wire.data(),48);require(restored,"Deserialize failed");
            restored->serialize(back.data());require(back==wire&&std::string(restored->getPluginNamespace())=="reference_restored"&&std::string(restored->getPluginType())==plugin.getPluginType(),"Restored fields/type/namespace differ");restored->destroy();creator->setPluginNamespace("");++roundtrips;
            auto rejectWire=[&](const void* data,size_t size){auto* p=creator->deserializePlugin("bad",data,size);if(p){p->destroy();throw std::runtime_error("Invalid wire accepted");}++badWire;};
            for(int field=0;field<6;++field){
                for(auto value:{int64_t(-1),int64_t(0),int64_t(1),(std::numeric_limits<int64_t>::max)(),wire[field]+1}){auto bad=wire;bad[field]=value;rejectWire(bad.data(),48);}
                for(int value:{-1,0,1,INT_MAX,int(wire[field])+1}){
                    std::array<int,6> bad{4,tokens,24,4,4096,256};bad[field]=value;bool rejected=false;
                    try{BF16KVFloatReferencePlugin p({bad[0],bad[1],bad[2],bad[3],bad[4],bad[5]});}catch(const std::invalid_argument&){rejected=true;}
                    require(rejected,"Invalid constructor profile accepted");++badConfig;
                }
            }
            for(size_t size:{size_t(0),size_t(24),size_t(47),size_t(49),size_t(56)})rejectWire(wire.data(),size);rejectWire(nullptr,48);
            require(creator->createPlugin("cpu",nullptr)==nullptr,"Unspecified fields accepted");
            PluginTensorDesc d[11]={desc({4,tokens,24,256}),desc({4,tokens,4,256}),desc({4,tokens,4,256}),desc({4,tokens,24,256}),desc({4,4,4096,256}),desc({4,4,4096,256}),desc({4,tokens}),desc({4,tokens}),desc({4,tokens,24,256}),desc({4,4,4096,256}),desc({4,4,4096,256})};
            const size_t workspace=size_t(4)*tokens*24*4096*4+2*size_t(4)*tokens*4*256*4;
            require(plugin.getWorkspaceSize(d,8,d+8,3)==workspace,"Valid reference workspace differs");
            for(int pos=0;pos<11;++pos){
                require(plugin.supportsFormatCombination(pos,d,8,3),"Valid format rejected");auto old=d[pos];
                for(auto type:{DataType::kHALF,DataType::kBF16,DataType::kINT8}){d[pos].type=type;require(!plugin.supportsFormatCombination(pos,d,8,3),"Invalid type accepted");++badFormat;}
                d[pos]=old;d[pos].format=TensorFormat::kCHW16;require(!plugin.supportsFormatCombination(pos,d,8,3),"Invalid format accepted");++badFormat;d[pos]=old;
            }
            require(!plugin.supportsFormatCombination(-1,d,8,3)&&!plugin.supportsFormatCombination(11,d,8,3)&&!plugin.supportsFormatCombination(0,d,7,3)&&!plugin.supportsFormatCombination(0,d,8,2)&&!plugin.supportsFormatCombination(0,nullptr,8,3),"Invalid format envelope accepted");badFormat+=5;
            for(int i=0;i<11;++i)for(int mode=0;mode<5;++mode){
                auto old=d[i];if(mode==0)++d[i].dims.nbDims;if(mode==1)d[i].type=DataType::kBF16;if(mode==2)d[i].format=TensorFormat::kCHW16;if(mode==3)++d[i].dims.d[0];if(mode==4)++d[i].dims.d[d[i].dims.nbDims-1];
                require(plugin.getWorkspaceSize(d,8,d+8,3)==0,"Invalid descriptor admitted workspace");++badDesc;
                require(plugin.enqueue(d,d+8,nullptr,nullptr,nullptr,nullptr)!=0,"Invalid descriptor enqueue accepted");++badEnqueue;d[i]=old;
            }
            require(plugin.getWorkspaceSize(nullptr,8,d+8,3)==0&&plugin.getWorkspaceSize(d,8,nullptr,3)==0&&plugin.getWorkspaceSize(d,7,d+8,3)==0&&plugin.getWorkspaceSize(d,8,d+8,2)==0,"Invalid workspace envelope accepted");badDesc+=4;
            const void* inputs[8]{};void* outputs[3]{};
            require(plugin.enqueue(d,d+8,nullptr,outputs,nullptr,nullptr)!=0&&plugin.enqueue(d,d+8,inputs,nullptr,nullptr,nullptr)!=0&&plugin.enqueue(d,nullptr,inputs,outputs,nullptr,nullptr)!=0&&plugin.enqueue(d,d+8,inputs,outputs,nullptr,nullptr)!=0,"Invalid enqueue envelope accepted");badEnqueue+=4;
            ++profiles;
        }
        require(profiles==2&&clones==2&&roundtrips==2&&badWire==72&&badConfig==60&&badFormat==98&&badDesc==118&&badEnqueue==118,"CPU contract coverage changed");
        std::printf("{\"status\":\"bf16_float_reference_plugin_cpu_passed\",\"profiles\":%d,\"clones\":%d,\"roundtrips\":%d,\"bad_wires\":%d,\"bad_constructors\":%d,\"bad_formats\":%d,\"bad_descriptors\":%d,\"invalid_enqueue_controls\":%d,\"runtime_version\":%d,\"header_version\":%d,\"loaded_tensorrt_dll\":\"%s\",\"gpu_enqueue_tested\":false,\"model_qualified\":false,\"performance_admitted\":false}\n",profiles,clones,roundtrips,badWire,badConfig,badFormat,badDesc,badEnqueue,getInferLibVersion(),NV_TENSORRT_VERSION,loaded);
        return 0;
    }catch(const std::exception& error){std::fprintf(stderr,"%s\n",error.what());return 1;}
}
