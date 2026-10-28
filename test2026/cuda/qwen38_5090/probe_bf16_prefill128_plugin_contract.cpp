// SPDX-License-Identifier: Apache-2.0
// CPU-only wrapper contract checks. No enqueue, CUDA API, builder or model call.
#include "qwen38_bf16_prefill128_kv_overlay_plugin.h"
#include "qwen38_bf16_low_batch_kv_overlay_plugin.h"
// Canonical FA offsets follow the separately qualified packed layout.
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
#include "qwen38_bf16_kv_overlay_plugin.h"
namespace Garnet {int Prefill128PluginCPUUnexpectedEnqueues() noexcept;}
using namespace Garnet;
using namespace nvinfer1;
namespace {
void require(bool v,const char* s){if(!v)throw std::runtime_error(s);}
PluginTensorDesc desc(std::initializer_list<int64_t> shape){
    PluginTensorDesc d{};d.type=DataType::kFLOAT;d.format=TensorFormat::kLINEAR;d.dims.nbDims=int(shape.size());
    int i=0;for(auto n:shape)d.dims.d[i++]=n;return d;
}
}
int main(){
    try{
        require(getInferLibVersion()==NV_TENSORRT_VERSION,"TensorRT runtime/header version differs");
        char loaded[MAX_PATH]{};const auto length=GetModuleFileNameA(GetModuleHandleA("nvinfer_10.dll"),loaded,MAX_PATH);
        require(length>0&&length<MAX_PATH,"Cannot identify loaded TensorRT DLL");for(DWORD i=0;i<length;++i)if(loaded[i]=='\\')loaded[i]='/';
        require(EnsureBF16KVPrefill128OverlayPluginRegistered(),"BF16 creator registration failed");
        auto* registry=getPluginRegistry();require(registry,"No TensorRT registry");
        auto* creator=registry->getPluginCreator("GarnetQwen38BF16KVPrefill128Overlay","1","");require(creator,"BF16 creator missing");
        require(EnsureBF16KVOverlayPluginRegistered(),"Legacy BF16 creator registration failed");
        auto* oldCreator=registry->getPluginCreator("GarnetQwen38BF16KVOverlay","1","");require(oldCreator,"Legacy creator missing");
        require(EnsureBF16KVLowBatchOverlayPluginRegistered(),"Old B1 registration failed");
        auto* b1Creator=registry->getPluginCreator("GarnetQwen38BF16KVLowBatchOverlay","1","");require(b1Creator,"Old B1 creator missing");
        int profiles=0,badWire=0,badDesc=0,badFormat=0,cross=0,legacyRoundTrips=0;
        for(int b:{1})for(int t:{128})for(int index=3;index<64;index+=4){
            const int64_t key=2482176+int64_t(index/4)*6676480;
            BF16KVPrefill128OverlayConfig config{{b,t,24,4,4096,256},106823680,key,key+2097152};require(ValidBF16KVPrefill128OverlayConfig(config),"Good BF16 config rejected");
            BF16KVPrefill128OverlayPlugin plugin(config);require(plugin.initialize()==0,"Initialize failed");
            require(std::string(plugin.getPluginType())=="GarnetQwen38BF16KVPrefill128Overlay"&&std::string(plugin.getPluginVersion())=="1"&&plugin.getNbOutputs()==1,"Wrong plugin identity");
            std::array<int64_t,9> expected{b,t,24,4,4096,256,106823680,key,key+2097152},wire{};
            require(plugin.getSerializationSize()==72,"Wrong wire size");plugin.serialize(wire.data());require(wire==expected,"Serialized fields differ");
            plugin.setPluginNamespace("study_bf16");auto* cloned=plugin.clone();require(cloned,"Clone failed");
            std::array<int64_t,9> cloneWire{};cloned->serialize(cloneWire.data());require(cloneWire==wire&&std::string(cloned->getPluginNamespace())=="study_bf16","Clone lost wire/namespace");cloned->destroy();
            auto* restored=creator->deserializePlugin("study",wire.data(),72);require(restored,"Deserialize failed");
            std::array<int64_t,9> restoredWire{};restored->serialize(restoredWire.data());require(restoredWire==wire&&std::string(restored->getPluginType())==plugin.getPluginType(),"Restored wire/type differs");restored->destroy();
            auto rejectWire=[&](const void* data,size_t bytes){auto* p=creator->deserializePlugin("study",data,bytes);if(p){p->destroy();throw std::runtime_error("Bad wire accepted");}++badWire;};
            for(int field=0;field<9;++field)for(auto value:{int64_t(-1),int64_t(0),int64_t(1),(std::numeric_limits<int64_t>::max)(),wire[field]+1}){
                if(value==wire[field])continue; // B1 is valid, not a negative wire.
                auto bad=wire;bad[field]=value;rejectWire(bad.data(),72);
            }
            for(auto bytes:{size_t(0),size_t(71),size_t(73),size_t(80)})rejectWire(wire.data(),bytes);
            rejectWire(nullptr,72);auto bad64=wire;bad64[1]=64;rejectWire(bad64.data(),72);
            auto legacy=wire;legacy[6]=173932544;rejectWire(legacy.data(),72);
            legacy=wire;legacy[8]=key+4194304;rejectWire(legacy.data(),72);
            require(creator->createPlugin("study",nullptr)==nullptr,"Unspecified field creation accepted");
            auto* crossed=oldCreator->deserializePlugin("study",wire.data(),72);if(crossed){crossed->destroy();throw std::runtime_error("Legacy creator accepted B1 wire");}++cross;
            auto* oldB1Cross=b1Creator->deserializePlugin("study",wire.data(),72);if(oldB1Cross){oldB1Cross->destroy();throw std::runtime_error("Old B1 accepted T128 wire");}++cross;
            for(int oldT:{4,32}){
                auto oldWire=wire;oldWire[1]=oldT;auto* oldB1=b1Creator->deserializePlugin("study",oldWire.data(),72);require(oldB1,"Old B1 profile rejected");
                std::array<int64_t,9> back{};oldB1->serialize(back.data());require(back==oldWire,"Old B1 wire differs");oldB1->destroy();++legacyRoundTrips;
                auto* rejected=creator->deserializePlugin("study",oldWire.data(),72);if(rejected){rejected->destroy();throw std::runtime_error("T128 accepted old B1 profile");}++cross;
            }
            for(int legacyBatch:{4,8})for(int oldT:{4,32}){
                auto oldWire=wire;oldWire[0]=legacyBatch;oldWire[1]=oldT;auto* oldPlugin=oldCreator->deserializePlugin("study",oldWire.data(),72);require(oldPlugin,"Legacy profile rejected");
                std::array<int64_t,9> back{};oldPlugin->serialize(back.data());require(back==oldWire,"Legacy roundtrip differs");oldPlugin->destroy();++legacyRoundTrips;
                auto* rejected=creator->deserializePlugin("study",oldWire.data(),72);if(rejected){rejected->destroy();throw std::runtime_error("B1 creator accepted B4/B8 wire");}++cross;
            }
            PluginTensorDesc inputs[8]={desc({b,t,24,256}),desc({b,t,4,256}),desc({b,t,4,256}),desc({b,t,24,256}),desc({b,106823680}),desc({b,t}),desc({b,t}),desc({b,t,24,256})};
            const size_t workspace=size_t(b)*t*24*4096*4;
            require(plugin.getWorkspaceSize(inputs,7,inputs+7,1)==workspace,"Good workspace differs");
            for(int pos=0;pos<8;++pos){
                require(plugin.supportsFormatCombination(pos,inputs,7,1),"Good format rejected");
                auto old=inputs[pos];inputs[pos].type=DataType::kHALF;require(!plugin.supportsFormatCombination(pos,inputs,7,1),"Half carrier accepted");++badFormat;
                inputs[pos]=old;inputs[pos].format=TensorFormat::kCHW16;require(!plugin.supportsFormatCombination(pos,inputs,7,1),"Nonlinear carrier accepted");++badFormat;inputs[pos]=old;
            }
            require(!plugin.supportsFormatCombination(-1,inputs,7,1)&&!plugin.supportsFormatCombination(8,inputs,7,1)&&!plugin.supportsFormatCombination(0,inputs,6,1)&&!plugin.supportsFormatCombination(0,inputs,7,2)&&!plugin.supportsFormatCombination(0,nullptr,7,1),"Bad format envelope accepted");badFormat+=5;
            for(int field=0;field<7;++field)for(int mode=0;mode<4;++mode){
                auto old=inputs[field];
                if(mode==0)inputs[field].dims.d[0]+=1;
                if(mode==1)inputs[field].dims.nbDims=0;
                if(mode==2)inputs[field].type=DataType::kHALF;
                if(mode==3)inputs[field].format=TensorFormat::kCHW16;
                require(plugin.getWorkspaceSize(inputs,7,inputs+7,1)==0,"Bad descriptor workspace accepted");++badDesc;inputs[field]=old;
            }
            require(plugin.getWorkspaceSize(inputs,6,inputs+7,1)==0&&plugin.getWorkspaceSize(inputs,7,inputs+7,2)==0&&plugin.getWorkspaceSize(nullptr,7,inputs+7,1)==0,"Bad workspace envelope accepted");badDesc+=3;
            plugin.terminate();++profiles;
        }
        require(profiles==16&&badWire==832&&badDesc==496&&badFormat==336&&cross==128&&legacyRoundTrips==96&&Prefill128PluginCPUUnexpectedEnqueues()==0,"Contract-control accounting changed");
        std::printf("{\"status\":\"prefill128_plugin_cpu_contract_passed\",\"profiles\":%d,\"roundtrips\":16,\"clones\":16,\"bad_wires\":%d,\"bad_descriptors\":%d,\"bad_formats\":%d,\"cross_dialect_rejections\":%d,\"legacy_roundtrips\":%d,\"unexpected_operator_calls\":%d,\"runtime_version\":%d,\"header_version\":%d,\"loaded_tensorrt_dll\":\"%s\",\"enqueue_tested\":false,\"model_qualified\":false,\"performance_admitted\":false}\n",profiles,badWire,badDesc,badFormat,cross,legacyRoundTrips,Prefill128PluginCPUUnexpectedEnqueues(),getInferLibVersion(),NV_TENSORRT_VERSION,loaded);return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
