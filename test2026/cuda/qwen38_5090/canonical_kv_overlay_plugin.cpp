// SPDX-License-Identifier: Apache-2.0
#include "canonical_kv_overlay_plugin.h"
#include <NvInfer.h>
#include <array>
#include <climits>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
namespace Garnet {
using namespace nvinfer1;
namespace {
bool overlap(const void* first,size_t firstBytes,const void* second,size_t secondBytes) noexcept {
    const auto a=reinterpret_cast<std::uintptr_t>(first),b=reinterpret_cast<std::uintptr_t>(second);
    const auto maximum=(std::numeric_limits<std::uintptr_t>::max)();
    if(!first||!second||a>maximum-firstBytes||b>maximum-secondBytes)return true;
    return a<b+secondBytes&&b<a+firstBytes;
}
std::array<std::int64_t,10> wire(const CanonicalKVOverlayConfig& c) noexcept {
    return {c.attention.batch,c.attention.tokens,c.attention.heads,c.attention.kv,c.attention.capacity,c.attention.dim,
            c.state_width,c.key_offset,c.value_offset,c.strategy};
}
}
bool ValidCanonicalKVOverlayConfig(const CanonicalKVOverlayConfig& c) noexcept {
    const auto a=c.attention;
    // Deliberately restricted to the pinned study geometry/layout. Offset admits
    // exactly its sixteen full-attention layers; no unrelated state is a cache.
    return a.batch==4&&a.tokens==4&&a.heads==24&&a.kv==4&&a.capacity==4096&&a.dim==256&&
        c.state_width==173932544&&c.key_offset>=2482176&&c.key_offset<=165543936&&
        (c.key_offset-2482176)%10870784==0&&c.value_offset==c.key_offset+4194304&&
        c.value_offset+4194304<=c.state_width&&(c.strategy==0||c.strategy==1);
}
CanonicalKVOverlayPlugin::CanonicalKVOverlayPlugin(CanonicalKVOverlayConfig c):config_(c){
    if(!ValidCanonicalKVOverlayConfig(c))throw std::invalid_argument("Unqualified canonical KV overlay configuration");
}
bool CanonicalKVOverlayPlugin::matches(const PluginTensorDesc* input,int count) const noexcept {
    if(!input||count!=7)return false;
    const auto c=config_.attention;
    const std::int64_t expected[7][4]={{c.batch,c.tokens,c.heads,c.dim},{c.batch,c.tokens,c.kv,c.dim},
        {c.batch,c.tokens,c.kv,c.dim},{c.batch,c.tokens,c.heads,c.dim},{c.batch,config_.state_width,0,0},
        {c.batch,c.tokens,0,0},{c.batch,c.tokens,0,0}};
    for(int i=0;i<7;++i){
        const int rank=i<4?4:2;
        if(input[i].dims.nbDims!=rank||input[i].type!=DataType::kFLOAT||input[i].format!=TensorFormat::kLINEAR)return false;
        for(int d=0;d<rank;++d)if(input[i].dims.d[d]!=expected[i][d])return false;
    }
    return true;
}
IPluginV2DynamicExt* CanonicalKVOverlayPlugin::clone() const noexcept {
    try{auto result=std::make_unique<CanonicalKVOverlayPlugin>(config_);result->namespace_=namespace_;return result.release();}catch(...){return nullptr;}
}
DimsExprs CanonicalKVOverlayPlugin::getOutputDimensions(int index,const DimsExprs* input,int count,IExprBuilder&) noexcept {
    if(index!=0||!input||count!=7||input[0].nbDims!=4)return {};
    return input[0];
}
bool CanonicalKVOverlayPlugin::supportsFormatCombination(int pos,const PluginTensorDesc* desc,int inputs,int outputs) noexcept {
    return desc&&inputs==7&&outputs==1&&pos>=0&&pos<8&&desc[pos].type==DataType::kFLOAT&&desc[pos].format==TensorFormat::kLINEAR;
}
size_t CanonicalKVOverlayPlugin::getWorkspaceSize(const PluginTensorDesc* input,int count,const PluginTensorDesc*,int outputs) const noexcept {
    return outputs==1&&matches(input,count)&&config_.strategy==1?Qwen35AttentionCoreWorkspace(config_.attention):0;
}
int CanonicalKVOverlayPlugin::enqueue(const PluginTensorDesc* input,const PluginTensorDesc* output,
    const void* const* inputs,void* const* outputs,void* workspace,cudaStream_t stream) noexcept {
    if(!matches(input,7)||!output||!inputs||!outputs||!outputs[0])return 1;
    if(output[0].dims.nbDims!=4||output[0].type!=DataType::kFLOAT||output[0].format!=TensorFormat::kLINEAR)return 1;
    for(int d=0;d<4;++d)if(output[0].dims.d[d]!=input[0].dims.d[d])return 1;
    const auto c=config_.attention;
    const size_t contextBytes=size_t(c.batch)*c.tokens*c.heads*c.dim*sizeof(float);
    const size_t chunkBytes=size_t(c.batch)*c.tokens*c.kv*c.dim*sizeof(float);
    const size_t controlsBytes=size_t(c.batch)*c.tokens*sizeof(float);
    const size_t stateBytes=size_t(c.batch)*size_t(config_.state_width)*sizeof(float);
    const size_t sizes[7]={contextBytes,chunkBytes,chunkBytes,contextBytes,stateBytes,controlsBytes,controlsBytes};
    for(int i=0;i<7;++i)if(overlap(outputs[0],contextBytes,inputs[i],sizes[i]))return 1;
    const size_t workspaceBytes=config_.strategy==1?Qwen35AttentionCoreWorkspace(c):0;
    if(workspaceBytes){
        if(overlap(workspace,workspaceBytes,outputs[0],contextBytes))return 1;
        for(int i=0;i<7;++i)if(overlap(workspace,workspaceBytes,inputs[i],sizes[i]))return 1;
    }
    // The canonical input is one contiguous [B,state_width] tensor. Derive both
    // readonly cache bases without creating/materializing per-layer K/V slices.
    const auto* canonical=static_cast<const float*>(inputs[4]);
    const void* views[8]={inputs[0],inputs[1],inputs[2],inputs[3],canonical+config_.key_offset,
        canonical+config_.value_offset,inputs[5],inputs[6]};
    return RunQwen35AttentionKVOverlayStrided(c,views,outputs[0],workspace,config_.strategy,size_t(config_.state_width),stream);
}
void CanonicalKVOverlayPlugin::serialize(void* data) const noexcept {
    if(data){const auto fields=wire(config_);std::memcpy(data,fields.data(),fields.size()*sizeof(fields[0]));}
}
class CanonicalKVOverlayCreator final:public IPluginCreator {
    std::string namespace_;PluginFieldCollection fields_{};
public:
    const char* getPluginName() const noexcept override{return "GarnetQwen35CanonicalKVOverlay";}
    const char* getPluginVersion() const noexcept override{return "1";}
    const PluginFieldCollection* getFieldNames() noexcept override{return &fields_;}
    IPluginV2* createPlugin(const char*,const PluginFieldCollection*) noexcept override{return nullptr;}
    IPluginV2* deserializePlugin(const char*,const void* data,size_t size) noexcept override {
        if(!data||size!=10*sizeof(std::int64_t))return nullptr;
        std::array<std::int64_t,10> fields{};std::memcpy(fields.data(),data,size);
        for(int i=0;i<6;++i)if(fields[i]<1||fields[i]>INT_MAX)return nullptr;
        if(fields[9]<0||fields[9]>1)return nullptr;
        CanonicalKVOverlayConfig config{{int(fields[0]),int(fields[1]),int(fields[2]),int(fields[3]),int(fields[4]),int(fields[5])},fields[6],fields[7],fields[8],int(fields[9])};
        if(!ValidCanonicalKVOverlayConfig(config))return nullptr;
        try{auto* result=new CanonicalKVOverlayPlugin(config);result->setPluginNamespace(namespace_.c_str());return result;}catch(...){return nullptr;}
    }
    void setPluginNamespace(const char* value) noexcept override{namespace_=value?value:"";}
    const char* getPluginNamespace() const noexcept override{return namespace_.c_str();}
};
bool EnsureCanonicalKVOverlayPluginRegistered(){
    static CanonicalKVOverlayCreator creator;static std::once_flag once;static bool okay=false;
    std::call_once(once,[&]{auto* registry=getPluginRegistry();okay=registry&&registry->registerCreator(creator,"");});return okay;
}
}
