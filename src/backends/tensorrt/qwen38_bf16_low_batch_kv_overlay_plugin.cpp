// SPDX-License-Identifier: Apache-2.0
#include "qwen38_bf16_low_batch_kv_overlay_plugin.h"
#include "../../cuda/qwen38_bf16_kv_commit.cuh"
#include <NvInfer.h>
#include <array>
#include <climits>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
namespace Garnet {
using namespace nvinfer1;
namespace {
std::array<std::int64_t,9> wire(const BF16KVLowBatchOverlayConfig& c) noexcept {
    return {c.attention.batch,c.attention.tokens,c.attention.heads,c.attention.kv,c.attention.capacity,c.attention.dim,
        c.state_width,c.key_offset,c.value_offset};
}
}
bool ValidBF16KVLowBatchOverlayConfig(const BF16KVLowBatchOverlayConfig& c) noexcept {
    const auto a=c.attention;
    return a.batch==1&&(a.tokens==4||a.tokens==32)&&a.heads==24&&a.kv==4&&a.capacity==4096&&a.dim==256&&
        c.state_width==Qwen38BF16KVCanonicalWidth&&c.key_offset>=2482176&&
        (c.key_offset-2482176)%6676480==0&&(c.key_offset-2482176)/6676480<16&&c.value_offset==c.key_offset+2097152;
}
BF16KVLowBatchOverlayPlugin::BF16KVLowBatchOverlayPlugin(BF16KVLowBatchOverlayConfig c):config_(c){
    if(!ValidBF16KVLowBatchOverlayConfig(c))throw std::invalid_argument("Unqualified BF16 canonical KV overlay configuration");
}
bool BF16KVLowBatchOverlayPlugin::matches(const PluginTensorDesc* input,int count) const noexcept {
    if(!input||count!=7)return false;const auto c=config_.attention;
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
IPluginV2DynamicExt* BF16KVLowBatchOverlayPlugin::clone() const noexcept {
    try{auto p=std::make_unique<BF16KVLowBatchOverlayPlugin>(config_);p->namespace_=namespace_;return p.release();}catch(...){return nullptr;}
}
DimsExprs BF16KVLowBatchOverlayPlugin::getOutputDimensions(int index,const DimsExprs* input,int count,IExprBuilder&) noexcept {
    return index==0&&input&&count==7?input[0]:DimsExprs{};
}
bool BF16KVLowBatchOverlayPlugin::supportsFormatCombination(int pos,const PluginTensorDesc* desc,int inputs,int outputs) noexcept {
    return desc&&inputs==7&&outputs==1&&pos>=0&&pos<8&&desc[pos].type==DataType::kFLOAT&&desc[pos].format==TensorFormat::kLINEAR;
}
size_t BF16KVLowBatchOverlayPlugin::getWorkspaceSize(const PluginTensorDesc* input,int count,const PluginTensorDesc*,int outputs) const noexcept {
    return outputs==1&&matches(input,count)?Qwen38BF16KVLowBatchCandidateWorkspace(config_.attention):0;
}
int BF16KVLowBatchOverlayPlugin::enqueue(const PluginTensorDesc* input,const PluginTensorDesc* output,
    const void* const* inputs,void* const* outputs,void* workspace,cudaStream_t stream) noexcept {
    if(!matches(input,7)||!output||!inputs||!outputs||!outputs[0])return 1;
    if(output[0].dims.nbDims!=4||output[0].type!=DataType::kFLOAT||output[0].format!=TensorFormat::kLINEAR)return 1;
    for(int d=0;d<4;++d)if(output[0].dims.d[d]!=input[0].dims.d[d])return 1;
    return RunQwen38BF16KVLowBatchCandidate(config_.attention,inputs,outputs[0],workspace,config_.key_offset,config_.value_offset,stream);
}
void BF16KVLowBatchOverlayPlugin::serialize(void* data) const noexcept {
    if(data){const auto fields=wire(config_);std::memcpy(data,fields.data(),fields.size()*sizeof(fields[0]));}
}
class BF16KVLowBatchOverlayCreator final:public IPluginCreator {
    std::string namespace_;PluginFieldCollection fields_{};
public:
    const char* getPluginName() const noexcept override{return "GarnetQwen38BF16KVLowBatchOverlay";}
    const char* getPluginVersion() const noexcept override{return "1";}
    const PluginFieldCollection* getFieldNames() noexcept override{return &fields_;}
    IPluginV2* createPlugin(const char*,const PluginFieldCollection*) noexcept override{return nullptr;}
    IPluginV2* deserializePlugin(const char*,const void* data,size_t size) noexcept override {
        if(!data||size!=9*sizeof(std::int64_t))return nullptr;
        std::array<std::int64_t,9> fields{};std::memcpy(fields.data(),data,size);
        for(int i=0;i<6;++i)if(fields[i]<1||fields[i]>INT_MAX)return nullptr;
        BF16KVLowBatchOverlayConfig config{{int(fields[0]),int(fields[1]),int(fields[2]),int(fields[3]),int(fields[4]),int(fields[5])},fields[6],fields[7],fields[8]};
        if(!ValidBF16KVLowBatchOverlayConfig(config))return nullptr;
        try{auto* p=new BF16KVLowBatchOverlayPlugin(config);p->setPluginNamespace(namespace_.c_str());return p;}catch(...){return nullptr;}
    }
    void setPluginNamespace(const char* value) noexcept override{namespace_=value?value:"";}
    const char* getPluginNamespace() const noexcept override{return namespace_.c_str();}
};
bool EnsureBF16KVLowBatchOverlayPluginRegistered(){
    static BF16KVLowBatchOverlayCreator creator;static std::once_flag once;static bool okay=false;
    std::call_once(once,[&]{auto* registry=getPluginRegistry();okay=registry&&registry->registerCreator(creator,"");});return okay;
}
}
