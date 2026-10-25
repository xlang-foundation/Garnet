// SPDX-License-Identifier: Apache-2.0
#include "qwen38_bf16_kv_float_reference_plugin.h"
#include <NvInfer.h>
#include <array>
#include <cstring>
#include <mutex>
#include <stdexcept>
namespace Garnet {
using namespace nvinfer1;
namespace {
using Wire=std::array<std::int64_t,6>;
static_assert(sizeof(Wire)==48,"Reference wire must contain exactly six int64 fields");
Wire wire(Qwen35AttentionCoreConfig c){return {c.batch,c.tokens,c.heads,c.kv,c.capacity,c.dim};}
bool valid(Wire c){return c[0]==4&&(c[1]==4||c[1]==32)&&c[2]==24&&c[3]==4&&c[4]==4096&&c[5]==256;}
}
BF16KVFloatReferencePlugin::BF16KVFloatReferencePlugin(Qwen35AttentionCoreConfig c):config_(c){
    if(!valid(wire(c)))throw std::invalid_argument("BF16 float-cache reference requires B4/T4or32/H24/KV4/C4096/D256");
}
bool BF16KVFloatReferencePlugin::inputsMatch(const PluginTensorDesc* in,int count) const noexcept {
    if(!in||count!=8)return false;const auto c=config_;
    const int64_t expected[8][4]={{c.batch,c.tokens,c.heads,c.dim},{c.batch,c.tokens,c.kv,c.dim},
        {c.batch,c.tokens,c.kv,c.dim},{c.batch,c.tokens,c.heads,c.dim},{c.batch,c.kv,c.capacity,c.dim},
        {c.batch,c.kv,c.capacity,c.dim},{c.batch,c.tokens,0,0},{c.batch,c.tokens,0,0}};
    for(int i=0;i<8;++i){
        if(in[i].dims.nbDims!=(i<6?4:2)||in[i].type!=DataType::kFLOAT||in[i].format!=TensorFormat::kLINEAR)return false;
        for(int d=0;d<in[i].dims.nbDims;++d)if(in[i].dims.d[d]!=expected[i][d])return false;
    }return true;
}
bool BF16KVFloatReferencePlugin::outputsMatch(const PluginTensorDesc* in,const PluginTensorDesc* out,int count) const noexcept {
    if(!in||!out||count!=3)return false;
    for(int i=0;i<3;++i){
        const int source=i==0?0:i==1?4:5;
        if(out[i].type!=DataType::kFLOAT||out[i].format!=TensorFormat::kLINEAR||out[i].dims.nbDims!=4)return false;
        for(int d=0;d<4;++d)if(out[i].dims.d[d]!=in[source].dims.d[d])return false;
    }return true;
}
IPluginV2DynamicExt* BF16KVFloatReferencePlugin::clone() const noexcept {
    try{auto* result=new BF16KVFloatReferencePlugin(config_);result->namespace_=namespace_;return result;}catch(...){return nullptr;}
}
DimsExprs BF16KVFloatReferencePlugin::getOutputDimensions(int index,const DimsExprs* in,int count,IExprBuilder&) noexcept {
    if(!in||count!=8||index<0||index>=3)return {};const int source=index==0?0:index==1?4:5;
    return in[source].nbDims==4?in[source]:DimsExprs{};
}
bool BF16KVFloatReferencePlugin::supportsFormatCombination(int pos,const PluginTensorDesc* desc,int inputs,int outputs) noexcept {
    return desc&&inputs==8&&outputs==3&&pos>=0&&pos<11&&desc[pos].type==DataType::kFLOAT&&desc[pos].format==TensorFormat::kLINEAR;
}
size_t BF16KVFloatReferencePlugin::getWorkspaceSize(const PluginTensorDesc* in,int count,const PluginTensorDesc* out,int outputs) const noexcept {
    return inputsMatch(in,count)&&outputsMatch(in,out,outputs)?Qwen38BF16KVFloatReferenceWorkspace(config_):0;
}
int BF16KVFloatReferencePlugin::enqueue(const PluginTensorDesc* in,const PluginTensorDesc* out,
    const void* const* inputs,void* const* outputs,void* workspace,cudaStream_t stream) noexcept {
    if(!inputsMatch(in,8)||!outputsMatch(in,out,3)||!inputs||!outputs)return 1;
    return RunQwen38BF16KVFloatReference(config_,inputs,outputs,workspace,stream);
}
void BF16KVFloatReferencePlugin::serialize(void* data) const noexcept {
    if(data){const auto value=wire(config_);std::memcpy(data,value.data(),sizeof(value));}
}
class BF16KVFloatReferenceCreator final:public IPluginCreator {
    std::string namespace_;PluginFieldCollection fields_{};
public:
    const char* getPluginName() const noexcept override{return "GarnetQwen38BF16KVFloatReference";}
    const char* getPluginVersion() const noexcept override{return "1";}
    const PluginFieldCollection* getFieldNames() noexcept override{return &fields_;}
    IPluginV2* createPlugin(const char*,const PluginFieldCollection*) noexcept override{return nullptr;}
    IPluginV2* deserializePlugin(const char*,const void* data,size_t size) noexcept override {
        if(!data||size!=sizeof(Wire))return nullptr;Wire value{};std::memcpy(value.data(),data,size);if(!valid(value))return nullptr;
        Qwen35AttentionCoreConfig config{int(value[0]),int(value[1]),int(value[2]),int(value[3]),int(value[4]),int(value[5])};
        try{auto* result=new BF16KVFloatReferencePlugin(config);result->setPluginNamespace(namespace_.c_str());return result;}catch(...){return nullptr;}
    }
    void setPluginNamespace(const char* value) noexcept override{namespace_=value?value:"";}
    const char* getPluginNamespace() const noexcept override{return namespace_.c_str();}
};
bool EnsureBF16KVFloatReferencePluginRegistered(){
    static BF16KVFloatReferenceCreator creator;static std::once_flag once;static bool okay=false;
    std::call_once(once,[&]{auto* registry=getPluginRegistry();okay=registry&&registry->registerCreator(creator,"");});return okay;
}
}
