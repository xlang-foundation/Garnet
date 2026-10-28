// SPDX-License-Identifier: Apache-2.0
#if !defined(GARNET_CPU_ONLY) && !defined(GARNET_METAL)
#include "qwen35_tree_plugin.h"
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <mutex>
#include <stdexcept>
namespace Garnet { namespace {
using namespace nvinfer1;
const char* type(int kind){return kind==0?"GarnetQwen35TreeMetadata":kind==1?"GarnetQwen35TreeDelta":"GarnetQwen35TreeAttention";}
int inputs(int kind){return kind==0?3:kind==1?12:8;}
Qwen35TreeMetadataConfig profile(const Qwen35TreePluginConfig& c){return {c.batch,c.nodes,c.capacity};}
Qwen35DeltaConfig delta(const Qwen35TreePluginConfig& c){return {c.key_heads,c.value_heads,128,128,c.kernel,c.epsilon};}
Qwen35AttentionCoreConfig attention(const Qwen35TreePluginConfig& c){return {c.batch,c.nodes,c.heads,c.kv,c.capacity,c.dim};}
bool shape(const PluginTensorDesc& d,std::initializer_list<int> expected,DataType dtype=DataType::kFLOAT){
    if(d.type!=dtype||d.format!=TensorFormat::kLINEAR||d.dims.nbDims!=expected.size())return false;int i=0;for(int n:expected)if(d.dims.d[i++]!=n)return false;return true;
}
class Creator final:public IPluginCreator {
    int kind_;std::string ns_;PluginFieldCollection fields_{};
public:
    explicit Creator(int k):kind_(k){}
    const char* getPluginName() const noexcept override{return type(kind_);}
    const char* getPluginVersion() const noexcept override{return "1";}
    const PluginFieldCollection* getFieldNames() noexcept override{return &fields_;}
    IPluginV2* createPlugin(const char*,const PluginFieldCollection*) noexcept override{return nullptr;}
    IPluginV2* deserializePlugin(const char*,const void* data,size_t bytes) noexcept override{
        if(!data||bytes!=sizeof(Qwen35TreePluginConfig))return nullptr;Qwen35TreePluginConfig c;std::memcpy(&c,data,bytes);
        if(c.kind!=kind_||!ValidQwen35TreePluginConfig(c))return nullptr;
        try{auto* p=new Qwen35TreePlugin(c);p->setPluginNamespace(ns_.c_str());return p;}catch(...){return nullptr;}
    }
    void setPluginNamespace(const char* s) noexcept override{ns_=s?s:"";}
    const char* getPluginNamespace() const noexcept override{return ns_.c_str();}
};
}
Qwen35TreePluginConfig TreeMetadataPluginConfig(Qwen35TreeMetadataConfig p){Qwen35TreePluginConfig c{};c.batch=p.batch;c.nodes=p.nodes;c.capacity=p.capacity;return c;}
Qwen35TreePluginConfig TreeDeltaPluginConfig(Qwen35DeltaConfig d,Qwen35TreeMetadataConfig p){auto c=TreeMetadataPluginConfig(p);c.kind=1;c.key_heads=d.kh;c.value_heads=d.vh;c.kernel=d.kernel;c.epsilon=d.epsilon;if(d.kd!=128||d.vd!=128)throw std::invalid_argument("Tree delta requires128-wide state");return c;}
Qwen35TreePluginConfig TreeAttentionPluginConfig(Qwen35AttentionCoreConfig a){auto c=TreeMetadataPluginConfig({a.batch,a.tokens,a.capacity});c.kind=2;c.heads=a.heads;c.kv=a.kv;c.dim=a.dim;return c;}
bool ValidQwen35TreePluginConfig(const Qwen35TreePluginConfig& c) noexcept{
    if(c.kind<0||c.kind>2||!ValidQwen35TreeMetadataConfig(profile(c)))return false;
    if(c.kind==0)return c.key_heads==0&&c.value_heads==0&&c.heads==0&&c.kv==0&&c.dim==0&&c.kernel==0&&c.epsilon==0;
    if(c.kind==1)return c.heads==0&&c.kv==0&&c.dim==0&&Qwen35TreeDeltaWorkspace(delta(c),profile(c))>0;
    return c.key_heads==0&&c.value_heads==0&&c.kernel==0&&c.epsilon==0&&Qwen35TreeAttentionWorkspace(attention(c))>0;
}
Qwen35TreePlugin::Qwen35TreePlugin(Qwen35TreePluginConfig c):c_(c){if(!ValidQwen35TreePluginConfig(c))throw std::invalid_argument("Invalid Qwen3.5 tree plugin configuration");}
bool Qwen35TreePlugin::Matches(const nvinfer1::PluginTensorDesc* d,int count) const noexcept{
    if(!d||count!=inputs(c_.kind))return false;
    int b=c_.batch,n=c_.nodes;
    if(c_.kind==0)return shape(d[0],{b,n})&&shape(d[1],{b,n})&&shape(d[2],{b,1});
    if(c_.kind==1){int vh=c_.value_heads,ch=2*c_.key_heads*128+vh*128;
        return shape(d[0],{b,n,ch})&&shape(d[1],{b,n,vh*128})&&shape(d[2],{b,n,vh})&&shape(d[3],{b,n,vh})&&
            shape(d[4],{b,ch,c_.kernel})&&shape(d[5],{b,vh,128,128})&&shape(d[6],{b,n})&&shape(d[7],{ch,1,c_.kernel})&&
            shape(d[8],{vh})&&shape(d[9],{vh})&&shape(d[10],{128})&&shape(d[11],{b,836},nvinfer1::DataType::kINT32);
    }
    return shape(d[0],{b,n,c_.heads,c_.dim})&&shape(d[1],{b,n,c_.kv,c_.dim})&&shape(d[2],{b,n,c_.kv,c_.dim})&&
        shape(d[3],{b,n,c_.heads,c_.dim})&&shape(d[4],{b,c_.kv,c_.capacity,c_.dim})&&shape(d[5],{b,c_.kv,c_.capacity,c_.dim})&&
        shape(d[6],{b,n})&&shape(d[7],{b,836},nvinfer1::DataType::kINT32);
}
nvinfer1::IPluginV2DynamicExt* Qwen35TreePlugin::clone() const noexcept{try{auto* p=new Qwen35TreePlugin(c_);p->ns_=ns_;return p;}catch(...){return nullptr;}}
const char* Qwen35TreePlugin::getPluginType() const noexcept{return type(c_.kind);}
nvinfer1::DataType Qwen35TreePlugin::getOutputDataType(int,const nvinfer1::DataType*,int) const noexcept{return c_.kind==0?nvinfer1::DataType::kINT32:nvinfer1::DataType::kFLOAT;}
nvinfer1::DimsExprs Qwen35TreePlugin::getOutputDimensions(int index,const nvinfer1::DimsExprs* in,int count,nvinfer1::IExprBuilder& builder) noexcept{
    if(!in||count!=inputs(c_.kind)||index<0||index>=getNbOutputs())return {};
    if(c_.kind!=0&&index==0)return in[c_.kind==1?1:0];
    nvinfer1::DimsExprs d{};d.nbDims=2;d.d[0]=builder.constant(c_.batch);d.d[1]=builder.constant(c_.kind==0?836:1);return d;
}
bool Qwen35TreePlugin::supportsFormatCombination(int pos,const nvinfer1::PluginTensorDesc* d,int in,int out) noexcept{
    if(!d||in!=inputs(c_.kind)||out!=getNbOutputs()||pos<0||pos>=in+out)return false;
    bool integer=c_.kind==0?pos==in:pos==(c_.kind==1?11:7);
    return d[pos].format==nvinfer1::TensorFormat::kLINEAR&&d[pos].type==(integer?nvinfer1::DataType::kINT32:nvinfer1::DataType::kFLOAT);
}
size_t Qwen35TreePlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* in,int count,const nvinfer1::PluginTensorDesc*,int out) const noexcept{
    if(out!=getNbOutputs()||!Matches(in,count))return 0;
    return c_.kind==0?0:c_.kind==1?Qwen35TreeDeltaWorkspace(delta(c_),profile(c_)):Qwen35TreeAttentionWorkspace(attention(c_));
}
int Qwen35TreePlugin::enqueue(const nvinfer1::PluginTensorDesc* in,const nvinfer1::PluginTensorDesc* out,const void* const* data,void* const* result,void* work,cudaStream_t stream) noexcept{
    using namespace nvinfer1;if(!Matches(in,inputs(c_.kind))||!out||!data||!result)return 1;
    if(c_.kind==0){if(!shape(out[0],{c_.batch,836},DataType::kINT32))return 1;return RunQwen35TreeMetadata(profile(c_),(const float*)data[0],(const float*)data[1],(const float*)data[2],(Qwen35TreeMetadata*)result[0],stream);}
    if(!shape(out[1],{c_.batch,1}))return 1;
    if(c_.kind==1){if(!shape(out[0],{c_.batch,c_.nodes,c_.value_heads*128}))return 1;return RunQwen35TreeDelta(delta(c_),profile(c_),data,(const Qwen35TreeMetadata*)data[11],(float*)result[0],(float*)result[1],work,stream);}
    if(!shape(out[0],{c_.batch,c_.nodes,c_.heads,c_.dim}))return 1;
    return RunQwen35TreeAttention(attention(c_),data,(const Qwen35TreeMetadata*)data[7],(float*)result[0],(float*)result[1],work,stream);
}
void Qwen35TreePlugin::serialize(void* out) const noexcept{if(out)std::memcpy(out,&c_,sizeof(c_));}
bool EnsureQwen35TreePluginsRegistered(){static Creator metadata(0),state(1),attn(2);static std::once_flag once;static bool okay=false;
    std::call_once(once,[&]{auto* registry=::getPluginRegistry();okay=registry&&registry->registerCreator(metadata,"")&&registry->registerCreator(state,"")&&registry->registerCreator(attn,"");});return okay;
}
bool Qwen35TreeMetadataProducerMatches(nvinfer1::INetworkDefinition& network,nvinfer1::ITensor* tensor,Qwen35TreeMetadataConfig expected) noexcept{
    using namespace nvinfer1;if(!tensor||tensor->getType()!=DataType::kINT32||!ValidQwen35TreeMetadataConfig(expected))return false;
    auto s=tensor->getDimensions();if(s.nbDims!=2||s.d[0]!=expected.batch||s.d[1]!=836)return false;
    for(int i=0;i<network.getNbLayers();++i){auto* layer=network.getLayer(i);if(!layer||layer->getType()!=LayerType::kPLUGIN_V2||layer->getNbOutputs()!=1||layer->getOutput(0)!=tensor)continue;
        auto& plugin=static_cast<IPluginV2Layer*>(layer)->getPlugin();if(std::strcmp(plugin.getPluginType(),type(0))||std::strcmp(plugin.getPluginVersion(),"1")||plugin.getSerializationSize()!=sizeof(Qwen35TreePluginConfig))return false;
        Qwen35TreePluginConfig c;plugin.serialize(&c);return ValidQwen35TreePluginConfig(c)&&c.kind==0&&c.batch==expected.batch&&c.nodes==expected.nodes&&c.capacity==expected.capacity;
    }return false;
}
nvinfer1::IPluginV2Layer* AddQwen35TreePlugin(nvinfer1::INetworkDefinition& network,const Qwen35TreePluginConfig& c,
    nvinfer1::ITensor* const* in,int count,std::vector<nvinfer1::IPluginV2*>& owned){
    if(!ValidQwen35TreePluginConfig(c)||!in||count!=inputs(c.kind))throw std::invalid_argument("Invalid native tree plugin arguments");
    for(int i=0;i<count;++i)if(!in[i])throw std::invalid_argument("Native tree plugin input missing");
    if(c.kind&&!Qwen35TreeMetadataProducerMatches(network,in[c.kind==1?11:7],profile(c)))throw std::invalid_argument("Tree metadata must come from the matching producer in this network");
    if(!EnsureQwen35TreePluginsRegistered())throw std::runtime_error("Native tree plugin registration failed");
    auto* plugin=new Qwen35TreePlugin(c);owned.push_back(plugin);auto* layer=network.addPluginV2(in,count,*plugin);
    if(!layer)throw std::runtime_error("Native tree plugin construction failed");return layer;
}
}
#endif
