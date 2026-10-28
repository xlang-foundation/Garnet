// SPDX-License-Identifier: Apache-2.0
#include "qwen35_compact_kv_plugin.h"
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <cstring>
#include <mutex>
#include <initializer_list>
namespace Garnet { namespace {
using namespace nvinfer1;
bool valid(Qwen35CompactKVConfig c){return (c.batch==1||c.batch==2||c.batch==4||c.batch==8)&&c.kv>=1&&c.kv<=32&&c.capacity>=32&&c.capacity<=8192&&c.tokens>=1&&c.tokens<=128&&c.tokens<=c.capacity&&c.dim>=1&&c.dim<=256;}
bool shape(const PluginTensorDesc& d,std::initializer_list<int> expected){if(d.type!=DataType::kFLOAT||d.format!=TensorFormat::kLINEAR||d.dims.nbDims!=expected.size())return false;int i=0;for(int n:expected)if(d.dims.d[i++]!=n)return false;return true;}
bool overlap(const void* a,size_t n,const void* b,size_t m){auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);return x>=y?x-y<m:y-x<n;}
}
IPluginV2DynamicExt* Qwen35CompactKVPlugin::clone() const noexcept {try{auto* p=new Qwen35CompactKVPlugin(c_);p->ns_=ns_;return p;}catch(...){return nullptr;}}
void Qwen35CompactKVPlugin::serialize(void* dst) const noexcept {if(dst)std::memcpy(dst,&c_,sizeof(c_));}
DimsExprs Qwen35CompactKVPlugin::getOutputDimensions(int index,const DimsExprs* in,int count,IExprBuilder& b) noexcept {
    if(!valid(c_)||!in||count!=4||index<0||index>2)return {};
    DimsExprs d{};if(index==2){d.nbDims=2;d.d[0]=b.constant(c_.batch);d.d[1]=b.constant(1);}
    else{d.nbDims=4;int values[]={c_.batch,c_.kv,c_.tokens,c_.dim};for(int i=0;i<4;++i)d.d[i]=b.constant(values[i]);}return d;
}
bool Qwen35CompactKVPlugin::supportsFormatCombination(int pos,const PluginTensorDesc* d,int in,int out) noexcept {return valid(c_)&&d&&in==4&&out==3&&pos>=0&&pos<7&&d[pos].type==DataType::kFLOAT&&d[pos].format==TensorFormat::kLINEAR;}
int Qwen35CompactKVPlugin::enqueue(const PluginTensorDesc* in,const PluginTensorDesc* out,const void* const* inputs,void* const* outputs,void* work,cudaStream_t stream) noexcept {
    if(!valid(c_)||!in||!out||!inputs||!outputs||!work)return 1;
    for(int i=0;i<4;++i)if(!inputs[i])return 1;for(int i=0;i<3;++i)if(!outputs[i])return 1;
    if(!shape(in[0],{c_.batch,c_.kv,c_.capacity,c_.dim})||!shape(in[1],{c_.batch,c_.kv,c_.capacity,c_.dim})||!shape(in[2],{c_.batch,c_.tokens})||!shape(in[3],{c_.batch,c_.tokens})||!shape(out[0],{c_.batch,c_.kv,c_.tokens,c_.dim})||!shape(out[1],{c_.batch,c_.kv,c_.tokens,c_.dim})||!shape(out[2],{c_.batch,1}))return 1;
    size_t full=size_t(c_.batch)*c_.kv*c_.capacity*c_.dim*4,small=size_t(c_.batch)*c_.kv*c_.tokens*c_.dim*4,controls=size_t(c_.batch)*c_.tokens*4;
    const void* p[]={inputs[0],inputs[1],inputs[2],inputs[3],outputs[0],outputs[1],outputs[2],work};size_t bytes[]={full,full,controls,controls,small,small,size_t(c_.batch)*4,sizeof(int)};
    for(int o=4;o<8;++o)for(int i=0;i<8;++i)if(i!=o&&overlap(p[o],bytes[o],p[i],bytes[i]))return 1;
    if(cudaMemsetAsync(work,0,sizeof(int),stream)!=cudaSuccess)return 1;
    if(PackQwen35CompactKV(c_,(const float*)inputs[0],(const float*)inputs[1],(const float*)inputs[2],(const float*)inputs[3],(float*)outputs[0],(float*)outputs[1],(int*)work,stream)!=cudaSuccess)return 1;
    return ReportQwen35CompactKVControl((const int*)work,(float*)outputs[2],c_.batch,stream)==cudaSuccess?0:1;
}
class Qwen35CompactKVCreator final:public IPluginCreator {
    std::string ns_;PluginFieldCollection fields_{};
public:
    const char* getPluginName() const noexcept override{return "GarnetQwen35CompactKV";}
    const char* getPluginVersion() const noexcept override{return "1";}
    const PluginFieldCollection* getFieldNames() noexcept override{return &fields_;}
    IPluginV2* createPlugin(const char*,const PluginFieldCollection*) noexcept override{return nullptr;}
    IPluginV2* deserializePlugin(const char*,const void* data,size_t bytes) noexcept override{if(!data||bytes!=sizeof(Qwen35CompactKVConfig))return nullptr;Qwen35CompactKVConfig c;std::memcpy(&c,data,bytes);if(!valid(c))return nullptr;try{auto* p=new Qwen35CompactKVPlugin(c);p->setPluginNamespace(ns_.c_str());return p;}catch(...){return nullptr;}}
    void setPluginNamespace(const char* value) noexcept override{ns_=value?value:"";}
    const char* getPluginNamespace() const noexcept override{return ns_.c_str();}
};
bool EnsureQwen35CompactKVPluginRegistered(){static Qwen35CompactKVCreator creator;static std::once_flag once;static bool okay=false;std::call_once(once,[&]{auto* registry=getPluginRegistry();okay=registry&&registry->registerCreator(creator,"");});return okay;}
}
