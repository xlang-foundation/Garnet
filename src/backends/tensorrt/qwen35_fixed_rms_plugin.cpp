// SPDX-License-Identifier: Apache-2.0
#include "qwen35_fixed_rms_plugin.h"
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <mutex>
extern "C" cudaError_t runRMSNormFP32(const float*,const float*,float*,int,int,float,cudaStream_t);
namespace Garnet { namespace {
bool valid(float epsilon){return std::isfinite(epsilon)&&epsilon>0;}
bool overlap(const void* a,size_t n,const void* b,size_t m){auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);return x>=y?x-y<m:y-x<n;}
}
using namespace nvinfer1;
IPluginV2DynamicExt* Qwen35FixedRMSPlugin::clone() const noexcept {
    try{auto* p=new Qwen35FixedRMSPlugin(epsilon_);p->ns_=ns_;return p;}catch(...){return nullptr;}
}
DimsExprs Qwen35FixedRMSPlugin::getOutputDimensions(int index,const DimsExprs* in,int count,IExprBuilder&) noexcept {
    return in&&index==0&&count==2&&in[0].nbDims>=1&&in[0].nbDims<=4&&in[1].nbDims==1?in[0]:DimsExprs{};
}
bool Qwen35FixedRMSPlugin::supportsFormatCombination(int pos,const PluginTensorDesc* d,int in,int out) noexcept {
    return d&&in==2&&out==1&&pos>=0&&pos<3&&d[pos].type==DataType::kFLOAT&&d[pos].format==TensorFormat::kLINEAR;
}
int Qwen35FixedRMSPlugin::initialize() noexcept{return valid(epsilon_)?0:1;}
void Qwen35FixedRMSPlugin::serialize(void* out) const noexcept{if(out)std::memcpy(out,&epsilon_,sizeof(epsilon_));}
int Qwen35FixedRMSPlugin::enqueue(const PluginTensorDesc* in,const PluginTensorDesc* out,const void* const* inputs,void* const* outputs,void*,cudaStream_t stream) noexcept {
    if(!in||!out||!inputs||!outputs||!inputs[0]||!inputs[1]||!outputs[0]||!valid(epsilon_))return 1;
    for(int i=0;i<2;++i)if(in[i].type!=DataType::kFLOAT||in[i].format!=TensorFormat::kLINEAR)return 1;
    if(out[0].type!=DataType::kFLOAT||out[0].format!=TensorFormat::kLINEAR||in[0].dims.nbDims<1||in[0].dims.nbDims>4||out[0].dims.nbDims!=in[0].dims.nbDims||in[1].dims.nbDims!=1)return 1;
    int rows=1;const int rank=in[0].dims.nbDims,hidden=in[0].dims.d[rank-1];
    if(hidden<1||hidden>16384||in[1].dims.d[0]!=hidden)return 1;
    for(int i=0;i<rank;++i){
        const int d=in[0].dims.d[i];if(d<1||out[0].dims.d[i]!=d)return 1;
        if(i+1<rank){if(rows>65536/d)return 1;rows*=d;}
    }
    for(const void* pointer:{inputs[0],inputs[1],static_cast<const void*>(outputs[0])})if(reinterpret_cast<uintptr_t>(pointer)%alignof(float))return 1;
    const size_t bytes=size_t(rows)*hidden*sizeof(float);
    if(overlap(outputs[0],bytes,inputs[0],bytes)||overlap(outputs[0],bytes,inputs[1],size_t(hidden)*sizeof(float)))return 1;
    return runRMSNormFP32(static_cast<const float*>(inputs[0]),static_cast<const float*>(inputs[1]),static_cast<float*>(outputs[0]),rows,hidden,epsilon_,stream)==cudaSuccess?0:1;
}
class Qwen35FixedRMSCreator final:public IPluginCreator {
    std::string ns_;PluginFieldCollection fields_{};
public:
    const char* getPluginName() const noexcept override{return "GarnetQwen35FixedRMS";}
    const char* getPluginVersion() const noexcept override{return "1";}
    const PluginFieldCollection* getFieldNames() noexcept override{return &fields_;}
    IPluginV2* createPlugin(const char*,const PluginFieldCollection*) noexcept override{return nullptr;}
    IPluginV2* deserializePlugin(const char*,const void* data,size_t bytes) noexcept override{
        if(!data||bytes!=sizeof(float))return nullptr;float epsilon;std::memcpy(&epsilon,data,bytes);if(!valid(epsilon))return nullptr;
        try{auto* p=new Qwen35FixedRMSPlugin(epsilon);p->setPluginNamespace(ns_.c_str());return p;}catch(...){return nullptr;}
    }
    void setPluginNamespace(const char* value) noexcept override{ns_=value?value:"";}
    const char* getPluginNamespace() const noexcept override{return ns_.c_str();}
};
bool EnsureQwen35FixedRMSPluginRegistered(){
    static Qwen35FixedRMSCreator creator;static std::once_flag once;static bool okay=false;
    std::call_once(once,[&]{auto* registry=getPluginRegistry();okay=registry&&registry->registerCreator(creator,"");});return okay;
}
}
