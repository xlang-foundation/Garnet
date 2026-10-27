// SPDX-License-Identifier: Apache-2.0
#include "modelopt_fp8_k_m4x4_plugin.h"
#include "modelopt_plugin_init_trace.h"
#include "../../cuda/modelopt_fp8.cuh"
#include <NvInfer.h>
#include <cmath>
#include <mutex>
#include <stdexcept>
namespace Garnet {
using namespace nvinfer1;
namespace {size_t AlignedKM4X4(size_t bytes){return (bytes+255)&~size_t(255);}}
ModelOptFP8KM4X4Plugin::ModelOptFP8KM4X4Plugin(int rows,int columns,float weightScale,float inputScale)
    :state_{rows,columns,weightScale,inputScale}{
    static_assert(sizeof(State)==16,"Stable separate KM4X4 FP8 wire");
    if(rows!=1024||columns!=5120||!std::isfinite(weightScale)||weightScale<=0||
       !std::isfinite(inputScale)||inputScale<=0||!std::isfinite(weightScale*inputScale)||weightScale*inputScale<=0)
        throw std::invalid_argument("KM4X4 FP8 requires exact full-attention K decode geometry and finite positive scales");
}
IPluginV2DynamicExt* ModelOptFP8KM4X4Plugin::clone() const noexcept{
    try{auto* result=new ModelOptFP8KM4X4Plugin(state_.rows,state_.columns,state_.weightScale,state_.inputScale);
        result->namespace_=namespace_;result->context_=context_;return result;}catch(...){return nullptr;}
}
ModelOptFP8KM4X4Plugin* ModelOptFP8KM4X4Plugin::Deserialize(const void* data,size_t length) noexcept{
    if(!data||length!=sizeof(State)+sizeof(uint32_t))return nullptr;
    State value{};uint32_t policy=0;std::memcpy(&value,data,sizeof(value));std::memcpy(&policy,static_cast<const unsigned char*>(data)+sizeof(value),sizeof(policy));
    if(policy!=0x4b343434u)return nullptr;
    try{return new ModelOptFP8KM4X4Plugin(value.rows,value.columns,value.weightScale,value.inputScale);}catch(...){return nullptr;}
}
DimsExprs ModelOptFP8KM4X4Plugin::getOutputDimensions(int index,const DimsExprs* input,int count,IExprBuilder& builder) noexcept{
    if(index||!input||count!=2||input[0].nbDims!=2||input[1].nbDims!=2||!input[0].d[0]||!input[0].d[1]||
       !input[0].d[0]->isConstant()||input[0].d[0]->getConstantValue()!=16||
       !input[0].d[1]->isConstant()||input[0].d[1]->getConstantValue()!=5120)return DimsExprs{};
    auto output=input[0];output.d[1]=builder.constant(state_.rows);return output;
}
bool ModelOptFP8KM4X4Plugin::supportsFormatCombination(int pos,const PluginTensorDesc* values,int inputs,int outputs) noexcept{
    if(!values||inputs!=2||outputs!=1||pos<0||pos>=3||values[pos].format!=TensorFormat::kLINEAR)return false;
    return pos==1?(values[pos].type==DataType::kINT8||values[pos].type==DataType::kINT32):values[pos].type==DataType::kBF16;
}
size_t ModelOptFP8KM4X4Plugin::getWorkspaceSize(const PluginTensorDesc* in,int count,const PluginTensorDesc* out,int outputs) const noexcept{
    if(!in||!out||count!=2||outputs!=1||in[0].dims.nbDims!=2||in[0].dims.d[0]!=16||in[0].dims.d[1]!=5120||
       in[1].dims.nbDims!=2||in[1].dims.d[0]!=state_.rows||in[1].dims.d[1]!=(in[1].type==DataType::kINT32?1280:5120)||
       out[0].dims.nbDims!=2||out[0].dims.d[0]!=16||out[0].dims.d[1]!=state_.rows)return 0;
    PluginTensorDesc all[3]={in[0],in[1],out[0]};
    for(int i=0;i<3;++i)if(!const_cast<ModelOptFP8KM4X4Plugin*>(this)->supportsFormatCombination(i,all,2,1))return 0;
    return AlignedKM4X4(16ULL*5120)+ModelOptFP8KM4X4BlasWorkspace;
}
int ModelOptFP8KM4X4Plugin::initialize() noexcept{
    return InitializeModelOptPluginContext(context_,[this]{return CreateModelOptFP8KM4X4Context(state_.rows);},"fp8_k_m4x4",this,state_.rows,state_.columns,4);
}
int ModelOptFP8KM4X4Plugin::enqueue(const PluginTensorDesc* in,const PluginTensorDesc* out,const void* const* input,
    void* const* output,void* workspace,cudaStream_t stream) noexcept{
    if(!input||!output||!input[0]||!input[1]||!output[0]||!workspace||!context_||!getWorkspaceSize(in,2,out,1))return 1;
    auto* packed=static_cast<unsigned char*>(workspace);auto* blas=packed+AlignedKM4X4(16ULL*5120);
    if(QuantizeModelOptFP8Activation(input[0],packed,state_.inputScale,16,5120,stream)!=cudaSuccess)return 1;
    return RunModelOptFP8KM4X4Matmul(context_,input[1],packed,output[0],16,state_.rows,5120,state_.weightScale*state_.inputScale,
        blas,ModelOptFP8KM4X4BlasWorkspace,stream)==CUBLAS_STATUS_SUCCESS?0:1;
}
class ModelOptFP8KM4X4Creator final:public IPluginCreator {
    std::string namespace_;PluginFieldCollection fields_{};
public:
    const char* getPluginName() const noexcept override{return "GarnetModelOptFP8KM4X4Canonical";}
    const char* getPluginVersion() const noexcept override{return "1";}
    const PluginFieldCollection* getFieldNames() noexcept override{return &fields_;}
    IPluginV2* createPlugin(const char*,const PluginFieldCollection*) noexcept override{return nullptr;}
    IPluginV2* deserializePlugin(const char*,const void* data,size_t length) noexcept override{
        auto* result=ModelOptFP8KM4X4Plugin::Deserialize(data,length);if(result)result->setPluginNamespace(namespace_.c_str());return result;
    }
    void setPluginNamespace(const char* value) noexcept override{namespace_=value?value:"";}
    const char* getPluginNamespace() const noexcept override{return namespace_.c_str();}
};
bool EnsureModelOptFP8KM4X4PluginRegistered(){
    static ModelOptFP8KM4X4Creator creator;static std::once_flag once;static bool okay=false;
    std::call_once(once,[&]{auto* registry=getPluginRegistry();okay=registry&&registry->registerCreator(creator,"");});return okay;
}
}
