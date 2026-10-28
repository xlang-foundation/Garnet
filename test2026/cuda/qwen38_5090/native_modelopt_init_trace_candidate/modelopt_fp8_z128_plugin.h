// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <NvInferRuntime.h>
#include "modelopt_fp8_z128_canonical.h"
#include <cstdint>
#include <cstring>
#include <string>
namespace Garnet {
class ModelOptFP8Z128Plugin final:public nvinfer1::IPluginV2DynamicExt {
    struct State {int32_t rows,columns;float weightScale,inputScale;} state_;
    std::string namespace_;
    std::shared_ptr<ModelOptFP8Z128Context> context_;
public:
    ModelOptFP8Z128Plugin(int rows,int columns,float weightScale,float inputScale);
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int,const nvinfer1::DimsExprs*,int,nvinfer1::IExprBuilder&) noexcept override;
    bool supportsFormatCombination(int,const nvinfer1::PluginTensorDesc*,int,int) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc*,int,const nvinfer1::DynamicPluginTensorDesc*,int) noexcept override {}
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc*,int,const nvinfer1::PluginTensorDesc*,int) const noexcept override;
    int enqueue(const nvinfer1::PluginTensorDesc*,const nvinfer1::PluginTensorDesc*,const void* const*,void* const*,void*,cudaStream_t) noexcept override;
    nvinfer1::DataType getOutputDataType(int,const nvinfer1::DataType*,int) const noexcept override{return nvinfer1::DataType::kBF16;}
    const char* getPluginType() const noexcept override{return "GarnetModelOptFP8Z128Canonical";}
    const char* getPluginVersion() const noexcept override{return "1";}
    int getNbOutputs() const noexcept override{return 1;}
    int initialize() noexcept override;
    void terminate() noexcept override{context_.reset();}
    size_t getSerializationSize() const noexcept override{return sizeof(State)+sizeof(uint32_t);}
    void serialize(void* data) const noexcept override{
        if(!data)return;std::memcpy(data,&state_,sizeof(State));const uint32_t policy=0x5a313238u;
        std::memcpy(static_cast<unsigned char*>(data)+sizeof(State),&policy,sizeof(policy));
    }
    static ModelOptFP8Z128Plugin* Deserialize(const void*,size_t) noexcept;
    void destroy() noexcept override{delete this;}
    void setPluginNamespace(const char* value) noexcept override{namespace_=value?value:"";}
    const char* getPluginNamespace() const noexcept override{return namespace_.c_str();}
};
bool EnsureModelOptFP8Z128PluginRegistered();
}
