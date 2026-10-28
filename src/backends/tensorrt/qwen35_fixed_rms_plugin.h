// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <NvInferRuntime.h>
#include <string>
namespace Garnet {
// Optional fixed row-reduction order. FP32 input, scale, accumulation and output;
// callers materialize Qwen's (1 + learned weight) scale before this boundary.
class Qwen35FixedRMSPlugin final:public nvinfer1::IPluginV2DynamicExt {
    float epsilon_;std::string ns_;
public:
    explicit Qwen35FixedRMSPlugin(float epsilon):epsilon_(epsilon){}
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int,const nvinfer1::DimsExprs*,int,nvinfer1::IExprBuilder&) noexcept override;
    bool supportsFormatCombination(int,const nvinfer1::PluginTensorDesc*,int,int) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc*,int,const nvinfer1::DynamicPluginTensorDesc*,int) noexcept override{}
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc*,int,const nvinfer1::PluginTensorDesc*,int) const noexcept override{return 0;}
    int enqueue(const nvinfer1::PluginTensorDesc*,const nvinfer1::PluginTensorDesc*,const void* const*,void* const*,void*,cudaStream_t) noexcept override;
    nvinfer1::DataType getOutputDataType(int,const nvinfer1::DataType*,int) const noexcept override{return nvinfer1::DataType::kFLOAT;}
    const char* getPluginType() const noexcept override{return "GarnetQwen35FixedRMS";}
    const char* getPluginVersion() const noexcept override{return "1";}
    int getNbOutputs() const noexcept override{return 1;}
    int initialize() noexcept override;
    void terminate() noexcept override{}
    size_t getSerializationSize() const noexcept override{return sizeof(float);}
    void serialize(void*) const noexcept override;
    void destroy() noexcept override{delete this;}
    void setPluginNamespace(const char* value) noexcept override{ns_=value?value:"";}
    const char* getPluginNamespace() const noexcept override{return ns_.c_str();}
};
bool EnsureQwen35FixedRMSPluginRegistered();
}
