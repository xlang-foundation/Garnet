// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "../common/streamed_bf16_projection.h"
#include <NvInferRuntime.h>
#include <memory>
#include <string>
namespace Garnet {
class StreamedBF16ProjectionPlugin final:public nvinfer1::IPluginV2DynamicExt {
    std::string serialized_,namespace_;
    StreamedBF16ProjectionSpec spec_;
    std::shared_ptr<StreamedBF16Projection> projection_;
public:
    explicit StreamedBF16ProjectionPlugin(StreamedBF16ProjectionSpec);
    explicit StreamedBF16ProjectionPlugin(std::string);
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int,const nvinfer1::DimsExprs*,int,nvinfer1::IExprBuilder&) noexcept override;
    bool supportsFormatCombination(int,const nvinfer1::PluginTensorDesc*,int,int) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc*,int,const nvinfer1::DynamicPluginTensorDesc*,int) noexcept override {}
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc*,int,const nvinfer1::PluginTensorDesc*,int) const noexcept override{return 32ULL<<20;}
    int enqueue(const nvinfer1::PluginTensorDesc*,const nvinfer1::PluginTensorDesc*,const void* const*,void* const*,void*,cudaStream_t) noexcept override;
    nvinfer1::DataType getOutputDataType(int,const nvinfer1::DataType*,int) const noexcept override{return spec_.floatOutput?nvinfer1::DataType::kFLOAT:nvinfer1::DataType::kBF16;}
    const char* getPluginType() const noexcept override{return "GarnetStreamedBF16Projection";}
    const char* getPluginVersion() const noexcept override{return spec_.fp8Compute?"3":(spec_.floatOutput?"2":"1");}
    int getNbOutputs() const noexcept override{return 1;}
    int initialize() noexcept override;
    void terminate() noexcept override{projection_.reset();}
    size_t getSerializationSize() const noexcept override{return serialized_.size();}
    void serialize(void* destination) const noexcept override;
    void destroy() noexcept override{delete this;}
    void setPluginNamespace(const char* value) noexcept override{namespace_=value?value:"";}
    const char* getPluginNamespace() const noexcept override{return namespace_.c_str();}
};
bool EnsureStreamedBF16ProjectionPluginRegistered();
}
