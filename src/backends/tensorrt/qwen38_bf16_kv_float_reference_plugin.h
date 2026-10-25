// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <NvInferRuntime.h>
#include <cstdint>
#include <string>
#include "../../cuda/qwen38_bf16_kv_float_reference.cuh"
namespace Garnet {
// Unwired research reference. Ordinary FP32 full caches hold BF16-roundtrip
// values; this type never accepts packed canonical buffers or compact offsets.
class BF16KVFloatReferencePlugin final:public nvinfer1::IPluginV2DynamicExt {
    Qwen35AttentionCoreConfig config_;std::string namespace_;
    bool inputsMatch(const nvinfer1::PluginTensorDesc*,int) const noexcept;
    bool outputsMatch(const nvinfer1::PluginTensorDesc*,const nvinfer1::PluginTensorDesc*,int) const noexcept;
public:
    explicit BF16KVFloatReferencePlugin(Qwen35AttentionCoreConfig);
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int,const nvinfer1::DimsExprs*,int,nvinfer1::IExprBuilder&) noexcept override;
    bool supportsFormatCombination(int,const nvinfer1::PluginTensorDesc*,int,int) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc*,int,const nvinfer1::DynamicPluginTensorDesc*,int) noexcept override{}
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc*,int,const nvinfer1::PluginTensorDesc*,int) const noexcept override;
    int enqueue(const nvinfer1::PluginTensorDesc*,const nvinfer1::PluginTensorDesc*,const void* const*,void* const*,void*,cudaStream_t) noexcept override;
    nvinfer1::DataType getOutputDataType(int,const nvinfer1::DataType*,int) const noexcept override{return nvinfer1::DataType::kFLOAT;}
    const char* getPluginType() const noexcept override{return "GarnetQwen38BF16KVFloatReference";}
    const char* getPluginVersion() const noexcept override{return "1";}
    int getNbOutputs() const noexcept override{return 3;}
    int initialize() noexcept override{return 0;}
    void terminate() noexcept override{}
    size_t getSerializationSize() const noexcept override{return 6*sizeof(std::int64_t);}
    void serialize(void*) const noexcept override;
    void destroy() noexcept override{delete this;}
    void setPluginNamespace(const char* value) noexcept override{namespace_=value?value:"";}
    const char* getPluginNamespace() const noexcept override{return namespace_.c_str();}
};
bool EnsureBF16KVFloatReferencePluginRegistered();
}
