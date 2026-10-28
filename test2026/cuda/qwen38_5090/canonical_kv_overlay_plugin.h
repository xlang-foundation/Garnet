// SPDX-License-Identifier: Apache-2.0
#pragma once
// Standalone study plugin candidate. Not linked into the Garnet runtime.
#include <NvInferRuntime.h>
#include <cstdint>
#include <string>
#include "attention_kv_delta_strided_core.cuh"
namespace Garnet {
struct CanonicalKVOverlayConfig {
    Qwen35AttentionCoreConfig attention{};
    std::int64_t state_width=0,key_offset=0,value_offset=0;
    int strategy=0; //0 compact,1 legacy geometry. Host chooses captured engine.
};
bool ValidCanonicalKVOverlayConfig(const CanonicalKVOverlayConfig&) noexcept;
class CanonicalKVOverlayPlugin final:public nvinfer1::IPluginV2DynamicExt {
    CanonicalKVOverlayConfig config_;std::string namespace_;
    bool matches(const nvinfer1::PluginTensorDesc*,int) const noexcept;
public:
    explicit CanonicalKVOverlayPlugin(CanonicalKVOverlayConfig);
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int,const nvinfer1::DimsExprs*,int,nvinfer1::IExprBuilder&) noexcept override;
    bool supportsFormatCombination(int,const nvinfer1::PluginTensorDesc*,int,int) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc*,int,const nvinfer1::DynamicPluginTensorDesc*,int) noexcept override{}
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc*,int,const nvinfer1::PluginTensorDesc*,int) const noexcept override;
    int enqueue(const nvinfer1::PluginTensorDesc*,const nvinfer1::PluginTensorDesc*,const void* const*,void* const*,void*,cudaStream_t) noexcept override;
    nvinfer1::DataType getOutputDataType(int,const nvinfer1::DataType*,int) const noexcept override{return nvinfer1::DataType::kFLOAT;}
    const char* getPluginType() const noexcept override{return "GarnetQwen35CanonicalKVOverlay";}
    const char* getPluginVersion() const noexcept override{return "1";}
    int getNbOutputs() const noexcept override{return 1;}
    int initialize() noexcept override{return 0;}
    void terminate() noexcept override{}
    size_t getSerializationSize() const noexcept override{return 10*sizeof(std::int64_t);}
    void serialize(void*) const noexcept override;
    void destroy() noexcept override{delete this;}
    void setPluginNamespace(const char* value) noexcept override{namespace_=value?value:"";}
    const char* getPluginNamespace() const noexcept override{return namespace_.c_str();}
};
bool EnsureCanonicalKVOverlayPluginRegistered();
}
