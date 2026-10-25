// SPDX-License-Identifier: Apache-2.0
#pragma once
// Separate opt-in wire type: packed BF16 caches must never use the FP32 plugin.
#include <NvInferRuntime.h>
#include <cstdint>
#include <string>
#include "../../cuda/qwen38_bf16_kv_overlay.cuh"
namespace Garnet {
struct BF16KVOverlayConfig {
    Qwen35AttentionCoreConfig attention{};
    std::int64_t state_width=0,key_offset=0,value_offset=0;
};
bool ValidBF16KVOverlayConfig(const BF16KVOverlayConfig&) noexcept;
class BF16KVOverlayPlugin final:public nvinfer1::IPluginV2DynamicExt {
    BF16KVOverlayConfig config_;std::string namespace_;
    bool matches(const nvinfer1::PluginTensorDesc*,int) const noexcept;
public:
    explicit BF16KVOverlayPlugin(BF16KVOverlayConfig);
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int,const nvinfer1::DimsExprs*,int,nvinfer1::IExprBuilder&) noexcept override;
    bool supportsFormatCombination(int,const nvinfer1::PluginTensorDesc*,int,int) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc*,int,const nvinfer1::DynamicPluginTensorDesc*,int) noexcept override{}
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc*,int,const nvinfer1::PluginTensorDesc*,int) const noexcept override;
    int enqueue(const nvinfer1::PluginTensorDesc*,const nvinfer1::PluginTensorDesc*,const void* const*,void* const*,void*,cudaStream_t) noexcept override;
    nvinfer1::DataType getOutputDataType(int,const nvinfer1::DataType*,int) const noexcept override{return nvinfer1::DataType::kFLOAT;}
    const char* getPluginType() const noexcept override{return "GarnetQwen38BF16KVOverlay";}
    const char* getPluginVersion() const noexcept override{return "1";}
    int getNbOutputs() const noexcept override{return 1;}
    int initialize() noexcept override{return 0;}
    void terminate() noexcept override{}
    size_t getSerializationSize() const noexcept override{return 9*sizeof(std::int64_t);}
    void serialize(void*) const noexcept override;
    void destroy() noexcept override{delete this;}
    void setPluginNamespace(const char* value) noexcept override{namespace_=value?value:"";}
    const char* getPluginNamespace() const noexcept override{return namespace_.c_str();}
};
bool EnsureBF16KVOverlayPluginRegistered();
}
