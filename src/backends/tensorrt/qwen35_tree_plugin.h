// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <NvInfer.h>
#include <string>
#include <vector>
#include "../../cuda/qwen35_tree_kernels.cuh"
#include "qwen35_delta_plugin.h"
namespace Garnet {
enum class Qwen35TreePluginKind:int {Metadata=0,Delta=1,Attention=2};
struct Qwen35TreePluginConfig {
    int kind=0,batch=0,nodes=0,capacity=0,key_heads=0,value_heads=0,heads=0,kv=0,dim=0,kernel=0;
    float epsilon=0;
};
Qwen35TreePluginConfig TreeMetadataPluginConfig(Qwen35TreeMetadataConfig);
Qwen35TreePluginConfig TreeDeltaPluginConfig(Qwen35DeltaConfig,Qwen35TreeMetadataConfig);
Qwen35TreePluginConfig TreeAttentionPluginConfig(Qwen35AttentionCoreConfig);
bool ValidQwen35TreePluginConfig(const Qwen35TreePluginConfig&) noexcept;
class Qwen35TreePlugin final:public nvinfer1::IPluginV2DynamicExt {
    Qwen35TreePluginConfig c_;std::string ns_;
    bool Matches(const nvinfer1::PluginTensorDesc*,int) const noexcept;
public:
    explicit Qwen35TreePlugin(Qwen35TreePluginConfig);
    const Qwen35TreePluginConfig& Config() const noexcept{return c_;}
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int,const nvinfer1::DimsExprs*,int,nvinfer1::IExprBuilder&) noexcept override;
    bool supportsFormatCombination(int,const nvinfer1::PluginTensorDesc*,int,int) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc*,int,const nvinfer1::DynamicPluginTensorDesc*,int) noexcept override{}
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc*,int,const nvinfer1::PluginTensorDesc*,int) const noexcept override;
    int enqueue(const nvinfer1::PluginTensorDesc*,const nvinfer1::PluginTensorDesc*,const void* const*,void* const*,void*,cudaStream_t) noexcept override;
    nvinfer1::DataType getOutputDataType(int,const nvinfer1::DataType*,int) const noexcept override;
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override{return "1";}
    int getNbOutputs() const noexcept override{return c_.kind==0?1:2;}
    int initialize() noexcept override{return 0;}
    void terminate() noexcept override{}
    size_t getSerializationSize() const noexcept override{return sizeof(c_);}
    void serialize(void*) const noexcept override;
    void destroy() noexcept override{delete this;}
    void setPluginNamespace(const char* v) noexcept override{ns_=v?v:"";}
    const char* getPluginNamespace() const noexcept override{return ns_.c_str();}
};
bool EnsureQwen35TreePluginsRegistered();
// Only the direct metadata producer in this same network is admitted. This
// binds B,N,capacity independently of the otherwise ambiguous [B,836] shape.
bool Qwen35TreeMetadataProducerMatches(nvinfer1::INetworkDefinition&,nvinfer1::ITensor*,Qwen35TreeMetadataConfig) noexcept;
nvinfer1::IPluginV2Layer* AddQwen35TreePlugin(nvinfer1::INetworkDefinition&,const Qwen35TreePluginConfig&,
    nvinfer1::ITensor* const*,int,std::vector<nvinfer1::IPluginV2*>& owned);
}
