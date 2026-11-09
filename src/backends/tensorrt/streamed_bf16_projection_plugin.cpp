// SPDX-License-Identifier: Apache-2.0
#include "streamed_bf16_projection_plugin.h"
#include <NvInfer.h>
#include <climits>
#include <cstring>
#include <iostream>
#include <mutex>
namespace Garnet {
using namespace nvinfer1;
StreamedBF16ProjectionPlugin::StreamedBF16ProjectionPlugin(StreamedBF16ProjectionSpec spec):
    serialized_(spec.Serialize()),spec_(std::move(spec)),projection_(new StreamedBF16Projection(spec_)){}
StreamedBF16ProjectionPlugin::StreamedBF16ProjectionPlugin(std::string value):
    serialized_(std::move(value)),spec_(StreamedBF16ProjectionSpec::Parse(serialized_)),projection_(new StreamedBF16Projection(spec_)){}
IPluginV2DynamicExt* StreamedBF16ProjectionPlugin::clone() const noexcept {
    try{auto* value=new StreamedBF16ProjectionPlugin(serialized_);value->namespace_=namespace_;
        // Engine/context clones share immutable mapping and bounded staging
        // ownership. Pool close, rather than each retired context, releases it.
        if(projection_)value->projection_=projection_;return value;}catch(...){return nullptr;}
}
int StreamedBF16ProjectionPlugin::initialize() noexcept {
    try{if(!projection_)projection_.reset(new StreamedBF16Projection(spec_));return 0;}catch(...){return 1;}
}
DimsExprs StreamedBF16ProjectionPlugin::getOutputDimensions(int index,const DimsExprs* inputs,int count,IExprBuilder& builder) noexcept {
    if(index||count!=1||inputs[0].nbDims<2)return DimsExprs{};
    auto output=inputs[0];output.d[output.nbDims-1]=builder.constant(spec_.columns);return output;
}
bool StreamedBF16ProjectionPlugin::supportsFormatCombination(int position,const PluginTensorDesc* values,int inputs,int outputs) noexcept {
    return inputs==1&&outputs==1&&position>=0&&position<2&&values[position].type==
        (position==1&&spec_.floatOutput?DataType::kFLOAT:DataType::kBF16)&&values[position].format==TensorFormat::kLINEAR;
}
int StreamedBF16ProjectionPlugin::enqueue(const PluginTensorDesc* input,const PluginTensorDesc*,const void* const* inputs,void* const* outputs,void* workspace,cudaStream_t stream) noexcept {
    try {
        if(!input||!inputs||!outputs||input[0].dims.nbDims<2||input[0].dims.d[input[0].dims.nbDims-1]!=spec_.inner)return 1;
        int64_t rows=1;for(int axis=0;axis<input[0].dims.nbDims-1;++axis){const auto extent=input[0].dims.d[axis];if(extent<1||extent>INT_MAX/rows)return 1;rows*=extent;}
        projection_->Run(inputs[0],outputs[0],int(rows),stream,workspace,32ULL<<20);return 0;
    }catch(const std::exception& error){std::cerr<<"[StreamedBF16Projection] "<<error.what()<<std::endl;return 1;}catch(...){return 1;}
}
void StreamedBF16ProjectionPlugin::serialize(void* destination) const noexcept {if(destination)std::memcpy(destination,serialized_.data(),serialized_.size());}
class StreamedProjectionCreator final:public IPluginCreator {
    std::string namespace_,version_;PluginFieldCollection fields_{};
public:
    explicit StreamedProjectionCreator(const char* version):version_(version){}
    const char* getPluginName() const noexcept override{return "GarnetStreamedBF16Projection";}
    const char* getPluginVersion() const noexcept override{return version_.c_str();}
    const PluginFieldCollection* getFieldNames() noexcept override{return &fields_;}
    IPluginV2* createPlugin(const char*,const PluginFieldCollection*) noexcept override{return nullptr;}
    IPluginV2* deserializePlugin(const char*,const void* data,size_t bytes) noexcept override {
        try{if(!data||!bytes||bytes>65536)return nullptr;auto* plugin=new StreamedBF16ProjectionPlugin(std::string(static_cast<const char*>(data),bytes));
            if(version_!=plugin->getPluginVersion()){delete plugin;return nullptr;}
            plugin->setPluginNamespace(namespace_.c_str());return plugin;}catch(...){return nullptr;}
    }
    void setPluginNamespace(const char* value) noexcept override{namespace_=value?value:"";}
    const char* getPluginNamespace() const noexcept override{return namespace_.c_str();}
};
bool EnsureStreamedBF16ProjectionPluginRegistered(){
    static StreamedProjectionCreator creator("1"),floatCreator("2"),fp8Creator("3");static std::once_flag once;static bool ready=false;
    std::call_once(once,[&]{auto* registry=getPluginRegistry();ready=registry&&registry->registerCreator(creator,"")&&registry->registerCreator(floatCreator,"")&&registry->registerCreator(fp8Creator,"");});return ready;
}
}
