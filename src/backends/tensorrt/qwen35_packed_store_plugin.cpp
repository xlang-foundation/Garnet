// SPDX-License-Identifier: Apache-2.0
#include "qwen35_packed_store_plugin.h"
#include <cuda_runtime.h>
#include <cstring>
#include <stdexcept>
#include <utility>
namespace Garnet {
using namespace nvinfer1;
bool ValidQwen35PackedStoreConfig(const std::vector<int32_t>& c) noexcept {
    if(c.size()<7||c[0]<=0||c[1]<=0||c[1]>c[0]||(c[2]!=0&&c[2]!=1)||(c[3]!=0&&c[3]!=1)||c[4]<1||c[4]>6||c.size()!=size_t(5+2*c[4]))return false;
    for(int i=0;i<c[4];++i){
        int start=c[5+2*i],width=c[6+2*i];
        if(start<c[1]||start>c[0]||width<=0||width>c[0]-start)return false;
        for(int j=0;j<i;++j){int other=c[5+2*j],length=c[6+2*j];if(start<other+length&&other<start+width)return false;}
    }
    return true;
}
namespace {
bool overlap(const void* a,size_t an,const void* b,size_t bn) noexcept {
    auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);return x<=y?y-x<an:x-y<bn;
}
class PackedStore final:public IPluginV3,public IPluginV3OneCore,public IPluginV3OneBuildV2,public IPluginV3OneRuntime {
    std::vector<int32_t> c_;PluginField field_{};PluginFieldCollection fields_{};
    bool descriptors(const PluginTensorDesc* in,int ni,const PluginTensorDesc* out,int no)const noexcept {
        if(!in||!out||ni!=c_[4]+2||no!=2)return false;
        for(int i=0;i<ni;++i)if(in[i].dims.nbDims!=2||in[i].type!=DataType::kFLOAT||in[i].format!=TensorFormat::kLINEAR)return false;
        auto batch=in[0].dims.d[0];if(batch!=1&&batch!=2&&batch!=4&&batch!=8)return false;
        for(int i=0;i<ni;++i)if(in[i].dims.d[0]!=batch||in[i].dims.d[1]!=(i==0?c_[0]:i==1?c_[1]:c_[6+2*(i-2)]))return false;
        for(int i=0;i<no;++i)if(out[i].dims.nbDims!=2||out[i].type!=DataType::kFLOAT||out[i].format!=TensorFormat::kLINEAR||out[i].dims.d[0]!=batch||out[i].dims.d[1]!=c_[i])return false;
        return true;
    }
public:
    explicit PackedStore(std::vector<int32_t> c):c_(std::move(c)) {
        if(!ValidQwen35PackedStoreConfig(c_))throw std::invalid_argument("Invalid Qwen3.5 packed store config");
        field_={"config",c_.data(),PluginFieldType::kINT32,int32_t(c_.size())};fields_={1,&field_};
    }
    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override {
        if(type==PluginCapabilityType::kCORE)return static_cast<IPluginV3OneCore*>(this);
        if(type==PluginCapabilityType::kBUILD)return static_cast<IPluginV3OneBuildV2*>(this);
        if(type==PluginCapabilityType::kRUNTIME)return static_cast<IPluginV3OneRuntime*>(this);return nullptr;
    }
    IPluginV3* clone() noexcept override {try{return new PackedStore(c_);}catch(...){return nullptr;}}
    const char* getPluginName()const noexcept override{return "GarnetQwen35PackedStore";}
    const char* getPluginVersion()const noexcept override{return "1";}
    const char* getPluginNamespace()const noexcept override{return "";}
    int32_t getNbOutputs()const noexcept override{return 2;}
    int32_t getAliasedInput(int32_t output) noexcept override{return output==0&&c_[3]?0:-1;}
    int32_t configurePlugin(const DynamicPluginTensorDesc* in,int32_t ni,const DynamicPluginTensorDesc* out,int32_t no) noexcept override {
        if(!in||!out||ni!=c_[4]+2||no!=2)return 1;
        PluginTensorDesc a[8]{},b[2]{};for(int i=0;i<ni;++i)a[i]=in[i].desc;for(int i=0;i<no;++i)b[i]=out[i].desc;
        return descriptors(a,ni,b,no)?0:1;
    }
    int32_t getOutputDataTypes(DataType* out,int32_t no,const DataType* in,int32_t ni)const noexcept override {
        if(!out||!in||no!=2||ni!=c_[4]+2)return 1;for(int i=0;i<ni;++i)if(in[i]!=DataType::kFLOAT)return 1;
        out[0]=out[1]=DataType::kFLOAT;return 0;
    }
    int32_t getOutputShapes(const DimsExprs* in,int32_t ni,const DimsExprs*,int32_t ns,DimsExprs* out,int32_t no,IExprBuilder&) noexcept override {
        if(!in||!out||ni!=c_[4]+2||no!=2||ns)return 1;for(int i=0;i<ni;++i)if(in[i].nbDims!=2)return 1;
        out[0]=in[0];out[1]=in[1];return 0;
    }
    bool supportsFormatCombination(int32_t pos,const DynamicPluginTensorDesc* io,int32_t ni,int32_t no) noexcept override {
        return io&&ni==c_[4]+2&&no==2&&pos>=0&&pos<ni+no&&io[pos].desc.type==DataType::kFLOAT&&io[pos].desc.format==TensorFormat::kLINEAR;
    }
    int32_t onShapeChange(const PluginTensorDesc* in,int32_t ni,const PluginTensorDesc* out,int32_t no) noexcept override{return descriptors(in,ni,out,no)?0:1;}
    int32_t enqueue(const PluginTensorDesc* in,const PluginTensorDesc* out,const void*const* inputs,void*const* outputs,void*,cudaStream_t stream) noexcept override {
        if(!descriptors(in,c_[4]+2,out,2)||!inputs||!outputs||!outputs[0]||!outputs[1])return 1;
        for(int i=0;i<c_[4]+2;++i)if(!inputs[i])return 1;
        size_t batch=size_t(in[0].dims.d[0]),root=batch*size_t(c_[0])*4,hidden=batch*size_t(c_[1])*4;
        if(bool(c_[3])!=(inputs[0]==outputs[0])||(!c_[3]&&overlap(inputs[0],root,outputs[0],root)))return 1;
        if(overlap(outputs[0],root,inputs[1],hidden)||overlap(outputs[0],root,outputs[1],hidden)||overlap(outputs[1],hidden,inputs[0],root))return 1;
        if(outputs[1]!=inputs[1]&&overlap(outputs[1],hidden,inputs[1],hidden))return 1;
        for(int i=0;i<c_[4];++i){size_t bytes=batch*size_t(c_[6+2*i])*4;if(overlap(outputs[0],root,inputs[i+2],bytes)||overlap(outputs[1],hidden,inputs[i+2],bytes))return 1;}
        if(!c_[3]&&cudaMemcpyAsync(outputs[0],inputs[0],root,cudaMemcpyDeviceToDevice,stream)!=cudaSuccess)return 1;
        for(int i=0;i<c_[4];++i){size_t width=size_t(c_[6+2*i])*4;if(cudaMemcpy2DAsync(static_cast<char*>(outputs[0])+size_t(c_[5+2*i])*4,size_t(c_[0])*4,inputs[i+2],width,width,batch,cudaMemcpyDeviceToDevice,stream)!=cudaSuccess)return 1;}
        if(outputs[1]!=inputs[1]&&cudaMemcpyAsync(outputs[1],inputs[1],hidden,cudaMemcpyDeviceToDevice,stream)!=cudaSuccess)return 1;
        if(c_[2]&&cudaMemcpy2DAsync(outputs[0],size_t(c_[0])*4,inputs[1],size_t(c_[1])*4,size_t(c_[1])*4,batch,cudaMemcpyDeviceToDevice,stream)!=cudaSuccess)return 1;
        return 0;
    }
    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override{return clone();}
    const PluginFieldCollection* getFieldsToSerialize() noexcept override{return &fields_;}
};
class Creator final:public IPluginCreatorV3One {
    PluginField field_{"config",nullptr,PluginFieldType::kINT32,0};PluginFieldCollection fields_{1,&field_};
public:
    IPluginV3* createPlugin(const char*,const PluginFieldCollection* fc,TensorRTPhase) noexcept override {
        if(!fc||fc->nbFields!=1||!fc->fields)return nullptr;const auto& f=fc->fields[0];
        if(!f.name||std::strcmp(f.name,"config")||!f.data||f.type!=PluginFieldType::kINT32||f.length<7||f.length>17)return nullptr;
        try{std::vector<int32_t> c(size_t(f.length));std::memcpy(c.data(),f.data,c.size()*sizeof(int32_t));if(!ValidQwen35PackedStoreConfig(c))return nullptr;return new PackedStore(std::move(c));}catch(...){return nullptr;}
    }
    const PluginFieldCollection* getFieldNames() noexcept override{return &fields_;}
    const char* getPluginName()const noexcept override{return "GarnetQwen35PackedStore";}
    const char* getPluginVersion()const noexcept override{return "1";}
    const char* getPluginNamespace()const noexcept override{return "";}
};
}
IPluginV3* CreateQwen35PackedStorePlugin(const std::vector<int32_t>& c){return new PackedStore(c);}
bool EnsureQwen35PackedStorePluginRegistered(){static Creator creator;static const bool registered=getPluginRegistry()->registerCreator(creator,"");return registered;}
}
