// SPDX-License-Identifier: Apache-2.0
// Pure CPU plugin contract: never initialize/enqueue; all GPU stubs must stay unused.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include "modelopt_fp8_z128_plugin.h"
#include "modelopt_fp8_z128_profile.h"
#include "modelopt_fp8_m128_plugin.h"
#include "modelopt_fp8_plugin.h"
#include <NvInfer.h>
#include <NvInferVersion.h>
#include <array>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
namespace Garnet {int FP8Z128CPUUnexpectedCalls() noexcept;}
using namespace Garnet;using namespace nvinfer1;
namespace {
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
PluginTensorDesc desc(int rows,int columns,DataType type){PluginTensorDesc d{};d.type=type;d.format=TensorFormat::kLINEAR;d.dims.nbDims=2;d.dims.d[0]=rows;d.dims.d[1]=columns;return d;}
}
int main(){
    try{
        require(getInferLibVersion()==NV_TENSORRT_VERSION,"TensorRT header/runtime differ");
        char loaded[MAX_PATH]{};auto length=GetModuleFileNameA(GetModuleHandleA("nvinfer_10.dll"),loaded,MAX_PATH);
        require(length>0&&length<MAX_PATH,"Cannot identify TensorRT DLL");for(DWORD i=0;i<length;++i)if(loaded[i]=='\\')loaded[i]='/';
        require(EnsureModelOptFP8PluginRegistered()&&EnsureModelOptFP8M128PluginRegistered()&&EnsureModelOptFP8Z128PluginRegistered(),"Creator registration failed");
        auto* registry=getPluginRegistry();require(registry,"No registry");
        auto* creator=registry->getPluginCreator("GarnetModelOptFP8Z128Canonical","1","");
        auto* oldCreator=registry->getPluginCreator("GarnetModelOptFP8","1","");
        auto* outputCreator=registry->getPluginCreator("GarnetModelOptFP8M128Canonical","1","");require(creator&&oldCreator&&outputCreator,"Creator missing");
        ModelOptFP8Z128Plugin plugin(6144,5120,.5f,.25f);std::array<uint32_t,5> wire{},other{};
        require(plugin.getSerializationSize()==20&&plugin.getNbOutputs()==1,"Wire/output size");plugin.serialize(wire.data());require(wire[0]==6144&&wire[1]==5120&&wire[4]==0x5a313238u,"Wrong Z wire fields");
        plugin.setPluginNamespace("z128_study");auto* clone=plugin.clone();require(clone,"No clone");clone->serialize(other.data());require(other==wire&&std::string(clone->getPluginNamespace())=="z128_study"&&std::string(clone->getPluginType())=="GarnetModelOptFP8Z128Canonical","Clone identity");clone->destroy();
        auto* restored=creator->deserializePlugin("Z128",wire.data(),20);require(restored,"No restored plugin");restored->serialize(other.data());require(other==wire,"Restored fields");restored->destroy();
        int badWire=0,badDescriptor=0,badFormat=0,badConstructor=0,policyRejections=0,crossWire=0;
        auto reject=[&](const void* data,size_t bytes){auto* value=creator->deserializePlugin("bad",data,bytes);if(value){value->destroy();throw std::runtime_error("Bad wire accepted");}++badWire;};
        for(int i=0;i<2;++i)for(uint32_t value:{0u,1u,128u,4096u,0xffffffffu}){auto bad=wire;bad[i]=value;reject(bad.data(),20);}
        for(int i=2;i<4;++i)for(uint32_t value:{0u,0x80000000u,0xbf800000u,0x7f800000u,0x7fc00000u}){auto bad=wire;bad[i]=value;reject(bad.data(),20);}
        for(uint32_t value:{0u,1u,127u,128u,129u,0xffffffffu}){auto bad=wire;bad[4]=value;reject(bad.data(),20);}
        for(size_t bytes:{size_t(0),size_t(1),size_t(15),size_t(16),size_t(19),size_t(21),size_t(24)})reject(wire.data(),bytes);reject(nullptr,20);
        for(auto* legacy:{oldCreator,outputCreator}){auto* crossed=legacy->deserializePlugin("old",wire.data(),20);require(!crossed,"Legacy creator accepted Z wire");++crossWire;}
        ModelOptFP8Plugin oldPlain(5120,6144,.5f,.25f),oldCanonical(5120,6144,.5f,.25f,true);ModelOptFP8M128Plugin oldOutput(5120,6144,.5f,.25f);
        for(auto* legacy:{static_cast<IPluginV2*>(&oldPlain),static_cast<IPluginV2*>(&oldCanonical),static_cast<IPluginV2*>(&oldOutput)}){
            std::array<uint32_t,5> oldWire{};legacy->serialize(oldWire.data());auto* value=creator->deserializePlugin("cross",oldWire.data(),legacy->getSerializationSize());require(!value,"Z accepted legacy wire");++crossWire;
        }
        require(creator->createPlugin("none",nullptr)==nullptr,"Ambiguous creator accepted");
        for(auto pair:{std::pair<int,int>{6143,5120},{6144,5119},{5120,6144},{0,5120},{6144,0},{-1,5120}}){
            try{ModelOptFP8Z128Plugin invalid(pair.first,pair.second,1,1);throw std::logic_error("Bad geometry accepted");}catch(const std::invalid_argument&){++badConstructor;}
        }
        for(int which=0;which<2;++which)for(float scale:{0.f,-1.f,std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()}){
            try{ModelOptFP8Z128Plugin invalid(6144,5120,which?1.f:scale,which?scale:1.f);throw std::logic_error("Bad scale accepted");}catch(const std::invalid_argument&){++badConstructor;}
        }
        for(float scale:{(std::numeric_limits<float>::max)(),(std::numeric_limits<float>::denorm_min)()}){
            try{ModelOptFP8Z128Plugin invalid(6144,5120,scale,scale);throw std::logic_error("Bad scale product accepted");}catch(const std::invalid_argument&){++badConstructor;}
        }
        PluginTensorDesc inputs[2]={desc(128,5120,DataType::kBF16),desc(6144,5120,DataType::kINT8)};auto output=desc(128,6144,DataType::kBF16);
        require(plugin.getWorkspaceSize(inputs,2,&output,1)==4849664,"Wrong Z workspace");
        inputs[1]=desc(6144,1280,DataType::kINT32);require(plugin.getWorkspaceSize(inputs,2,&output,1)==4849664,"Opaque INT32 rejected");inputs[1]=desc(6144,5120,DataType::kINT8);
        for(int t:{16,32}){inputs[0].dims.d[0]=output.dims.d[0]=t;require(plugin.getWorkspaceSize(inputs,2,&output,1)==0,"Z128 accepted old token shape");++badDescriptor;}
        inputs[0].dims.d[0]=output.dims.d[0]=128;
        for(int part=0;part<3;++part)for(int field=0;field<5;++field){auto* selected=part<2?&inputs[part]:&output;auto before=*selected;
            if(field<2)++selected->dims.d[field];else if(field==2)selected->dims.nbDims=3;else if(field==3)selected->type=DataType::kFLOAT;else selected->format=TensorFormat::kCHW16;
            require(plugin.getWorkspaceSize(inputs,2,&output,1)==0,"Bad descriptor accepted");++badDescriptor;*selected=before;
        }
        for(int count:{1,3}){require(plugin.getWorkspaceSize(inputs,count,&output,1)==0,"Bad input count");++badDescriptor;}
        for(int count:{0,2}){require(plugin.getWorkspaceSize(inputs,2,&output,count)==0,"Bad output count");++badDescriptor;}
        require(plugin.getWorkspaceSize(nullptr,2,&output,1)==0&&plugin.getWorkspaceSize(inputs,2,nullptr,1)==0,"Null descriptor");badDescriptor+=2;
        PluginTensorDesc all[3]={inputs[0],inputs[1],output};
        for(int pos=0;pos<3;++pos){require(plugin.supportsFormatCombination(pos,all,2,1),"Good format rejected");auto before=all[pos];all[pos].type=DataType::kHALF;require(!plugin.supportsFormatCombination(pos,all,2,1),"Half accepted");++badFormat;all[pos]=before;all[pos].format=TensorFormat::kCHW16;require(!plugin.supportsFormatCombination(pos,all,2,1),"Nonlinear accepted");++badFormat;all[pos]=before;}
        require(!plugin.supportsFormatCombination(-1,all,2,1)&&!plugin.supportsFormatCombination(3,all,2,1)&&!plugin.supportsFormatCombination(0,all,1,1)&&!plugin.supportsFormatCombination(0,all,2,0)&&!plugin.supportsFormatCombination(0,nullptr,2,1),"Bad format envelope");badFormat+=5;
        for(const char* value:{"true","01"," 1","1 ","2","-1"}){try{ModelOptFP8Z128GDNZSelected(value,1,128,128,6144,5120,true);throw std::logic_error("Invalid policy accepted");}catch(const std::invalid_argument&){++policyRejections;}}
        for(const char* value:{static_cast<const char*>(nullptr),"","0"})require(!ModelOptFP8Z128GDNZSelected(value,1,128,128,6144,5120,true),"Default/off changed");
        require(ModelOptFP8Z128GDNZSelected("1",1,128,128,6144,5120,true),"Study not selected");
        require(!ModelOptFP8Z128GDNZSelected("1",4,32,128,6144,5120,true)&&!ModelOptFP8Z128GDNZSelected("1",8,16,128,6144,5120,true)&&!ModelOptFP8Z128GDNZSelected("1",1,32,32,6144,5120,true)&&!ModelOptFP8Z128GDNZSelected("1",1,128,128,6144,5120,false),"Foreign profile selected");
        for(auto pair:{std::pair<int,int>{5120,6144},{6144,6144}}){try{ModelOptFP8Z128GDNZSelected("1",1,128,128,pair.first,pair.second,true);throw std::logic_error("Foreign Z geometry selected");}catch(const std::invalid_argument&){++policyRejections;}}
        require(badWire==34&&badConstructor==18&&badDescriptor==23&&badFormat==11&&policyRejections==8&&crossWire==5&&FP8Z128CPUUnexpectedCalls()==0,"Accounting/GPU stub changed");
        printf("{\"status\":\"fp8_z128_plugin_cpu_contract_passed\",\"wire_bytes\":20,\"wire_policy\":%u,\"bad_wires\":%d,\"bad_constructors\":%d,\"bad_descriptors\":%d,\"bad_formats\":%d,\"policy_rejections\":%d,\"cross_wire_rejections\":%d,\"workspace_bytes\":4849664,\"unexpected_gpu_calls\":%d,\"runtime_version\":%d,\"header_version\":%d,\"loaded_tensorrt_dll\":\"%s\",\"enqueue_tested\":false,\"model_qualified\":false,\"performance_admitted\":false}\n",wire[4],badWire,badConstructor,badDescriptor,badFormat,policyRejections,crossWire,FP8Z128CPUUnexpectedCalls(),getInferLibVersion(),NV_TENSORRT_VERSION,loaded);return 0;
    }catch(const std::exception& error){fprintf(stderr,"%s\n",error.what());return 1;}
}
