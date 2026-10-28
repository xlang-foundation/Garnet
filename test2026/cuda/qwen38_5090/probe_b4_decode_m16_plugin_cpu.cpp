// Actual TRT plugin contract with CPU-only factories/operators; no GPU/model.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include "modelopt_fp8_gdn_m16_plugin.h"
#include "modelopt_fp8_plugin.h"
#include <NvInfer.h>
#include <NvInferVersion.h>
#include <array>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
namespace Garnet {void GDNM16CPUFactoryMode(const char*);int GDNM16CPUFactoryCalls();int GDNM16CPUUnexpectedCalls();}
using namespace Garnet;using namespace nvinfer1;
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
PluginTensorDesc desc(int rows,int columns,DataType type){PluginTensorDesc d{};d.type=type;d.format=TensorFormat::kLINEAR;d.dims.nbDims=2;d.dims.d[0]=rows;d.dims.d[1]=columns;return d;}
int main(){try{
    require(getInferLibVersion()==NV_TENSORRT_VERSION,"TRT header/runtime differ");
    char loaded[MAX_PATH]{};auto length=GetModuleFileNameA(GetModuleHandleA("nvinfer_10.dll"),loaded,MAX_PATH);require(length&&length<MAX_PATH,"TRT loaded path unavailable");for(DWORD i=0;i<length;++i)if(loaded[i]=='\\')loaded[i]='/';
    require(EnsureModelOptFP8PluginRegistered()&&EnsureModelOptFP8GDNM16PluginRegistered(),"Creator registration failed");auto* registry=getPluginRegistry();require(registry,"No registry");auto* creator=registry->getPluginCreator("GarnetModelOptFP8GDNM16Canonical","1","");auto* old=registry->getPluginCreator("GarnetModelOptFP8","1","");require(creator&&old,"Creator missing");
    int wires=0,descriptors=0,formats=0,constructors=0,cross=0,initializations=0;
    for(int n:{10240,6144}){
        ModelOptFP8GDNM16Plugin p(n,5120,.5f,.25f);p.setPluginNamespace("gdn_m16_study");std::array<uint32_t,5> wire{},copy{};require(p.getSerializationSize()==20&&p.getNbOutputs()==1,"Wire/output size");p.serialize(wire.data());require(wire[0]==uint32_t(n)&&wire[1]==5120&&wire[4]==0x47313634u,"Wire fields");
        auto* clone=p.clone();require(clone,"No clone");clone->serialize(copy.data());require(copy==wire&&std::string(clone->getPluginType())=="GarnetModelOptFP8GDNM16Canonical"&&std::string(clone->getPluginNamespace())=="gdn_m16_study","Clone identity");clone->destroy();
        auto* restored=creator->deserializePlugin("good",wire.data(),20);require(restored,"Valid wire failed");restored->serialize(copy.data());require(copy==wire,"Restored wire changed");restored->destroy();
        auto reject=[&](const void* value,size_t bytes){auto* result=creator->deserializePlugin("bad",value,bytes);if(result){result->destroy();throw std::runtime_error("Malformed wire accepted");}++wires;};
        for(uint32_t bad:{0u,1u,16u,128u,5120u,12288u,0xffffffffu}){auto value=wire;value[0]=bad;reject(value.data(),20);}
        for(uint32_t bad:{0u,1u,16u,128u,6144u,12288u,0xffffffffu}){auto value=wire;value[1]=bad;reject(value.data(),20);}
        for(int field:{2,3})for(uint32_t bad:{0u,0x80000000u,0xbf800000u,0x7f800000u,0x7fc00000u}){auto value=wire;value[field]=bad;reject(value.data(),20);}
        for(uint32_t bad:{0u,1u,16u,128u,0x51313238u,0x47313635u,0xffffffffu}){auto value=wire;value[4]=bad;reject(value.data(),20);}
        for(size_t size:{size_t(0),size_t(1),size_t(15),size_t(16),size_t(19),size_t(21),size_t(24)})reject(wire.data(),size);reject(nullptr,20);
        auto* wrong=old->deserializePlugin("new",wire.data(),20);require(!wrong,"Legacy accepted new wire");++cross;
        ModelOptFP8Plugin legacy(5120,6144,.5f,.25f),canonical(5120,6144,.5f,.25f,true);
        for(auto* value:{&legacy,&canonical}){std::array<uint32_t,5> foreign{};value->serialize(foreign.data());reject(foreign.data(),value->getSerializationSize());++cross;}
        for(int badN:{0,-1,1,5120,12288,65536}){try{ModelOptFP8GDNM16Plugin bad(badN,5120,.5f,.25f);throw std::logic_error("Bad N accepted");}catch(const std::invalid_argument&){++constructors;}}
        for(int badK:{0,-1,1,5121,6144,65536}){try{ModelOptFP8GDNM16Plugin bad(n,badK,.5f,.25f);throw std::logic_error("Bad K accepted");}catch(const std::invalid_argument&){++constructors;}}
        for(float scale:{0.f,-1.f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()})for(int field:{0,1}){try{ModelOptFP8GDNM16Plugin bad(n,5120,field?1.f:scale,field?scale:1.f);throw std::logic_error("Bad scale accepted");}catch(const std::invalid_argument&){++constructors;}}
        for(float scale:{std::numeric_limits<float>::max(),std::numeric_limits<float>::min()}){try{ModelOptFP8GDNM16Plugin bad(n,5120,scale,scale);throw std::logic_error("Bad scale product accepted");}catch(const std::invalid_argument&){++constructors;}}
        PluginTensorDesc in[]={desc(16,5120,DataType::kBF16),desc(n,1280,DataType::kINT32)},out[]={desc(16,n,DataType::kBF16)};require(p.getWorkspaceSize(in,2,out,1)==(4ULL<<20)+16ULL*5120,"Wrong workspace");
        in[1]=desc(n,5120,DataType::kINT8);require(p.getWorkspaceSize(in,2,out,1)==(4ULL<<20)+16ULL*5120,"INT8 packing rejected");in[1]=desc(n,1280,DataType::kINT32);
        require(!p.getWorkspaceSize(nullptr,2,out,1)&&!p.getWorkspaceSize(in,1,out,1)&&!p.getWorkspaceSize(in,2,nullptr,1)&&!p.getWorkspaceSize(in,2,out,2),"Bad descriptor cardinality accepted");descriptors+=4;
        for(int tensor=0;tensor<3;++tensor)for(int dim=0;dim<2;++dim)for(int bad:{0,-1,1,4,32,128,4096}){PluginTensorDesc a[]={in[0],in[1]},b[]={out[0]};auto& d=tensor==2?b[0]:a[tensor];d.dims.d[dim]=bad;require(!p.getWorkspaceSize(a,2,b,1),"Bad descriptor geometry accepted");++descriptors;}
        for(int tensor=0;tensor<3;++tensor)for(DataType bad:{DataType::kFLOAT,DataType::kHALF,DataType::kINT64}){PluginTensorDesc a[]={in[0],in[1]},b[]={out[0]};(tensor==2?b[0]:a[tensor]).type=bad;require(!p.getWorkspaceSize(a,2,b,1),"Bad type accepted");++formats;}
        require(p.enqueue(in,out,nullptr,nullptr,nullptr,nullptr)==1,"Null enqueue accepted");
        for(const char* mode:{"null","std","unknown"}){GDNM16CPUFactoryMode(mode);require(p.initialize()==1,"Factory failure accepted");++initializations;p.terminate();}
        GDNM16CPUFactoryMode("good");const int before=GDNM16CPUFactoryCalls();require(p.initialize()==0&&p.initialize()==0,"Initialize/cache failed");initializations+=2;require(GDNM16CPUFactoryCalls()==before+1,"Repeated initialize recreated context");clone=p.clone();require(clone&&clone->initialize()==0,"Initialized clone lost context");++initializations;require(GDNM16CPUFactoryCalls()==before+1,"Clone recreated context");clone->destroy();p.terminate();require(p.initialize()==0&&GDNM16CPUFactoryCalls()==before+2,"Terminate/reinitialize failed");++initializations;p.terminate();
    }
    require(wires==82&&constructors==44&&descriptors==92&&formats==18&&cross==6&&initializations==14&&GDNM16CPUFactoryCalls()==10&&GDNM16CPUUnexpectedCalls()==0,"Contract count or unexpected operator");
    std::printf("{\"status\":\"B4T4_GDN_M16_plugin_wire_and_initialize_CPU_contract_passed\",\"wire_rejections\":%d,\"constructor_rejections\":%d,\"descriptor_rejections\":%d,\"format_rejections\":%d,\"cross_wire_rejections\":%d,\"initialize_calls\":%d,\"factory_calls\":%d,\"GPU_operator_calls\":%d,\"TRT_DLL\":\"%s\",\"TRT_version\":%d}\n",wires,constructors,descriptors,formats,cross,initializations,GDNM16CPUFactoryCalls(),GDNM16CPUUnexpectedCalls(),loaded,NV_TENSORRT_VERSION);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
