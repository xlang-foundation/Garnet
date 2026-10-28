// Compile against CPU interface fixtures, once with original helper and once with derivative.
// Every SDK call is a CPU stub; no CUDA DLL or Garnet DLL is linked or loaded.
#include "modelopt_fp8_z128_canonical.h"
#include <cuda_runtime_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <map>
namespace {
std::string mode;
int calls=0,failAt=0,throwAt=0,matmuls=0;
bool reconstruct=false;uint32_t config[9]{};std::map<void*,uint64_t> layoutColumns;
std::vector<std::string> transcript;
const char* expectedHex="430000000d0000001100000001000000000000000100000012000000000000000100000060fd01001c0000001c0000000e000e00440000000000000000000000";
bool step(const std::string& name){
    transcript.push_back(name);++calls;
    if(calls==throwAt)throw std::runtime_error("intentional_CPU_SDK_fault");
    return calls==failAt;
}
void expected(cublasLtMatmulAlgo_t& algo){
    for(size_t i=0;i<64;++i){char raw[3]={expectedHex[2*i],expectedHex[2*i+1],0};algo.bytes[i]=static_cast<unsigned char>(std::strtoul(raw,nullptr,16));}
}
void* handle(){return reinterpret_cast<void*>(size_t(calls+100));}
}
cudaError_t cudaGetDevice(int* value){if(step("cudaGetDevice"))return 7;*value=0;return 0;}
cudaError_t cudaGetDeviceProperties(cudaDeviceProp* value,int device){if(step("cudaGetDeviceProperties:"+std::to_string(device)))return 7;value->major=mode=="major"?9:12;value->minor=mode=="minor"?1:0;return 0;}
cudaError_t cudaSetDevice(int value){step("cudaSetDevice:"+std::to_string(value));return 0;}
size_t cublasLtGetVersion(){return step("cublasLtGetVersion")||mode=="version"?0:130400;}
cublasStatus_t cublasLtCreate(cublasLtHandle_t* value){if(step("cublasLtCreate"))return 7;*value=handle();return 0;}
cublasStatus_t cublasLtDestroy(cublasLtHandle_t){step("cublasLtDestroy");return 0;}
cublasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t* value,int compute,int scale){if(step("cublasLtMatmulDescCreate:"+std::to_string(compute)+":"+std::to_string(scale)))return 7;*value=handle();return 0;}
cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t){step("cublasLtMatmulDescDestroy");return 0;}
cublasStatus_t cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t,int attr,const void* value,size_t bytes){return step("cublasLtMatmulDescSetAttribute:"+std::to_string(attr)+":"+std::to_string(bytes)+":"+std::to_string(*static_cast<const int*>(value)))?7:0;}
cublasStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t* value,int type,uint64_t rows,uint64_t columns,int64_t ld){if(step("cublasLtMatrixLayoutCreate:"+std::to_string(type)+":"+std::to_string(rows)+":"+std::to_string(columns)+":"+std::to_string(ld)))return 7;*value=handle();layoutColumns[*value]=columns;return 0;}
cublasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t){step("cublasLtMatrixLayoutDestroy");return 0;}
cublasStatus_t cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t* value){if(step("cublasLtMatmulPreferenceCreate"))return 7;*value=handle();return 0;}
cublasStatus_t cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t){step("cublasLtMatmulPreferenceDestroy");return 0;}
cublasStatus_t cublasLtMatmulPreferenceSetAttribute(cublasLtMatmulPreference_t,int attr,const void* value,size_t bytes){return step("cublasLtMatmulPreferenceSetAttribute:"+std::to_string(attr)+":"+std::to_string(bytes)+":"+std::to_string(*static_cast<const size_t*>(value)))?7:0;}
cublasStatus_t cublasLtMatmulAlgoGetHeuristic(cublasLtHandle_t,cublasLtMatmulDesc_t,cublasLtMatrixLayout_t,cublasLtMatrixLayout_t,cublasLtMatrixLayout_t,cublasLtMatrixLayout_t,cublasLtMatmulPreference_t,int requested,cublasLtMatmulHeuristicResult_t* values,int* found){
    if(step("cublasLtMatmulAlgoGetHeuristic:"+std::to_string(requested)))return 7;
    *found=mode=="found0"?0:((mode=="skip_failed"||mode=="mismatch_then_expected")?2:1);
    for(int i=0;i<*found;++i){values[i].state=0;expected(values[i].algo);}
    if(mode=="skip_failed")values[0].state=7;
    if(mode=="all_failed")values[0].state=7;
    if(mode=="opaque"||mode=="mismatch_then_expected")values[0].algo.bytes[63]^=1;
    return 0;
}
cublasStatus_t cublasLtMatmulAlgoConfigGetAttribute(const cublasLtMatmulAlgo_t*,int attr,void* value,size_t bytes,size_t* written){
    if(step("cublasLtMatmulAlgoConfigGetAttribute:"+std::to_string(attr)+":"+std::to_string(bytes)))return 7;
    if(reconstruct){
        if(attr<0||attr>=9||bytes!=(attr>=7?2:4))throw std::runtime_error("bad_CPU_attribute_size");
        uint32_t result=config[attr];if(mode=="get_value_"+std::to_string(attr))++result;
        if(bytes==2)*static_cast<uint16_t*>(value)=static_cast<uint16_t>(result);else *static_cast<uint32_t*>(value)=result;
        *written=mode=="get_bytes_"+std::to_string(attr)?bytes-1:bytes;return 0;
    }
    *static_cast<int32_t*>(value)=mode=="id"?35:67;*written=mode=="id_bytes"?3:sizeof(int32_t);return 0;
}
cublasStatus_t cublasLtMatmulAlgoCheck(cublasLtHandle_t,cublasLtMatmulDesc_t,cublasLtMatrixLayout_t,cublasLtMatrixLayout_t input,cublasLtMatrixLayout_t,cublasLtMatrixLayout_t,const cublasLtMatmulAlgo_t*,cublasLtMatmulHeuristicResult_t* value){
    if(step("cublasLtMatmulAlgoCheck"))return 7;const auto suffix=std::to_string(layoutColumns[input]);
    value->state=(mode=="check_state"||mode=="check_state_"+suffix)?7:0;
    value->workspaceSize=(mode=="workspace"||mode=="workspace_"+suffix)?(4ULL<<20)+1:0;return 0;
}
cublasStatus_t cublasLtMatmulAlgoInit(cublasLtHandle_t,int compute,int scale,int a,int b,int c,int d,int id,cublasLtMatmulAlgo_t* algo){
    if(step("cublasLtMatmulAlgoInit:"+std::to_string(compute)+":"+std::to_string(scale)+":"+std::to_string(a)+":"+std::to_string(b)+":"+std::to_string(c)+":"+std::to_string(d)+":"+std::to_string(id)))return 7;
    std::memset(algo,0,sizeof(*algo));for(auto& field:config)field=0;config[0]=id;return 0;
}
cublasStatus_t cublasLtMatmulAlgoConfigSetAttribute(cublasLtMatmulAlgo_t*,int attr,const void* value,size_t bytes){
    if(attr<=0||attr>=9||bytes!=(attr>=7?2:4))throw std::runtime_error("bad_CPU_set_attribute_size");
    const uint32_t field=bytes==2?*static_cast<const uint16_t*>(value):*static_cast<const uint32_t*>(value);
    if(step("cublasLtMatmulAlgoConfigSetAttribute:"+std::to_string(attr)+":"+std::to_string(bytes)+":"+std::to_string(field)))return 7;
    config[attr]=field;return 0;
}
cublasStatus_t cublasLtMatmul(cublasLtHandle_t,cublasLtMatmulDesc_t,const void*,const void*,cublasLtMatrixLayout_t,const void*,cublasLtMatrixLayout_t,const void*,const void*,cublasLtMatrixLayout_t,void*,cublasLtMatrixLayout_t,const cublasLtMatmulAlgo_t*,void*,size_t,cudaStream_t){++matmuls;return 14;}
int main(int argc,char** argv){
    if(argc!=2&&argc!=3)return 2;mode=argv[1];reconstruct=argc==3&&std::strcmp(argv[2],"reconstruct")==0;
    if(argc==3&&!reconstruct)return 2;
    if(mode.rfind("fail_",0)==0)failAt=std::atoi(mode.c_str()+5);
    if(mode.rfind("throw_",0)==0)throwAt=std::atoi(mode.c_str()+6);
    bool created=false,cached=false,infoOkay=false,thrown=false;Garnet::ModelOptFP8Z128Info info;
    try{
        auto context=(reconstruct?Garnet::CreateModelOptFP8Z128ReconstructedContext():Garnet::CreateModelOptFP8Z128Context());created=bool(context);
        infoOkay=Garnet::GetModelOptFP8Z128Info(context,info);
        if(mode=="cached"){auto other=(reconstruct?Garnet::CreateModelOptFP8Z128ReconstructedContext():Garnet::CreateModelOptFP8Z128Context());cached=context&&other==context;}
    }catch(const std::runtime_error&){thrown=true;}
    std::printf("{\"mode\":\"%s\",\"created\":%s,\"cached\":%s,\"info_okay\":%s,\"thrown\":%s,\"algorithm\":%d,\"workspace\":%zu,\"unexpected_matmuls\":%d,\"calls\":[",mode.c_str(),created?"true":"false",cached?"true":"false",infoOkay?"true":"false",thrown?"true":"false",info.algorithm,info.requiredWorkspace,matmuls);
    for(size_t i=0;i<transcript.size();++i)std::printf("%s\"%s\"",i?",":"",transcript[i].c_str());
    std::puts("]}");return matmuls?3:0;
}
