// Observe unchanged actual typed helper selection. No model data or matmul calls.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <cuda_runtime_api.h>
#include "modelopt_fp8_z128_canonical.h"
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>
namespace {
struct HeuristicCall {int status=0,requested=0,found=0;std::vector<cublasLtMatmulHeuristicResult_t> values;};
struct ConfigCall {int status=0,attribute=0;size_t requested=0,written=0;std::string value;};
struct CheckCall {int status=0,state=0;size_t workspace=0;};
std::mutex auditMutex;
std::vector<HeuristicCall> heuristicCalls;std::vector<ConfigCall> configCalls;std::vector<CheckCall> checkCalls;
int matmulCalls=0;
std::string Hex(const void* value,size_t bytes){
    std::ostringstream out;out<<std::hex<<std::setfill('0');
    for(size_t i=0;i<bytes;++i)out<<std::setw(2)<<unsigned(static_cast<const unsigned char*>(value)[i]);return out.str();
}
cublasStatus_t CUBLASWINAPI AuditHeuristic(cublasLtHandle_t h,cublasLtMatmulDesc_t op,
    cublasLtMatrixLayout_t a,cublasLtMatrixLayout_t b,cublasLtMatrixLayout_t c,cublasLtMatrixLayout_t d,
    cublasLtMatmulPreference_t pref,int requested,cublasLtMatmulHeuristicResult_t values[],int* found){
    const auto status=::cublasLtMatmulAlgoGetHeuristic(h,op,a,b,c,d,pref,requested,values,found);
    HeuristicCall row;row.status=int(status);row.requested=requested;row.found=found?*found:-1;
    if(status==CUBLAS_STATUS_SUCCESS&&values&&row.found>=0&&row.found<=requested)row.values.assign(values,values+row.found);
    heuristicCalls.push_back(row);return status;
}
cublasStatus_t CUBLASWINAPI AuditConfig(const cublasLtMatmulAlgo_t* algo,cublasLtMatmulAlgoConfigAttributes_t attr,void* value,size_t bytes,size_t* written){
    const auto status=::cublasLtMatmulAlgoConfigGetAttribute(algo,attr,value,bytes,written);
    ConfigCall row;row.status=int(status);row.attribute=int(attr);row.requested=bytes;row.written=written?*written:0;
    if(status==CUBLAS_STATUS_SUCCESS&&value&&row.written<=bytes)row.value=Hex(value,row.written);
    configCalls.push_back(row);return status;
}
cublasStatus_t CUBLASWINAPI AuditCheck(cublasLtHandle_t h,cublasLtMatmulDesc_t op,
    cublasLtMatrixLayout_t a,cublasLtMatrixLayout_t b,cublasLtMatrixLayout_t c,cublasLtMatrixLayout_t d,
    const cublasLtMatmulAlgo_t* algo,cublasLtMatmulHeuristicResult_t* result){
    const auto status=::cublasLtMatmulAlgoCheck(h,op,a,b,c,d,algo,result);
    CheckCall row;row.status=int(status);
    if(status==CUBLAS_STATUS_SUCCESS&&result){row.state=int(result->state);row.workspace=result->workspaceSize;}
    checkCalls.push_back(row);return status;
}
cublasStatus_t CUBLASWINAPI AuditMatmul(cublasLtHandle_t h,cublasLtMatmulDesc_t op,const void* alpha,const void* a,
    cublasLtMatrixLayout_t ad,const void* b,cublasLtMatrixLayout_t bd,const void* beta,const void* c,
    cublasLtMatrixLayout_t cd,void* d,cublasLtMatrixLayout_t dd,const cublasLtMatmulAlgo_t* algo,
    void* workspace,size_t bytes,cudaStream_t stream){
    ++matmulCalls;return ::cublasLtMatmul(h,op,alpha,a,ad,b,bd,beta,c,cd,d,dd,algo,workspace,bytes,stream);
}
}
#define cublasLtMatmulAlgoGetHeuristic AuditHeuristic
#define cublasLtMatmulAlgoConfigGetAttribute AuditConfig
#define cublasLtMatmulAlgoCheck AuditCheck
#define cublasLtMatmul AuditMatmul
#include "modelopt_fp8_z128_canonical.cpp"
#undef cublasLtMatmul
#undef cublasLtMatmulAlgoCheck
#undef cublasLtMatmulAlgoConfigGetAttribute
#undef cublasLtMatmulAlgoGetHeuristic

extern "C" __declspec(dllexport) int __cdecl AuditZ128Context(const char* path) noexcept{
    try{
        std::lock_guard<std::mutex> lock(auditMutex);if(!path||!*path)return 10;
        heuristicCalls.clear();configCalls.clear();checkCalls.clear();matmulCalls=0;
        const auto before=cudaPeekAtLastError();int device=-1;const auto deviceStatus=cudaGetDevice(&device);
        auto context=Garnet::CreateModelOptFP8Z128Context();Garnet::ModelOptFP8Z128Info info;
        const bool infoOkay=Garnet::GetModelOptFP8Z128Info(context,info);
        const auto after=cudaPeekAtLastError();char module[MAX_PATH]{};
        const auto moduleBytes=GetModuleFileNameA(GetModuleHandleA("cublasLt64_13.dll"),module,MAX_PATH);
        // All extra ID queries and report I/O happen after actual helper selection returns.
        std::ostringstream out;out<<"{\"status\":\"actual_Z128_context_observed\",\"created\":"<<(context?"true":"false")
            <<",\"info_okay\":"<<(infoOkay?"true":"false")<<",\"device\":"<<device<<",\"device_status\":"<<int(deviceStatus)
            <<",\"cuda_error_before\":"<<int(before)<<",\"cuda_error_after\":"<<int(after)<<",\"matmul_calls\":"<<matmulCalls
            <<",\"helper_info\":{\"algorithm\":"<<info.algorithm<<",\"device\":"<<info.device<<",\"required_workspace\":"<<info.requiredWorkspace
            <<",\"opaque\":\""<<Hex(&info.opaque,sizeof(info.opaque))<<"\"},\"cublaslt_module_path_hex\":\""<<Hex(module,moduleBytes)<<"\",\"heuristics\":[";
        for(size_t i=0;i<heuristicCalls.size();++i){
            if(i)out<<',';const auto& row=heuristicCalls[i];out<<"{\"api_status\":"<<row.status<<",\"requested\":"<<row.requested<<",\"found\":"<<row.found<<",\"candidates\":[";
            for(size_t j=0;j<row.values.size();++j){
                if(j)out<<',';const auto& candidate=row.values[j];int32_t id=-1;size_t written=0;
                int idStatus=-1;if(candidate.state==CUBLAS_STATUS_SUCCESS)idStatus=int(::cublasLtMatmulAlgoConfigGetAttribute(&candidate.algo,CUBLASLT_ALGO_CONFIG_ID,&id,sizeof(id),&written));
                out<<"{\"state\":"<<int(candidate.state)<<",\"workspace\":"<<candidate.workspaceSize<<",\"opaque\":\""<<Hex(&candidate.algo,sizeof(candidate.algo))
                    <<"\",\"post_context_ID_status\":"<<idStatus<<",\"post_context_ID\":"<<id<<",\"post_context_ID_bytes\":"<<written<<'}';
            }out<<"]}";
        }out<<"],\"config_calls\":[";
        for(size_t i=0;i<configCalls.size();++i){if(i)out<<',';const auto& row=configCalls[i];out<<"{\"api_status\":"<<row.status<<",\"attribute\":"<<row.attribute<<",\"requested\":"<<row.requested<<",\"written\":"<<row.written<<",\"value_hex\":\""<<row.value<<"\"}";}
        out<<"],\"check_calls\":[";
        for(size_t i=0;i<checkCalls.size();++i){if(i)out<<',';const auto& row=checkCalls[i];out<<"{\"api_status\":"<<row.status<<",\"state\":"<<row.state<<",\"workspace\":"<<row.workspace<<'}';}
        out<<"],\"algorithm_substituted\":false,\"gpu_model_executed\":false,\"model_or_performance_qualified\":false}";
        context.reset();const std::string raw=out.str();
        const auto file=CreateFileA(path,GENERIC_WRITE,0,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);if(file==INVALID_HANDLE_VALUE)return 11;
        DWORD written=0;const BOOL okay=WriteFile(file,raw.data(),DWORD(raw.size()),&written,nullptr);const BOOL closed=CloseHandle(file);
        return okay&&closed&&written==raw.size()?0:12;
    }catch(...){return 13;}
}
