// SPDX-License-Identifier: Apache-2.0
#include "modelopt_fp8_z128_canonical.h"
#include "modelopt_fp8_z128_init_trace.h"
#include <cuda_runtime_api.h>
#include <cmath>
#include <cstring>
#include <mutex>
#include <unordered_map>
namespace Garnet {
struct ModelOptFP8Z128Context {
    int device=-1;
    cublasLtHandle_t handle=nullptr;
    cublasLtMatmulDesc_t operation=nullptr;
    cublasLtMatrixLayout_t weight=nullptr,input=nullptr,output=nullptr;
    ModelOptFP8Z128Info info;
    std::mutex mutex;
    ~ModelOptFP8Z128Context(){
        int previous=-1;cudaGetDevice(&previous);if(device>=0)cudaSetDevice(device);
        if(output)cublasLtMatrixLayoutDestroy(output);
        if(input)cublasLtMatrixLayoutDestroy(input);
        if(weight)cublasLtMatrixLayoutDestroy(weight);
        if(operation)cublasLtMatmulDescDestroy(operation);
        if(handle)cublasLtDestroy(handle);
        if(previous>=0)cudaSetDevice(previous);
    }
};
namespace {
std::mutex contextsMutex;
std::unordered_map<int,std::weak_ptr<ModelOptFP8Z128Context>> contexts;
struct CanonicalSelection {
    cublasLtMatrixLayout_t input=nullptr,output=nullptr;
    cublasLtMatmulPreference_t preference=nullptr;
    ~CanonicalSelection(){
        if(preference)cublasLtMatmulPreferenceDestroy(preference);
        if(output)cublasLtMatrixLayoutDestroy(output);
        if(input)cublasLtMatrixLayoutDestroy(input);
    }
};
}
std::shared_ptr<ModelOptFP8Z128Context> CreateModelOptFP8Z128Context(){
    ModelOptFP8Z128InitTrace trace;
#if CUDART_VERSION >= 12080
    int device=-1;cudaDeviceProp properties{};
    if(trace.api("cudaGetDevice",[&](){return cudaGetDevice(&device);})!=cudaSuccess||trace.api("cudaGetDeviceProperties",[&](){return cudaGetDeviceProperties(&properties,device);})!=cudaSuccess||
       properties.major!=12||properties.minor!=0||trace.api("cublasLtGetVersion",[&](){return cublasLtGetVersion();})!=130400)return trace.empty<ModelOptFP8Z128Context>("device_or_version",device,properties.major,properties.minor);
    std::lock_guard<std::mutex> lock(contextsMutex);
    if(auto previous=contexts[device].lock()){trace.success(true);return previous;}
    auto context=std::make_shared<ModelOptFP8Z128Context>();context->device=device;
    const auto okay=[](cublasStatus_t status){return status==CUBLAS_STATUS_SUCCESS;};
    if(!okay(trace.api("cublasLtCreate",[&](){return cublasLtCreate(&context->handle);}))||
       !okay(trace.api("cublasLtMatmulDescCreate",[&](){return cublasLtMatmulDescCreate(&context->operation,CUBLAS_COMPUTE_32F,CUDA_R_32F);})))return trace.empty<ModelOptFP8Z128Context>("handle_or_operation");
    const cublasOperation_t transA=CUBLAS_OP_T,transB=CUBLAS_OP_N;
    if(!okay(trace.api("cublasLtMatmulDescSetAttribute",[&](){return cublasLtMatmulDescSetAttribute(context->operation,CUBLASLT_MATMUL_DESC_TRANSA,&transA,sizeof(transA));}))||
       !okay(trace.api("cublasLtMatmulDescSetAttribute",[&](){return cublasLtMatmulDescSetAttribute(context->operation,CUBLASLT_MATMUL_DESC_TRANSB,&transB,sizeof(transB));}))||
       !okay(trace.api("cublasLtMatrixLayoutCreate",[&](){return cublasLtMatrixLayoutCreate(&context->weight,CUDA_R_8F_E4M3,5120,6144,5120);}))||
       !okay(trace.api("cublasLtMatrixLayoutCreate",[&](){return cublasLtMatrixLayoutCreate(&context->input,CUDA_R_8F_E4M3,5120,128,5120);}))||
       !okay(trace.api("cublasLtMatrixLayoutCreate",[&](){return cublasLtMatrixLayoutCreate(&context->output,CUDA_R_16BF,6144,128,6144);})))return trace.empty<ModelOptFP8Z128Context>("operation_or_actual_layout");
    CanonicalSelection canonical;
    if(!okay(trace.api("cublasLtMatrixLayoutCreate",[&](){return cublasLtMatrixLayoutCreate(&canonical.input,CUDA_R_8F_E4M3,5120,32,5120);}))||
       !okay(trace.api("cublasLtMatrixLayoutCreate",[&](){return cublasLtMatrixLayoutCreate(&canonical.output,CUDA_R_16BF,6144,32,6144);}))||
       !okay(trace.api("cublasLtMatmulPreferenceCreate",[&](){return cublasLtMatmulPreferenceCreate(&canonical.preference);})))return trace.empty<ModelOptFP8Z128Context>("canonical_layout_or_preference");
    // Preserve actual M32 preference, then AlgoCheck actual M128 descriptors.
    const size_t canonicalWorkspace=4ULL<<20;
    if(!okay(trace.api("cublasLtMatmulPreferenceSetAttribute",[&](){return cublasLtMatmulPreferenceSetAttribute(canonical.preference,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&canonicalWorkspace,sizeof(canonicalWorkspace));})))return trace.empty<ModelOptFP8Z128Context>("workspace_preference");
    cublasLtMatmulHeuristicResult_t choices[8]{};int found=0;
    if(!okay(trace.api("cublasLtMatmulAlgoGetHeuristic",[&](){return cublasLtMatmulAlgoGetHeuristic(context->handle,context->operation,context->weight,canonical.input,
        canonical.output,canonical.output,canonical.preference,8,choices,&found);})))return trace.empty<ModelOptFP8Z128Context>("heuristic_api");
    trace.note("heuristic","found",found);
    if(found>=0&&found<=8)for(int j=0;j<found;++j)trace.note("candidate","heuristic",j,int(choices[j].state),static_cast<long long>(choices[j].workspaceSize),&choices[j].algo,sizeof(choices[j].algo));
    // Exact opaque configuration from closed real128 Z control on pinned13.4.
    // Opaque bytes are never interpreted; any library/device selection drift fails closed.
    static_assert(sizeof(cublasLtMatmulAlgo_t)==64,"Pinned opaque ABI");
    const unsigned char expected[64]={0x43,0x00,0x00,0x00,0x0d,0x00,0x00,0x00,0x11,0x00,0x00,0x00,0x01,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x01,0x00,0x00,0x00,0x12,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x01,0x00,0x00,0x00,0x60,0xfd,0x01,0x00,0x1c,0x00,0x00,0x00,0x1c,0x00,0x00,0x00,0x0e,0x00,0x0e,0x00,0x44,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00};
    for(int i=0;i<found;++i){
        if(choices[i].state!=CUBLAS_STATUS_SUCCESS)continue;
        if(std::memcmp(&choices[i].algo,expected,sizeof(expected)))return trace.empty<ModelOptFP8Z128Context>("opaque_mismatch");
        trace.note("selection","opaque_equal",i);
        int32_t id=-1;size_t bytes=0;
        if(!okay(trace.api("cublasLtMatmulAlgoConfigGetAttribute",[&](){return cublasLtMatmulAlgoConfigGetAttribute(&choices[i].algo,CUBLASLT_ALGO_CONFIG_ID,&id,sizeof(id),&bytes);}))||bytes!=sizeof(id)||id!=67)return trace.empty<ModelOptFP8Z128Context>("algorithm_ID",id,static_cast<long long>(bytes));
        cublasLtMatmulHeuristicResult_t checked{};
        if(!okay(trace.api("cublasLtMatmulAlgoCheck",[&](){return cublasLtMatmulAlgoCheck(context->handle,context->operation,context->weight,context->input,
            context->output,context->output,&choices[i].algo,&checked);}))||checked.state!=CUBLAS_STATUS_SUCCESS||checked.workspaceSize>ModelOptFP8Z128BlasWorkspace)return trace.empty<ModelOptFP8Z128Context>("actual_algorithm_check",int(checked.state),static_cast<long long>(checked.workspaceSize));
        context->info.device=device;context->info.algorithm=id;context->info.requiredWorkspace=checked.workspaceSize;context->info.opaque=choices[i].algo;
        contexts[device]=context;trace.success(false);return context;
    }
#endif
    return trace.empty<ModelOptFP8Z128Context>("no_supported_candidate_or_CUDART");
}
bool GetModelOptFP8Z128Info(const std::shared_ptr<ModelOptFP8Z128Context>& context,ModelOptFP8Z128Info& info) noexcept{
    if(!context)return false;
    try{std::lock_guard<std::mutex> lock(context->mutex);info=context->info;return info.algorithm==67;}catch(...){return false;}
}
cublasStatus_t RunModelOptFP8Z128Matmul(const std::shared_ptr<ModelOptFP8Z128Context>& context,
    const void* weights,const void* input,void* output,int batch,int rows,int columns,float globalScale,
    void* workspace,size_t workspaceBytes,cudaStream_t stream) noexcept{
    if(!context)return CUBLAS_STATUS_NOT_INITIALIZED;
    if(!weights||!input||!output||!workspace||batch!=128||rows!=6144||columns!=5120||
       workspaceBytes!=ModelOptFP8Z128BlasWorkspace||!std::isfinite(globalScale)||globalScale<=0)return CUBLAS_STATUS_INVALID_VALUE;
    try{
        int device=-1;if(cudaGetDevice(&device)!=cudaSuccess||device!=context->device)return CUBLAS_STATUS_INVALID_VALUE;
        std::lock_guard<std::mutex> lock(context->mutex);const float beta=0;
        return cublasLtMatmul(context->handle,context->operation,&globalScale,weights,context->weight,input,context->input,
            &beta,output,context->output,output,context->output,&context->info.opaque,workspace,workspaceBytes,stream);
    }catch(...){return CUBLAS_STATUS_INTERNAL_ERROR;}
}
}
