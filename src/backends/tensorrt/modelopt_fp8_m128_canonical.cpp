// SPDX-License-Identifier: Apache-2.0
#include "modelopt_fp8_m128_canonical.h"
#include <cuda_runtime_api.h>
#include <cmath>
#include <mutex>
#include <unordered_map>
namespace Garnet {
struct ModelOptFP8M128Context {
    int device=-1;
    cublasLtHandle_t handle=nullptr;
    cublasLtMatmulDesc_t operation=nullptr;
    cublasLtMatrixLayout_t weight=nullptr,input=nullptr,output=nullptr;
    ModelOptFP8M128Info info;
    std::mutex mutex;
    ~ModelOptFP8M128Context(){
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
std::unordered_map<int,std::weak_ptr<ModelOptFP8M128Context>> contexts;
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
std::shared_ptr<ModelOptFP8M128Context> CreateModelOptFP8M128Context(){
#if CUDART_VERSION >= 12080
    int device=-1;cudaDeviceProp properties{};
    if(cudaGetDevice(&device)!=cudaSuccess||cudaGetDeviceProperties(&properties,device)!=cudaSuccess||
       properties.major!=12||properties.minor!=0||cublasLtGetVersion()!=130400)return {};
    std::lock_guard<std::mutex> lock(contextsMutex);
    if(auto previous=contexts[device].lock())return previous;
    auto context=std::make_shared<ModelOptFP8M128Context>();context->device=device;
    const auto okay=[](cublasStatus_t status){return status==CUBLAS_STATUS_SUCCESS;};
    if(!okay(cublasLtCreate(&context->handle))||
       !okay(cublasLtMatmulDescCreate(&context->operation,CUBLAS_COMPUTE_32F,CUDA_R_32F)))return {};
    const cublasOperation_t transA=CUBLAS_OP_T,transB=CUBLAS_OP_N;
    if(!okay(cublasLtMatmulDescSetAttribute(context->operation,CUBLASLT_MATMUL_DESC_TRANSA,&transA,sizeof(transA)))||
       !okay(cublasLtMatmulDescSetAttribute(context->operation,CUBLASLT_MATMUL_DESC_TRANSB,&transB,sizeof(transB)))||
       !okay(cublasLtMatrixLayoutCreate(&context->weight,CUDA_R_8F_E4M3,6144,5120,6144))||
       !okay(cublasLtMatrixLayoutCreate(&context->input,CUDA_R_8F_E4M3,6144,128,6144))||
       !okay(cublasLtMatrixLayoutCreate(&context->output,CUDA_R_16BF,5120,128,5120)))return {};
    CanonicalSelection canonical;
    if(!okay(cublasLtMatrixLayoutCreate(&canonical.input,CUDA_R_8F_E4M3,6144,16,6144))||
       !okay(cublasLtMatrixLayoutCreate(&canonical.output,CUDA_R_16BF,5120,16,5120))||
       !okay(cublasLtMatmulPreferenceCreate(&canonical.preference)))return {};
    // Preserve the tested M16 selection preference, then validate M128 separately.
    const size_t canonicalWorkspace=4ULL<<20;
    if(!okay(cublasLtMatmulPreferenceSetAttribute(canonical.preference,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&canonicalWorkspace,sizeof(canonicalWorkspace))))return {};
    cublasLtMatmulHeuristicResult_t choices[8]{};int found=0;
    if(!okay(cublasLtMatmulAlgoGetHeuristic(context->handle,context->operation,context->weight,canonical.input,
        canonical.output,canonical.output,canonical.preference,8,choices,&found)))return {};
    for(int i=0;i<found;++i){
        if(choices[i].state!=CUBLAS_STATUS_SUCCESS)continue;
        int32_t id=-1;size_t bytes=0;
        if(!okay(cublasLtMatmulAlgoConfigGetAttribute(&choices[i].algo,CUBLASLT_ALGO_CONFIG_ID,&id,sizeof(id),&bytes))||bytes!=sizeof(id)||id!=35)return {};
        cublasLtMatmulHeuristicResult_t checked{};
        if(!okay(cublasLtMatmulAlgoCheck(context->handle,context->operation,context->weight,context->input,
            context->output,context->output,&choices[i].algo,&checked))||checked.state!=CUBLAS_STATUS_SUCCESS||checked.workspaceSize>ModelOptFP8M128BlasWorkspace)return {};
        context->info.device=device;context->info.algorithm=id;context->info.requiredWorkspace=checked.workspaceSize;context->info.opaque=choices[i].algo;
        contexts[device]=context;return context;
    }
#endif
    return {};
}
bool GetModelOptFP8M128Info(const std::shared_ptr<ModelOptFP8M128Context>& context,ModelOptFP8M128Info& info) noexcept{
    if(!context)return false;
    try{std::lock_guard<std::mutex> lock(context->mutex);info=context->info;return info.algorithm==35;}catch(...){return false;}
}
cublasStatus_t RunModelOptFP8M128Matmul(const std::shared_ptr<ModelOptFP8M128Context>& context,
    const void* weights,const void* input,void* output,int batch,int rows,int columns,float globalScale,
    void* workspace,size_t workspaceBytes,cudaStream_t stream) noexcept{
    if(!context)return CUBLAS_STATUS_NOT_INITIALIZED;
    if(!weights||!input||!output||!workspace||batch!=128||rows!=5120||columns!=6144||
       workspaceBytes!=ModelOptFP8M128BlasWorkspace||!std::isfinite(globalScale)||globalScale<=0)return CUBLAS_STATUS_INVALID_VALUE;
    try{
        int device=-1;if(cudaGetDevice(&device)!=cudaSuccess||device!=context->device)return CUBLAS_STATUS_INVALID_VALUE;
        std::lock_guard<std::mutex> lock(context->mutex);const float beta=0;
        return cublasLtMatmul(context->handle,context->operation,&globalScale,weights,context->weight,input,context->input,
            &beta,output,context->output,output,context->output,&context->info.opaque,workspace,workspaceBytes,stream);
    }catch(...){return CUBLAS_STATUS_INTERNAL_ERROR;}
}
}
