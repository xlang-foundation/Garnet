// SPDX-License-Identifier: Apache-2.0
#include "modelopt_fp8_z128_canonical.h"
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
#if CUDART_VERSION >= 12080
    int device=-1;cudaDeviceProp properties{};
    if(cudaGetDevice(&device)!=cudaSuccess||cudaGetDeviceProperties(&properties,device)!=cudaSuccess||
       properties.major!=12||properties.minor!=0||cublasLtGetVersion()!=130400)return {};
    std::lock_guard<std::mutex> lock(contextsMutex);
    if(auto previous=contexts[device].lock())return previous;
    auto context=std::make_shared<ModelOptFP8Z128Context>();context->device=device;
    const auto okay=[](cublasStatus_t status){return status==CUBLAS_STATUS_SUCCESS;};
    if(!okay(cublasLtCreate(&context->handle))||
       !okay(cublasLtMatmulDescCreate(&context->operation,CUBLAS_COMPUTE_32F,CUDA_R_32F)))return {};
    const cublasOperation_t transA=CUBLAS_OP_T,transB=CUBLAS_OP_N;
    if(!okay(cublasLtMatmulDescSetAttribute(context->operation,CUBLASLT_MATMUL_DESC_TRANSA,&transA,sizeof(transA)))||
       !okay(cublasLtMatmulDescSetAttribute(context->operation,CUBLASLT_MATMUL_DESC_TRANSB,&transB,sizeof(transB)))||
       !okay(cublasLtMatrixLayoutCreate(&context->weight,CUDA_R_8F_E4M3,5120,6144,5120))||
       !okay(cublasLtMatrixLayoutCreate(&context->input,CUDA_R_8F_E4M3,5120,128,5120))||
       !okay(cublasLtMatrixLayoutCreate(&context->output,CUDA_R_16BF,6144,128,6144)))return {};
    CanonicalSelection canonical;
    if(!okay(cublasLtMatrixLayoutCreate(&canonical.input,CUDA_R_8F_E4M3,5120,32,5120))||
       !okay(cublasLtMatrixLayoutCreate(&canonical.output,CUDA_R_16BF,6144,32,6144))||
       !okay(cublasLtMatmulPreferenceCreate(&canonical.preference)))return {};
    // Preserve actual M32 preference, then AlgoCheck actual M128 descriptors.
    const size_t canonicalWorkspace=4ULL<<20;
    if(!okay(cublasLtMatmulPreferenceSetAttribute(canonical.preference,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&canonicalWorkspace,sizeof(canonicalWorkspace))))return {};
    cublasLtMatmulHeuristicResult_t choices[8]{};int found=0;
    if(!okay(cublasLtMatmulAlgoGetHeuristic(context->handle,context->operation,context->weight,canonical.input,
        canonical.output,canonical.output,canonical.preference,8,choices,&found)))return {};
    // Exact opaque configuration from closed real128 Z control on pinned13.4.
    // Opaque bytes are never interpreted; any library/device selection drift fails closed.
    static_assert(sizeof(cublasLtMatmulAlgo_t)==64,"Pinned opaque ABI");
    const unsigned char expected[64]={0x43,0x00,0x00,0x00,0x0d,0x00,0x00,0x00,0x11,0x00,0x00,0x00,0x01,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x01,0x00,0x00,0x00,0x12,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x01,0x00,0x00,0x00,0x60,0xfd,0x01,0x00,0x1c,0x00,0x00,0x00,0x1c,0x00,0x00,0x00,0x0e,0x00,0x0e,0x00,0x44,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00};
    for(int i=0;i<found;++i){
        if(choices[i].state!=CUBLAS_STATUS_SUCCESS)continue;
        if(std::memcmp(&choices[i].algo,expected,sizeof(expected)))return {};
        int32_t id=-1;size_t bytes=0;
        if(!okay(cublasLtMatmulAlgoConfigGetAttribute(&choices[i].algo,CUBLASLT_ALGO_CONFIG_ID,&id,sizeof(id),&bytes))||bytes!=sizeof(id)||id!=67)return {};
        cublasLtMatmulHeuristicResult_t checked{};
        if(!okay(cublasLtMatmulAlgoCheck(context->handle,context->operation,context->weight,context->input,
            context->output,context->output,&choices[i].algo,&checked))||checked.state!=CUBLAS_STATUS_SUCCESS||checked.workspaceSize>ModelOptFP8Z128BlasWorkspace)return {};
        context->info.device=device;context->info.algorithm=id;context->info.requiredWorkspace=checked.workspaceSize;context->info.opaque=choices[i].algo;
        contexts[device]=context;return context;
    }
#endif
    return {};
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
