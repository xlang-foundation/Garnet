// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdlib>
#include <string>
#include "bf16_projection_lt_prefill.cuh"
#include "fp8_projection.cuh"
#include <memory>
#include <mutex>
#include <unordered_map>
namespace Garnet {
namespace {
struct ProjectionContext {
    int device=-1;cublasHandle_t handle=nullptr;std::mutex mutex;
    bool prefillLtRequested=false;std::unique_ptr<PrefillLtState> prefillLt;
#if CUDART_VERSION >= 12080
    cublasLtHandle_t fp8Handle=nullptr;
#endif
    ~ProjectionContext(){
        int previous=-1;cudaGetDevice(&previous);cudaSetDevice(device);
        prefillLt.reset();
#if CUDART_VERSION >= 12080
        if(fp8Handle)cublasLtDestroy(fp8Handle);
#endif
        if(handle)cublasDestroy(handle);if(previous>=0)cudaSetDevice(previous);
    }
};
std::mutex poolMutex;
std::unordered_map<int,std::weak_ptr<ProjectionContext>> contexts;
}
std::shared_ptr<void> CreateBF16ProjectionContext(){
    int device=-1;if(cudaGetDevice(&device)!=cudaSuccess)return {};
    std::lock_guard<std::mutex> lock(poolMutex);
    if(auto owner=contexts[device].lock())return owner;
    auto owner=std::make_shared<ProjectionContext>();owner->device=device;
    if(cublasCreate(&owner->handle)!=CUBLAS_STATUS_SUCCESS)return {};
    // Compute32F alone permits reductions in the lower output precision.
    // Keep the declared FP32 accumulation contract through split reductions.
    if(cublasSetMathMode(owner->handle,static_cast<cublasMath_t>(CUBLAS_DEFAULT_MATH |
        CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION))!=CUBLAS_STATUS_SUCCESS)return {};
    const char* prefillLt=std::getenv("GARNET_QWEN35_PREFILL_LT_DOWN");
    owner->prefillLtRequested=prefillLt&&std::string(prefillLt)=="1";
    contexts[device]=owner;return owner;
}
static int RunBF16ProjectionImpl(const std::shared_ptr<void>& opaque,const void* input,const void* weight,
    void* output,int rows,int inner,int columns,cudaStream_t stream,void* workspace,size_t workspaceBytes,bool floatOutput) noexcept {
    if(!opaque||!input||!weight||!output||rows<=0||inner<=0||columns<=0)return 1;
    auto* context=static_cast<ProjectionContext*>(opaque.get());
    std::lock_guard<std::mutex> lock(context->mutex);
    int device=-1;if(cudaGetDevice(&device)!=cudaSuccess||device!=context->device)return 1;
    if(cublasSetStream(context->handle,stream)!=CUBLAS_STATUS_SUCCESS)return 1;
    // A shared handle may enqueue onto different context streams. Do not let
    // those asynchronous GEMMs alias an implicit handle-wide scratch buffer.
    if(workspaceBytes&&!workspace)return 1;
    if(!floatOutput&&context->prefillLtRequested&&workspaceBytes==0&&inner==9216&&columns==2560&&(rows==256||rows==512)){
        if(!context->prefillLt){
            cudaStreamCaptureStatus capture;
            if(cudaStreamIsCapturing(stream,&capture)!=cudaSuccess||capture!=cudaStreamCaptureStatusNone)return 1;
            auto candidate=std::unique_ptr<PrefillLtState>(new(std::nothrow) PrefillLtState);
            if(!candidate||!candidate->initialize())return 1;
            context->prefillLt=std::move(candidate);
        }
        return context->prefillLt->run(input,weight,output,rows,stream);
    }
    if(cublasSetWorkspace(context->handle,workspaceBytes?workspace:nullptr,workspaceBytes)!=CUBLAS_STATUS_SUCCESS)return 1;
    const float alpha=1,beta=0;
    // Row-major X*W^T represented as column-major W*X^T. Explicit FP32
    // accumulation fixes the precision contract independently of TRT tactics.
    return cublasGemmEx(context->handle,CUBLAS_OP_T,CUBLAS_OP_N,columns,rows,inner,
        &alpha,weight,CUDA_R_16BF,inner,input,CUDA_R_16BF,inner,&beta,
        output,floatOutput?CUDA_R_32F:CUDA_R_16BF,columns,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP)==CUBLAS_STATUS_SUCCESS?0:1;
}
int RunBF16Projection(const std::shared_ptr<void>& context,const void* input,const void* weight,void* output,
    int rows,int inner,int columns,cudaStream_t stream,void* workspace,size_t bytes) noexcept {
    return RunBF16ProjectionImpl(context,input,weight,output,rows,inner,columns,stream,workspace,bytes,false);
}
int RunBF16ProjectionFloat32(const std::shared_ptr<void>& context,const void* input,const void* weight,void* output,
    int rows,int inner,int columns,cudaStream_t stream,void* workspace,size_t bytes) noexcept {
    return RunBF16ProjectionImpl(context,input,weight,output,rows,inner,columns,stream,workspace,bytes,true);
}
int RunStreamedFP8Projection(const std::shared_ptr<void>& opaque,const void* input,const void* weight,void* output,
    int rows,int inner,int columns,cudaStream_t stream,void* workspace,size_t bytes) noexcept {
#if CUDART_VERSION >= 12080
    if(!opaque||!input||!weight||!output||(bytes&&!workspace)||
        rows<16||rows>65536||rows%16||inner<16||inner>16384||inner%16||
        columns<16||columns>32768||columns%16)return 1;
    auto* context=static_cast<ProjectionContext*>(opaque.get());
    std::lock_guard<std::mutex> lock(context->mutex);
    int device=-1;
    if(cudaGetDevice(&device)!=cudaSuccess||device!=context->device)return 1;
    if(!context->fp8Handle){
        int major=0,minor=0;
        if(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device)!=cudaSuccess||
            cudaDeviceGetAttribute(&minor,cudaDevAttrComputeCapabilityMinor,device)!=cudaSuccess||
            major*10+minor<89||cublasLtCreate(&context->fp8Handle)!=CUBLAS_STATUS_SUCCESS)return 1;
    }
    return RunFP8Projection(context->fp8Handle,input,weight,output,rows,inner,columns,stream,workspace,bytes)?0:1;
#else
    return 1;
#endif
}
}
