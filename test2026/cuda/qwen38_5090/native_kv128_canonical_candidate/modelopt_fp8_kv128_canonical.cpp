// SPDX-License-Identifier: Apache-2.0
#include "modelopt_fp8_kv128_canonical.h"
#include "modelopt_fp8_kv128_init_trace.h"
#include <cuda_runtime_api.h>
#include <cmath>
#include <cstring>
#include <mutex>
#include <unordered_map>
namespace Garnet {
struct ModelOptFP8KV128Context {
    int device=-1;
    cublasLtHandle_t handle=nullptr;
    cublasLtMatmulDesc_t operation=nullptr;
    cublasLtMatrixLayout_t weight=nullptr,input=nullptr,output=nullptr;
    ModelOptFP8KV128Info info;
    std::mutex mutex;
    ~ModelOptFP8KV128Context(){
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
std::unordered_map<int,std::weak_ptr<ModelOptFP8KV128Context>> contexts;
template<class T> bool KVConfigEqual(const cublasLtMatmulAlgo_t& algo,cublasLtMatmulAlgoConfigAttributes_t attribute,
    T expected,const char* name,ModelOptFP8KV128InitTrace& trace){
    T value{};size_t bytes=0;
    const auto status=trace.api("cublasLtMatmulAlgoConfigGetAttribute",[&](){return cublasLtMatmulAlgoConfigGetAttribute(&algo,attribute,&value,sizeof(value),&bytes);});
    trace.note("configuration",name,static_cast<long long>(value),static_cast<long long>(bytes),static_cast<long long>(expected));
    return status==CUBLAS_STATUS_SUCCESS&&bytes==sizeof(value)&&value==expected;
}
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
std::shared_ptr<ModelOptFP8KV128Context> CreateModelOptFP8KV128Context(){
    ModelOptFP8KV128InitTrace trace;
#if CUDART_VERSION >= 12080
    int device=-1;cudaDeviceProp properties{};
    if(trace.api("cudaGetDevice",[&](){return cudaGetDevice(&device);})!=cudaSuccess||trace.api("cudaGetDeviceProperties",[&](){return cudaGetDeviceProperties(&properties,device);})!=cudaSuccess||
       properties.major!=12||properties.minor!=0||trace.api("cublasLtGetVersion",[&](){return cublasLtGetVersion();})!=130400)return trace.empty<ModelOptFP8KV128Context>("device_or_version",device,properties.major,properties.minor);
    std::lock_guard<std::mutex> lock(contextsMutex);
    if(auto previous=contexts[device].lock()){trace.success(true);return previous;}
    auto context=std::make_shared<ModelOptFP8KV128Context>();context->device=device;
    const auto okay=[](cublasStatus_t status){return status==CUBLAS_STATUS_SUCCESS;};
    if(!okay(trace.api("cublasLtCreate",[&](){return cublasLtCreate(&context->handle);}))||
       !okay(trace.api("cublasLtMatmulDescCreate",[&](){return cublasLtMatmulDescCreate(&context->operation,CUBLAS_COMPUTE_32F,CUDA_R_32F);})))return trace.empty<ModelOptFP8KV128Context>("handle_or_operation");
    const cublasOperation_t transA=CUBLAS_OP_T,transB=CUBLAS_OP_N;
    if(!okay(trace.api("cublasLtMatmulDescSetAttribute",[&](){return cublasLtMatmulDescSetAttribute(context->operation,CUBLASLT_MATMUL_DESC_TRANSA,&transA,sizeof(transA));}))||
       !okay(trace.api("cublasLtMatmulDescSetAttribute",[&](){return cublasLtMatmulDescSetAttribute(context->operation,CUBLASLT_MATMUL_DESC_TRANSB,&transB,sizeof(transB));}))||
       !okay(trace.api("cublasLtMatrixLayoutCreate",[&](){return cublasLtMatrixLayoutCreate(&context->weight,CUDA_R_8F_E4M3,5120,1024,5120);}))||
       !okay(trace.api("cublasLtMatrixLayoutCreate",[&](){return cublasLtMatrixLayoutCreate(&context->input,CUDA_R_8F_E4M3,5120,128,5120);}))||
       !okay(trace.api("cublasLtMatrixLayoutCreate",[&](){return cublasLtMatrixLayoutCreate(&context->output,CUDA_R_16BF,1024,128,1024);})))return trace.empty<ModelOptFP8KV128Context>("operation_or_actual_layout");
    CanonicalSelection canonical;
    if(!okay(trace.api("cublasLtMatrixLayoutCreate",[&](){return cublasLtMatrixLayoutCreate(&canonical.input,CUDA_R_8F_E4M3,5120,32,5120);}))||
       !okay(trace.api("cublasLtMatrixLayoutCreate",[&](){return cublasLtMatrixLayoutCreate(&canonical.output,CUDA_R_16BF,1024,32,1024);}))||
       !okay(trace.api("cublasLtMatmulPreferenceCreate",[&](){return cublasLtMatmulPreferenceCreate(&canonical.preference);})))return trace.empty<ModelOptFP8KV128Context>("canonical_layout_or_preference");
    // Preserve actual M32 preference, then AlgoCheck actual M128 descriptors.
    const size_t canonicalWorkspace=4ULL<<20;
    if(!okay(trace.api("cublasLtMatmulPreferenceSetAttribute",[&](){return cublasLtMatmulPreferenceSetAttribute(canonical.preference,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&canonicalWorkspace,sizeof(canonicalWorkspace));})))return trace.empty<ModelOptFP8KV128Context>("workspace_preference");
    cublasLtMatmulHeuristicResult_t choices[8]{};int found=0;
    if(!okay(trace.api("cublasLtMatmulAlgoGetHeuristic",[&](){return cublasLtMatmulAlgoGetHeuristic(context->handle,context->operation,context->weight,canonical.input,
        canonical.output,canonical.output,canonical.preference,8,choices,&found);})))return trace.empty<ModelOptFP8KV128Context>("heuristic_api");
    trace.note("heuristic","found",found);
    if(found>=0&&found<=8)for(int j=0;j<found;++j)trace.note("candidate","heuristic",j,int(choices[j].state),static_cast<long long>(choices[j].workspaceSize),&choices[j].algo,sizeof(choices[j].algo));
    // The live SDK supplies the M32 control. Never import or patch opaque bytes.
    static_assert(sizeof(cublasLtMatmulAlgo_t)==64,"Pinned algorithm ABI");
    static_assert(CUBLASLT_MATMUL_TILE_64x64==15&&CUBLASLT_MATMUL_STAGES_64x6==18&&CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE==2,"Pinned public enum ABI");
    if(found<1||found>8)return trace.empty<ModelOptFP8KV128Context>("heuristic_count",found);
    for(int i=0;i<found;++i){
        if(choices[i].state!=CUBLAS_STATUS_SUCCESS)continue;
        if(!KVConfigEqual<int32_t>(choices[i].algo,CUBLASLT_ALGO_CONFIG_ID,35,"ID",trace)||
           !KVConfigEqual<uint32_t>(choices[i].algo,CUBLASLT_ALGO_CONFIG_TILE_ID,CUBLASLT_MATMUL_TILE_64x64,"TILE_ID",trace)||
           !KVConfigEqual<uint32_t>(choices[i].algo,CUBLASLT_ALGO_CONFIG_STAGES_ID,CUBLASLT_MATMUL_STAGES_64x6,"STAGES_ID",trace)||
           !KVConfigEqual<int32_t>(choices[i].algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,7,"SPLITK_NUM",trace)||
           !KVConfigEqual<uint32_t>(choices[i].algo,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE,"REDUCTION_SCHEME",trace)||
           !KVConfigEqual<uint32_t>(choices[i].algo,CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,1,"CTA_SWIZZLING",trace))
            return trace.empty<ModelOptFP8KV128Context>("documented_M32_configuration");
        trace.note("selection","documented_M32_control",i,0,0,&choices[i].algo,sizeof(choices[i].algo));
        cublasLtMatmulHeuristicResult_t control{};
        const auto controlStatus=trace.api("cublasLtMatmulAlgoCheck",[&](){return cublasLtMatmulAlgoCheck(context->handle,context->operation,context->weight,canonical.input,
            canonical.output,canonical.output,&choices[i].algo,&control);});
        trace.note("check","M32",int(control.state),static_cast<long long>(control.workspaceSize));
        if(!okay(controlStatus)||control.state!=CUBLAS_STATUS_SUCCESS||control.workspaceSize>ModelOptFP8KV128BlasWorkspace)
            return trace.empty<ModelOptFP8KV128Context>("control_algorithm_check",int(control.state),static_cast<long long>(control.workspaceSize));
        cublasLtMatmulHeuristicResult_t checked{};
        if(!okay(trace.api("cublasLtMatmulAlgoCheck",[&](){return cublasLtMatmulAlgoCheck(context->handle,context->operation,context->weight,context->input,
            context->output,context->output,&choices[i].algo,&checked);}))||checked.state!=CUBLAS_STATUS_SUCCESS||checked.workspaceSize>ModelOptFP8KV128BlasWorkspace)return trace.empty<ModelOptFP8KV128Context>("actual_algorithm_check",int(checked.state),static_cast<long long>(checked.workspaceSize));
        trace.note("check","M128",int(checked.state),static_cast<long long>(checked.workspaceSize));
        context->info.device=device;context->info.algorithm=35;context->info.requiredWorkspace=checked.workspaceSize;context->info.opaque=choices[i].algo;
        contexts[device]=context;trace.success(false);return context;
    }
#endif
    return trace.empty<ModelOptFP8KV128Context>("no_supported_candidate_or_CUDART");
}
bool GetModelOptFP8KV128Info(const std::shared_ptr<ModelOptFP8KV128Context>& context,ModelOptFP8KV128Info& info) noexcept{
    if(!context)return false;
    try{std::lock_guard<std::mutex> lock(context->mutex);info=context->info;return info.algorithm==35;}catch(...){return false;}
}
cublasStatus_t RunModelOptFP8KV128Matmul(const std::shared_ptr<ModelOptFP8KV128Context>& context,
    const void* weights,const void* input,void* output,int batch,int rows,int columns,float globalScale,
    void* workspace,size_t workspaceBytes,cudaStream_t stream) noexcept{
    if(!context)return CUBLAS_STATUS_NOT_INITIALIZED;
    if(!weights||!input||!output||!workspace||batch!=128||rows!=1024||columns!=5120||
       workspaceBytes!=ModelOptFP8KV128BlasWorkspace||!std::isfinite(globalScale)||globalScale<=0)return CUBLAS_STATUS_INVALID_VALUE;
    try{
        int device=-1;if(cudaGetDevice(&device)!=cudaSuccess||device!=context->device)return CUBLAS_STATUS_INVALID_VALUE;
        std::lock_guard<std::mutex> lock(context->mutex);const float beta=0;
        return cublasLtMatmul(context->handle,context->operation,&globalScale,weights,context->weight,input,context->input,
            &beta,output,context->output,output,context->output,&context->info.opaque,workspace,workspaceBytes,stream);
    }catch(...){return CUBLAS_STATUS_INTERNAL_ERROR;}
}
}
