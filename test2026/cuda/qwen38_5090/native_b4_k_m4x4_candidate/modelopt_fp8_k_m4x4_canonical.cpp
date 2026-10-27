// SPDX-License-Identifier: Apache-2.0
#include "modelopt_fp8_k_m4x4_canonical.h"
#include "modelopt_fp8_k_m4x4_init_trace.h"
#include <cuda_runtime_api.h>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <mutex>
#include <map>
namespace Garnet {
struct ModelOptFP8KM4X4Context {
    int device=-1;
    cublasLtHandle_t handle=nullptr;
    cublasLtMatmulDesc_t operation=nullptr;
    cublasLtMatrixLayout_t weight=nullptr,input=nullptr,output=nullptr;
    ModelOptFP8KM4X4Info info;
    std::mutex mutex;
    ~ModelOptFP8KM4X4Context(){
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
std::map<std::pair<int,int>,std::weak_ptr<ModelOptFP8KM4X4Context>> contexts;
template<class T> bool KM4X4ConfigEqual(const cublasLtMatmulAlgo_t& algo,cublasLtMatmulAlgoConfigAttributes_t attribute,
    T expected,const char* name,ModelOptFP8KM4X4InitTrace& trace){
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
std::shared_ptr<ModelOptFP8KM4X4Context> CreateModelOptFP8KM4X4Context(int rows){
    ModelOptFP8KM4X4InitTrace trace;
    if(rows!=1024)return trace.empty<ModelOptFP8KM4X4Context>("unqualified_projection_geometry",rows);
#if CUDART_VERSION >= 12080
    int device=-1;cudaDeviceProp properties{};
    if(trace.api("cudaGetDevice",[&](){return cudaGetDevice(&device);})!=cudaSuccess||trace.api("cudaGetDeviceProperties",[&](){return cudaGetDeviceProperties(&properties,device);})!=cudaSuccess||
       properties.major!=12||properties.minor!=0||trace.api("cublasLtGetVersion",[&](){return cublasLtGetVersion();})!=130400)return trace.empty<ModelOptFP8KM4X4Context>("device_or_version",device,properties.major,properties.minor);
    const auto key=std::make_pair(device,rows);
    std::lock_guard<std::mutex> lock(contextsMutex);
    if(auto previous=contexts[key].lock()){trace.success(true);return previous;}
    auto context=std::make_shared<ModelOptFP8KM4X4Context>();context->device=device;
    const auto okay=[](cublasStatus_t status){return status==CUBLAS_STATUS_SUCCESS;};
    if(!okay(trace.api("cublasLtCreate",[&](){return cublasLtCreate(&context->handle);}))||
       !okay(trace.api("cublasLtMatmulDescCreate",[&](){return cublasLtMatmulDescCreate(&context->operation,CUBLAS_COMPUTE_32F,CUDA_R_32F);})))return trace.empty<ModelOptFP8KM4X4Context>("handle_or_operation");
    const cublasOperation_t transA=CUBLAS_OP_T,transB=CUBLAS_OP_N;
    if(!okay(trace.api("cublasLtMatmulDescSetAttribute",[&](){return cublasLtMatmulDescSetAttribute(context->operation,CUBLASLT_MATMUL_DESC_TRANSA,&transA,sizeof(transA));}))||
       !okay(trace.api("cublasLtMatmulDescSetAttribute",[&](){return cublasLtMatmulDescSetAttribute(context->operation,CUBLASLT_MATMUL_DESC_TRANSB,&transB,sizeof(transB));}))||
       !okay(trace.api("cublasLtMatrixLayoutCreate",[&](){return cublasLtMatrixLayoutCreate(&context->weight,CUDA_R_8F_E4M3,5120,rows,5120);}))||
       !okay(trace.api("cublasLtMatrixLayoutCreate",[&](){return cublasLtMatrixLayoutCreate(&context->input,CUDA_R_8F_E4M3,5120,4,5120);}))||
       !okay(trace.api("cublasLtMatrixLayoutCreate",[&](){return cublasLtMatrixLayoutCreate(&context->output,CUDA_R_16BF,rows,4,rows);})))return trace.empty<ModelOptFP8KM4X4Context>("operation_or_actual_layout");
    CanonicalSelection canonical;
    if(!okay(trace.api("cublasLtMatrixLayoutCreate",[&](){return cublasLtMatrixLayoutCreate(&canonical.input,CUDA_R_8F_E4M3,5120,4,5120);}))||
       !okay(trace.api("cublasLtMatrixLayoutCreate",[&](){return cublasLtMatrixLayoutCreate(&canonical.output,CUDA_R_16BF,rows,4,rows);}))||
       !okay(trace.api("cublasLtMatmulPreferenceCreate",[&](){return cublasLtMatmulPreferenceCreate(&canonical.preference);})))return trace.empty<ModelOptFP8KM4X4Context>("canonical_layout_or_preference");
    // Select the live M4 reference and check both reference and persistent actual M4 layouts.
    const size_t canonicalWorkspace=4ULL<<20;
    if(!okay(trace.api("cublasLtMatmulPreferenceSetAttribute",[&](){return cublasLtMatmulPreferenceSetAttribute(canonical.preference,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&canonicalWorkspace,sizeof(canonicalWorkspace));})))return trace.empty<ModelOptFP8KM4X4Context>("workspace_preference");
    cublasLtMatmulHeuristicResult_t choices[8]{};int found=0;
    if(!okay(trace.api("cublasLtMatmulAlgoGetHeuristic",[&](){return cublasLtMatmulAlgoGetHeuristic(context->handle,context->operation,context->weight,canonical.input,
        canonical.output,canonical.output,canonical.preference,8,choices,&found);})))return trace.empty<ModelOptFP8KM4X4Context>("heuristic_api");
    trace.note("heuristic","found",found);
    if(found>=0&&found<=8)for(int j=0;j<found;++j)trace.note("candidate","heuristic",j,int(choices[j].state),static_cast<long long>(choices[j].workspaceSize),&choices[j].algo,sizeof(choices[j].algo));
    // The live SDK supplies the M4 control. Never import or patch opaque bytes.
    static_assert(sizeof(cublasLtMatmulAlgo_t)==64,"Pinned algorithm ABI");
    static_assert(CUBLASLT_MATMUL_TILE_64x32==13&&CUBLASLT_MATMUL_STAGES_64x5==17,"Pinned public enum ABI");
    if(found<1||found>8)return trace.empty<ModelOptFP8KM4X4Context>("heuristic_count",found);
    for(int i=0;i<found;++i){
        if(choices[i].state!=CUBLAS_STATUS_SUCCESS)continue;
        if(!KM4X4ConfigEqual<int32_t>(choices[i].algo,CUBLASLT_ALGO_CONFIG_ID,35,"ID",trace)||
           !KM4X4ConfigEqual<uint32_t>(choices[i].algo,CUBLASLT_ALGO_CONFIG_TILE_ID,uint32_t(CUBLASLT_MATMUL_TILE_64x32),"TILE_ID",trace)||
           !KM4X4ConfigEqual<uint32_t>(choices[i].algo,CUBLASLT_ALGO_CONFIG_STAGES_ID,uint32_t(CUBLASLT_MATMUL_STAGES_64x5),"STAGES_ID",trace)||
           !KM4X4ConfigEqual<int32_t>(choices[i].algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,3,"SPLITK_NUM",trace)||
           !KM4X4ConfigEqual<uint32_t>(choices[i].algo,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,2,"REDUCTION_SCHEME",trace)||
           !KM4X4ConfigEqual<uint32_t>(choices[i].algo,CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,0u,"CTA_SWIZZLING",trace)||
           !KM4X4ConfigEqual<int32_t>(choices[i].algo,CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,1,"CUSTOM_OPTION",trace))
            return trace.empty<ModelOptFP8KM4X4Context>("observed_M4_configuration");
        trace.note("selection","observed_M4_control",i,0,0,&choices[i].algo,sizeof(choices[i].algo));
        cublasLtMatmulHeuristicResult_t control{};
        const auto controlStatus=trace.api("cublasLtMatmulAlgoCheck",[&](){return cublasLtMatmulAlgoCheck(context->handle,context->operation,context->weight,canonical.input,
            canonical.output,canonical.output,&choices[i].algo,&control);});
        trace.note("check","M4",int(control.state),static_cast<long long>(control.workspaceSize));
        if(!okay(controlStatus)||control.state!=CUBLAS_STATUS_SUCCESS||control.workspaceSize>ModelOptFP8KM4X4BlasWorkspace)
            return trace.empty<ModelOptFP8KM4X4Context>("control_algorithm_check",int(control.state),static_cast<long long>(control.workspaceSize));
        cublasLtMatmulHeuristicResult_t checked{};
        if(!okay(trace.api("cublasLtMatmulAlgoCheck",[&](){return cublasLtMatmulAlgoCheck(context->handle,context->operation,context->weight,context->input,
            context->output,context->output,&choices[i].algo,&checked);}))||checked.state!=CUBLAS_STATUS_SUCCESS||checked.workspaceSize>ModelOptFP8KM4X4BlasWorkspace)return trace.empty<ModelOptFP8KM4X4Context>("actual_algorithm_check",int(checked.state),static_cast<long long>(checked.workspaceSize));
        trace.note("check","M4_actual",int(checked.state),static_cast<long long>(checked.workspaceSize));
        context->info.device=device;context->info.algorithm=35;context->info.rows=rows;context->info.requiredWorkspace=checked.workspaceSize;context->info.opaque=choices[i].algo;
        contexts[key]=context;trace.success(false);return context;
    }
#endif
    return trace.empty<ModelOptFP8KM4X4Context>("no_supported_candidate_or_CUDART");
}
bool GetModelOptFP8KM4X4Info(const std::shared_ptr<ModelOptFP8KM4X4Context>& context,ModelOptFP8KM4X4Info& info) noexcept{
    if(!context)return false;
    try{std::lock_guard<std::mutex> lock(context->mutex);info=context->info;return info.algorithm==35&&info.rows==1024&&info.inputRowsPerMatmul==4&&info.matmulsPerEnqueue==4;}catch(...){return false;}
}
cublasStatus_t RunModelOptFP8KM4X4Matmul(const std::shared_ptr<ModelOptFP8KM4X4Context>& context,
    const void* weights,const void* input,void* output,int batch,int rows,int columns,float globalScale,
    void* workspace,size_t workspaceBytes,cudaStream_t stream) noexcept{
    if(!context)return CUBLAS_STATUS_NOT_INITIALIZED;
    if(!weights||!input||!output||!workspace||batch!=16||rows!=context->info.rows||columns!=5120||
       workspaceBytes!=ModelOptFP8KM4X4BlasWorkspace||!std::isfinite(globalScale)||globalScale<=0)return CUBLAS_STATUS_INVALID_VALUE;
    try{
        int device=-1;if(cudaGetDevice(&device)!=cudaSuccess||device!=context->device)return CUBLAS_STATUS_INVALID_VALUE;
        std::lock_guard<std::mutex> lock(context->mutex);const float beta=0;
        // Preserve the actual M4 shape for each physical row's four-token block.
        // Quantized input and BF16 output are contiguous [16,K] and [16,N].
        // Each ordered call uses the original weight, scalar, workspace and stream.
        for(int group=0;group<4;++group){
            const auto* groupInput=static_cast<const unsigned char*>(input)+size_t(group)*4*5120;
            auto* groupOutput=static_cast<unsigned char*>(output)+size_t(group)*4*1024*sizeof(uint16_t);
            const auto status=cublasLtMatmul(context->handle,context->operation,&globalScale,weights,context->weight,groupInput,context->input,
                &beta,groupOutput,context->output,groupOutput,context->output,&context->info.opaque,workspace,workspaceBytes,stream);
            if(status!=CUBLAS_STATUS_SUCCESS)return status;
        }
        return CUBLAS_STATUS_SUCCESS;
    }catch(...){return CUBLAS_STATUS_INTERNAL_ERROR;}
}
}
