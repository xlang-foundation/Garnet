// Actual separate helper constructor and public configuration checks; no GPU matmul.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include "modelopt_fp8_z128_canonical.h"
#include <cuda_runtime_api.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
static void hex(const void* value,size_t bytes){const auto* p=static_cast<const unsigned char*>(value);for(size_t i=0;i<bytes;++i)std::printf("%02x",unsigned(p[i]));}
int main(){
    try{
        auto first=Garnet::CreateModelOptFP8Z128ReconstructedContext();auto second=first?Garnet::CreateModelOptFP8Z128ReconstructedContext():std::shared_ptr<Garnet::ModelOptFP8Z128Context>{};Garnet::ModelOptFP8Z128Info a,b;
        const bool info=Garnet::GetModelOptFP8Z128Info(first,a)&&Garnet::GetModelOptFP8Z128Info(second,b);const bool cached=first&&second==first;
        std::printf("{\"status\":\"actual_separate_Z128_reconstruction_constructor_collected\",\"created\":%s,\"cached\":%s,\"info_okay\":%s,\"device\":%d,\"algorithm\":%d,\"workspace\":%zu,\"opaque\":\"",first?"true":"false",cached?"true":"false",info?"true":"false",a.device,a.algorithm,a.requiredWorkspace);hex(&a.opaque,sizeof(a.opaque));
        std::printf("\",\"cached_opaque_exact\":%s,\"cublaslt_version\":%zu,\"cuda_error\":%d,\"module_paths\":[",std::memcmp(&a.opaque,&b.opaque,sizeof(a.opaque))==0?"true":"false",cublasLtGetVersion(),int(cudaPeekAtLastError()));
        HMODULE blas=GetModuleHandleA("cublasLt64_13.dll"),runtime=nullptr,driver=GetModuleHandleA("nvcuda.dll");
        if(!blas||!driver||!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCSTR>(&cudaGetDevice),&runtime)||!runtime)return 4;
        const char* names[]={"cublasLt64_13.dll","cudaGetDevice","nvcuda.dll"};const HMODULE modules[]={blas,runtime,driver};
        for(int i=0;i<3;++i){char path[MAX_PATH]{};const auto bytes=GetModuleFileNameA(modules[i],path,MAX_PATH);if(!bytes||bytes>=MAX_PATH)return 4;std::printf("%s{\"name\":\"%s\",\"path_hex\":\"",i?",":"",names[i]);hex(path,bytes);std::printf("\"}");}
        int runtimeVersion=0,driverVersion=0;const auto runtimeCode=cudaRuntimeGetVersion(&runtimeVersion);const auto driverCode=cudaDriverGetVersion(&driverVersion);
        std::printf("],\"cuda_runtime_linkage\":\"%s\",\"named_cudart_DLL_present\":%s,\"cuda_runtime_status\":%d,\"cuda_runtime_version\":%d,\"cuda_runtime_header_version\":%d,\"cuda_driver_status\":%d,\"cuda_driver_version\":%d,\"configuration\":[",
            runtime==GetModuleHandleA(nullptr)?"static":"dynamic",GetModuleHandleA("cudart64_13.dll")?"true":"false",int(runtimeCode),runtimeVersion,CUDART_VERSION,int(driverCode),driverVersion);
        struct Field{const char* name;cublasLtMatmulAlgoConfigAttributes_t attribute;size_t bytes;};
        const Field fields[]={{"ID",CUBLASLT_ALGO_CONFIG_ID,4},{"TILE_ID",CUBLASLT_ALGO_CONFIG_TILE_ID,4},
            {"STAGES_ID",CUBLASLT_ALGO_CONFIG_STAGES_ID,4},{"SPLITK_NUM",CUBLASLT_ALGO_CONFIG_SPLITK_NUM,4},
            {"REDUCTION_SCHEME",CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,4},{"CTA_SWIZZLING",CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,4},
            {"CUSTOM_OPTION",CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,4},{"INNER_SHAPE_ID",CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID,2},
            {"CLUSTER_SHAPE_ID",CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID,2}};
        for(size_t i=0;info&&i<9;++i){uint32_t value32=0;uint16_t value16=0;size_t written=0;const auto& field=fields[i];void* value=field.bytes==2?static_cast<void*>(&value16):static_cast<void*>(&value32);
            const auto code=cublasLtMatmulAlgoConfigGetAttribute(&a.opaque,field.attribute,value,field.bytes,&written);
            std::printf("%s{\"name\":\"%s\",\"status\":%d,\"value\":%u,\"requested\":%zu,\"written\":%zu}",i?",":"",field.name,int(code),field.bytes==2?unsigned(value16):unsigned(value32),field.bytes,written);
        }
        std::puts("],\"GPU_matmul_or_model_executed\":false,\"numerical_equivalence_established\":false}");
        return first&&cached&&info&&a.algorithm==67&&a.requiredWorkspace<=Garnet::ModelOptFP8Z128BlasWorkspace?0:1;
    }catch(...){return 3;}
}
