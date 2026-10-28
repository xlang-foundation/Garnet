// SPDX-License-Identifier: Apache-2.0
// Standalone metadata compatibility only. No model weights or matmul execution.
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cuda_runtime_api.h>
#include <cublasLt.h>
#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
constexpr size_t Workspace=4ULL<<20;
void require(bool okay,const char* label){if(!okay)throw std::runtime_error(label);}
struct Attribute {const char* name;cublasLtMatmulAlgoConfigAttributes_t key;};
const Attribute attributes[]={
    {"ID",CUBLASLT_ALGO_CONFIG_ID},{"TILE_ID",CUBLASLT_ALGO_CONFIG_TILE_ID},
    {"STAGES_ID",CUBLASLT_ALGO_CONFIG_STAGES_ID},{"CTA_SWIZZLING",CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING},
    {"CUSTOM_OPTION",CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION},{"SPLITK_NUM",CUBLASLT_ALGO_CONFIG_SPLITK_NUM},
    {"REDUCTION_SCHEME",CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME},{"INNER_SHAPE_ID",CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID},
    {"CLUSTER_SHAPE_ID",CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID}};
struct Value {int status=0;size_t bytes=0;int32_t value=0;bool operator==(const Value& b)const{return status==b.status&&bytes==b.bytes&&value==b.value;}};
using Values=std::array<Value,9>;
Values read(const cublasLtMatmulAlgo_t& algorithm){
    Values result{};for(size_t i=0;i<result.size();++i){auto& v=result[i];v.status=int(cublasLtMatmulAlgoConfigGetAttribute(&algorithm,attributes[i].key,&v.value,sizeof(v.value),&v.bytes));}return result;
}
void emit(std::ostream& out,const Values& values){
    out<<"{";for(size_t i=0;i<values.size();++i){if(i)out<<",";const auto& v=values[i];out<<std::quoted(attributes[i].name)<<":{\"status\":"<<v.status<<",\"bytes\":"<<v.bytes<<",\"value\":"<<v.value<<"}";}out<<"}";
}
bool observed(const Values& values){
    const int expected[]={67,int(CUBLASLT_MATMUL_TILE_64x32),int(CUBLASLT_MATMUL_STAGES_64x5),1,18};
    for(size_t i=0;i<5;++i)if(values[i].status!=CUBLAS_STATUS_SUCCESS||values[i].bytes!=sizeof(int32_t)||values[i].value!=expected[i])return false;return true;
}
std::string module(const char* name){char path[32768]{};auto handle=GetModuleHandleA(name);require(handle&&GetModuleFileNameA(handle,path,sizeof(path))>0,"resolved_module_path");return path;}
std::string uuid(const cudaUUID_t& id){std::ostringstream s;s<<"GPU-"<<std::hex<<std::setfill('0');for(int i=0;i<16;++i){if(i==4||i==6||i==8||i==10)s<<'-';s<<std::setw(2)<<unsigned(static_cast<unsigned char>(id.bytes[i]));}return s.str();}
struct Resources {
    cublasLtHandle_t handle=nullptr;cublasLtMatmulDesc_t operation=nullptr;cublasLtMatrixLayout_t weight=nullptr;
    std::array<cublasLtMatrixLayout_t,3> input{},output{};cublasLtMatmulPreference_t preference=nullptr;
    ~Resources(){if(preference)cublasLtMatmulPreferenceDestroy(preference);for(auto x:output)if(x)cublasLtMatrixLayoutDestroy(x);for(auto x:input)if(x)cublasLtMatrixLayoutDestroy(x);if(weight)cublasLtMatrixLayoutDestroy(weight);if(operation)cublasLtMatmulDescDestroy(operation);if(handle)cublasLtDestroy(handle);}
};
}
int main(int argc,char** argv){
    if(argc!=2)return 2;std::ostringstream report;
    try{
        int device=-1,runtime=0,driver=0;cudaDeviceProp properties{};
        require(cudaGetDevice(&device)==cudaSuccess&&device==0,"cuda_device");require(cudaGetDeviceProperties(&properties,device)==cudaSuccess,"cuda_properties");
        require(properties.major==12&&properties.minor==0&&uuid(properties.uuid)=="GPU-265ee66c-9b79-cd3b-0c63-dd6942ddb5e6","pinned_SM120_GPU_UUID");
        require(cudaRuntimeGetVersion(&runtime)==cudaSuccess&&cudaDriverGetVersion(&driver)==cudaSuccess,"runtime_versions");require(cublasLtGetVersion()==130400,"pinned_cuBLASLt130400");
        Resources r;require(cublasLtCreate(&r.handle)==CUBLAS_STATUS_SUCCESS,"handle");require(cublasLtMatmulDescCreate(&r.operation,CUBLAS_COMPUTE_32F,CUDA_R_32F)==CUBLAS_STATUS_SUCCESS,"operation");
        const cublasOperation_t transA=CUBLAS_OP_T,transB=CUBLAS_OP_N;
        require(cublasLtMatmulDescSetAttribute(r.operation,CUBLASLT_MATMUL_DESC_TRANSA,&transA,sizeof(transA))==CUBLAS_STATUS_SUCCESS&&cublasLtMatmulDescSetAttribute(r.operation,CUBLASLT_MATMUL_DESC_TRANSB,&transB,sizeof(transB))==CUBLAS_STATUS_SUCCESS,"transpose");
        require(cublasLtMatrixLayoutCreate(&r.weight,CUDA_R_8F_E4M3,5120,12288,5120)==CUBLAS_STATUS_SUCCESS,"weight_layout");
        const int sizes[]={16,32,128};for(size_t i=0;i<3;++i){require(cublasLtMatrixLayoutCreate(&r.input[i],CUDA_R_8F_E4M3,5120,sizes[i],5120)==CUBLAS_STATUS_SUCCESS,"input_layout");require(cublasLtMatrixLayoutCreate(&r.output[i],CUDA_R_16BF,12288,sizes[i],12288)==CUBLAS_STATUS_SUCCESS,"output_layout");}
        require(cublasLtMatmulPreferenceCreate(&r.preference)==CUBLAS_STATUS_SUCCESS&&cublasLtMatmulPreferenceSetAttribute(r.preference,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&Workspace,sizeof(Workspace))==CUBLAS_STATUS_SUCCESS,"workspace_preference");
        cublasLtMatmulHeuristicResult_t choices[8]{};int found=0;
        require(cublasLtMatmulAlgoGetHeuristic(r.handle,r.operation,r.weight,r.input[0],r.output[0],r.output[0],r.preference,8,choices,&found)==CUBLAS_STATUS_SUCCESS&&found>=1&&found<=8,"M16_SDK_heuristic");
        int selected=-1;for(int i=0;i<found;++i)if(choices[i].state==CUBLAS_STATUS_SUCCESS){selected=i;break;}require(selected>=0,"first_successful_candidate");
        const auto& algorithm=choices[selected].algo;const auto before=read(algorithm);require(observed(before),"first_successful_M16_matches_actual_T32_public_configuration");
        report<<"{\"status\":\"QG_M16_SDK_control_M16_M32_M128_compatible\",\"GPU_UUID\":"<<std::quoted(uuid(properties.uuid))<<",\"device\":"<<device<<",\"SM\":120,\"header_CUDART\":"<<CUDART_VERSION<<",\"runtime_version\":"<<runtime<<",\"driver_API_version\":"<<driver<<",\"cublasLt_version\":"<<cublasLtGetVersion()<<",\"geometry\":{\"N\":12288,\"K\":5120},\"workspace_limit\":"<<Workspace<<",\"heuristic_found\":"<<found<<",\"first_successful_index\":"<<selected<<",\"SDK_enums\":{\"TILE_64x32\":"<<int(CUBLASLT_MATMUL_TILE_64x32)<<",\"STAGES_64x5\":"<<int(CUBLASLT_MATMUL_STAGES_64x5)<<"},\"public_configuration\":";emit(report,before);
        report<<",\"checks\":[";
        for(size_t i=0;i<3;++i){cublasLtMatmulHeuristicResult_t checked{};const auto status=cublasLtMatmulAlgoCheck(r.handle,r.operation,r.weight,r.input[i],r.output[i],r.output[i],&algorithm,&checked);require(status==CUBLAS_STATUS_SUCCESS&&checked.state==CUBLAS_STATUS_SUCCESS&&checked.workspaceSize<=Workspace,"same_SDK_algorithm_AlgoCheck");require(read(algorithm)==before,"algorithm_public_attributes_readonly");if(i)report<<",";report<<"{\"M\":"<<sizes[i]<<",\"API_status\":"<<int(status)<<",\"state\":"<<int(checked.state)<<",\"required_workspace\":"<<checked.workspaceSize<<"}";}
        report<<"],\"loaded_modules\":{\"cublasLt64_13.dll\":"<<std::quoted(module("cublasLt64_13.dll"))<<",\"cudart64_13.dll\":"<<std::quoted(module("cudart64_13.dll"))<<"},\"no_model_weights_loaded\":true,\"no_matmul_executed\":true,\"no_opaque_algorithm_import_or_patch\":true,\"model_or_performance_qualified\":false}";
        std::ofstream file(argv[1],std::ios::binary);require(bool(file),"report_open");file<<report.str();require(bool(file),"report_write");return 0;
    }catch(const std::exception& error){std::ofstream file(argv[1],std::ios::binary);file<<"{\"status\":\"failed_SDK_or_configuration_check\",\"error\":"<<std::quoted(error.what())<<",\"model_or_performance_qualified\":false}";return 1;}
}
