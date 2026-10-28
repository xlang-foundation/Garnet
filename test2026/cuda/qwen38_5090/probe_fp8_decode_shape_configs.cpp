// SDK geometry/algorithm admission only. No tensor allocation or matmul/model.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <cuda_runtime_api.h>
#include <cublasLt.h>
#include <array>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
struct Event{std::string name;int status;};std::vector<Event> events;
void record(const char* name,int status){events.push_back({name,status});}
void okay(const char* name,int status){record(name,status);if(status)throw std::runtime_error(name);}
std::string quoted(const std::string& s){std::ostringstream o;o<<std::quoted(s);return o.str();}
std::string hex(const void* value,size_t bytes){auto* p=static_cast<const unsigned char*>(value);std::ostringstream s;s<<std::hex<<std::setfill('0');for(size_t i=0;i<bytes;++i)s<<std::setw(2)<<unsigned(p[i]);return s.str();}
struct Plan{
    cublasLtMatmulDesc_t op=nullptr;cublasLtMatrixLayout_t w=nullptr,x4=nullptr,y4=nullptr,x16=nullptr,y16=nullptr;cublasLtMatmulPreference_t pref=nullptr;
    ~Plan(){if(pref)record("cublasLtMatmulPreferenceDestroy",cublasLtMatmulPreferenceDestroy(pref));for(auto p:{y16,x16,y4,x4,w})if(p)record("cublasLtMatrixLayoutDestroy",cublasLtMatrixLayoutDestroy(p));if(op)record("cublasLtMatmulDescDestroy",cublasLtMatmulDescDestroy(op));}
};
struct Handle{cublasLtHandle_t value=nullptr;~Handle(){if(value)record("cublasLtDestroy",cublasLtDestroy(value));}};
const cublasLtMatmulAlgoConfigAttributes_t attributes[]={CUBLASLT_ALGO_CONFIG_ID,CUBLASLT_ALGO_CONFIG_TILE_ID,CUBLASLT_ALGO_CONFIG_STAGES_ID,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION};
}
int main(){try{
    int device=-1,driver=0,runtime=0;cudaDeviceProp p{};okay("cudaGetDevice",cudaGetDevice(&device));okay("cudaGetDeviceProperties",cudaGetDeviceProperties(&p,device));okay("cudaDriverGetVersion",cudaDriverGetVersion(&driver));okay("cudaRuntimeGetVersion",cudaRuntimeGetVersion(&runtime));const auto version=cublasLtGetVersion();
    if(device!=0||p.major!=12||p.minor!=0||version!=130400||CUDART_VERSION<12080)throw std::runtime_error("Unpinned device/SDK geometry query");
    char path[MAX_PATH]{};auto bytes=GetModuleFileNameA(GetModuleHandleA("cublasLt64_13.dll"),path,MAX_PATH);if(!bytes||bytes>=MAX_PATH)throw std::runtime_error("cuBLASLt module path unavailable");for(DWORD i=0;i<bytes;++i)if(path[i]=='\\')path[i]='/';
    std::ostringstream shapes;shapes<<'[';
    {
        Handle handle;okay("cublasLtCreate",cublasLtCreate(&handle.value));
        struct Geometry{const char* label;int n,k;};const Geometry geometries[]={{"GDN_QKV",10240,5120},{"GDN_Z",6144,5120},{"FA_KV",1024,5120},{"FA_Q",12288,5120},{"FA_O",5120,6144}};
        bool firstShape=true;
        for(const auto& g:geometries){
            Plan a;okay("cublasLtMatmulDescCreate",cublasLtMatmulDescCreate(&a.op,CUBLAS_COMPUTE_32F,CUDA_R_32F));const cublasOperation_t transA=CUBLAS_OP_T,transB=CUBLAS_OP_N;
            okay("cublasLtMatmulDescSetAttribute:TRANSA",cublasLtMatmulDescSetAttribute(a.op,CUBLASLT_MATMUL_DESC_TRANSA,&transA,sizeof(transA)));okay("cublasLtMatmulDescSetAttribute:TRANSB",cublasLtMatmulDescSetAttribute(a.op,CUBLASLT_MATMUL_DESC_TRANSB,&transB,sizeof(transB)));
            okay("cublasLtMatrixLayoutCreate:W",cublasLtMatrixLayoutCreate(&a.w,CUDA_R_8F_E4M3,g.k,g.n,g.k));
            okay("cublasLtMatrixLayoutCreate:X4",cublasLtMatrixLayoutCreate(&a.x4,CUDA_R_8F_E4M3,g.k,4,g.k));okay("cublasLtMatrixLayoutCreate:Y4",cublasLtMatrixLayoutCreate(&a.y4,CUDA_R_16BF,g.n,4,g.n));
            okay("cublasLtMatrixLayoutCreate:X16",cublasLtMatrixLayoutCreate(&a.x16,CUDA_R_8F_E4M3,g.k,16,g.k));okay("cublasLtMatrixLayoutCreate:Y16",cublasLtMatrixLayoutCreate(&a.y16,CUDA_R_16BF,g.n,16,g.n));
            okay("cublasLtMatmulPreferenceCreate",cublasLtMatmulPreferenceCreate(&a.pref));const size_t limit=4ULL<<20;okay("cublasLtMatmulPreferenceSetAttribute",cublasLtMatmulPreferenceSetAttribute(a.pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&limit,sizeof(limit)));
            for(int m:{4,16}){
                cublasLtMatmulHeuristicResult_t choices[8]{};int found=0;auto x=m==4?a.x4:a.x16,y=m==4?a.y4:a.y16;
                okay("cublasLtMatmulAlgoGetHeuristic",cublasLtMatmulAlgoGetHeuristic(handle.value,a.op,a.w,x,y,y,a.pref,8,choices,&found));if(found<1||found>8)throw std::runtime_error("Invalid heuristic count");
                shapes<<(firstShape?"":",")<<"{\"label\":"<<quoted(g.label)<<",\"M\":"<<m<<",\"N\":"<<g.n<<",\"K\":"<<g.k<<",\"workspace_limit\":"<<limit<<",\"found\":"<<found<<",\"choices\":[";firstShape=false;
                for(int i=0;i<found;++i){
                    shapes<<(i?",":"")<<"{\"index\":"<<i<<",\"state\":"<<int(choices[i].state)<<",\"heuristic_workspace\":"<<choices[i].workspaceSize<<",\"opaque\":"<<quoted(hex(&choices[i].algo,sizeof(choices[i].algo)))<<",\"configuration\":[";
                    if(choices[i].state==CUBLAS_STATUS_SUCCESS)for(size_t j=0;j<7;++j){int32_t value=0;size_t written=0;okay("cublasLtMatmulAlgoConfigGetAttribute",cublasLtMatmulAlgoConfigGetAttribute(&choices[i].algo,attributes[j],&value,sizeof(value),&written));if(written!=4)throw std::runtime_error("Unexpected configuration width");shapes<<(j?",":"")<<value;}
                    shapes<<"],\"checks\":[";
                    if(choices[i].state==CUBLAS_STATUS_SUCCESS)for(int checked:{4,16}){cublasLtMatmulHeuristicResult_t result{};const auto status=cublasLtMatmulAlgoCheck(handle.value,a.op,a.w,checked==4?a.x4:a.x16,checked==4?a.y4:a.y16,checked==4?a.y4:a.y16,&choices[i].algo,&result);record("cublasLtMatmulAlgoCheck",status);shapes<<(checked==4?"":",")<<"{\"M\":"<<checked<<",\"status\":"<<int(status)<<",\"state\":"<<int(result.state)<<",\"workspace\":"<<result.workspaceSize<<'}';}
                    shapes<<"]}";
                }
                shapes<<"]}";
            }
        }
    }
    shapes<<']';std::ostringstream out;out<<"{\"status\":\"actual_live_FP8_M4_M16_geometry_configurations_collected_no_matmul\",\"device\":"<<device<<",\"device_name\":"<<quoted(p.name)<<",\"uuid_hex\":"<<quoted(hex(p.uuid.bytes,16))<<",\"compute_capability\":["<<p.major<<','<<p.minor<<"],\"driver_version\":"<<driver<<",\"runtime_version\":"<<runtime<<",\"CUDART_VERSION\":"<<CUDART_VERSION<<",\"cublasLt_version\":"<<version<<",\"cublasLt_DLL\":"<<quoted(path)<<",\"weight_dtype\":\"E4M3\",\"activation_dtype\":\"E4M3\",\"output_dtype\":\"BF16\",\"compute_dtype\":\"FP32\",\"TRANSA\":\"T\",\"TRANSB\":\"N\",\"geometries\":"<<shapes.str()<<",\"API_events\":[";
    for(size_t i=0;i<events.size();++i)out<<(i?",":"")<<"{\"name\":"<<quoted(events[i].name)<<",\"status\":"<<events[i].status<<'}';out<<"],\"matmul_executed\":false,\"model_loaded\":false,\"performance_qualified\":false}";std::puts(out.str().c_str());return 0;
}catch(const std::exception& e){std::fprintf(stderr,"SDK geometry query failed: %s\n",e.what());return 1;}}
