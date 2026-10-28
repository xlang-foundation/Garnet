// Read documented configuration attributes of two retained opaque descriptors.
// No descriptor changes, context construction, AlgoCheck, or GPU matmul calls.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cublasLt.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
static bool decode(const char* hex,cublasLtMatmulAlgo_t& result){
    if(std::strlen(hex)!=sizeof(result)*2)return false;
    auto* bytes=reinterpret_cast<unsigned char*>(&result);
    for(size_t i=0;i<sizeof(result);++i){char pair[3]={hex[i*2],hex[i*2+1],0};char* end=nullptr;const auto v=std::strtoul(pair,&end,16);if(end!=pair+2)return false;bytes[i]=static_cast<unsigned char>(v);}
    return true;
}
static void hex(const void* value,size_t bytes){const auto* raw=static_cast<const unsigned char*>(value);for(size_t i=0;i<bytes;++i)std::printf("%02x",unsigned(raw[i]));}
int main(int argc,char** argv){
    static_assert(sizeof(cublasLtMatmulAlgo_t)==64,"Pinned algorithm ABI");
    if(argc!=3)return 2;cublasLtMatmulAlgo_t values[2]{};if(!decode(argv[1],values[0])||!decode(argv[2],values[1]))return 3;
    const auto version=cublasLtGetVersion();if(version!=130400)return 4;
    char module[MAX_PATH]{};const DWORD bytes=GetModuleFileNameA(GetModuleHandleA("cublasLt64_13.dll"),module,MAX_PATH);if(!bytes||bytes>=MAX_PATH)return 5;
    struct Attribute{const char* name;cublasLtMatmulAlgoConfigAttributes_t id;size_t bytes;};
    const Attribute attrs[]={
        {"ID",CUBLASLT_ALGO_CONFIG_ID,4},{"TILE_ID",CUBLASLT_ALGO_CONFIG_TILE_ID,4},
        {"STAGES_ID",CUBLASLT_ALGO_CONFIG_STAGES_ID,4},{"SPLITK_NUM",CUBLASLT_ALGO_CONFIG_SPLITK_NUM,4},
        {"REDUCTION_SCHEME",CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,4},{"CTA_SWIZZLING",CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,4},
        {"CUSTOM_OPTION",CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,4},{"INNER_SHAPE_ID",CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID,2},
        {"CLUSTER_SHAPE_ID",CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID,2}};
    std::printf("{\"status\":\"retained_opaque_documented_config_queries_collected\",\"cublaslt_version\":%zu,\"module_path_hex\":\"",version);hex(module,bytes);std::printf("\",\"descriptors\":[");
    for(int i=0;i<2;++i){
        std::printf("%s{\"label\":\"%s\",\"opaque\":\"",i?",":"",i?"actual_failed_plugin_first_candidate":"qualified_M32");hex(&values[i],sizeof(values[i]));std::printf("\",\"attributes\":[");
        for(size_t j=0;j<sizeof(attrs)/sizeof(attrs[0]);++j){
            uint64_t buffer=0;size_t written=0;const auto& attr=attrs[j];const auto code=cublasLtMatmulAlgoConfigGetAttribute(&values[i],attr.id,&buffer,attr.bytes,&written);
            std::printf("%s{\"name\":\"%s\",\"attribute\":%d,\"requested_bytes\":%zu,\"status\":%d,\"written\":%zu,\"value_hex\":\"",j?",":"",attr.name,int(attr.id),attr.bytes,int(code),written);hex(&buffer,attr.bytes);std::printf("\"}");
        }
        std::printf("]}");
    }
    std::puts("],\"algorithm_substituted\":false,\"GPU_model_or_matmul_executed\":false,\"functional_equivalence_established\":false}");return 0;
}
