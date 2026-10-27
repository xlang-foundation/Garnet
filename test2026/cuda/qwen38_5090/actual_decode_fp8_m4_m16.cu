// SPDX-License-Identifier: Apache-2.0
// Untimed SDK intervention on original inputs; no Garnet DLL or model execution.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublasLt.h>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "modelopt_cublaslt.h"
#include "modelopt_fp8.cuh"
static cublasLtMatmulAlgo_t referenceAlgorithm{};
static std::array<int32_t,7> referenceConfig{};
static bool referenceValid=false;
static const char* mode="none";
static int stepIndex=-1,repeatIndex=-1,rowIndex=-1;
static const char* projection="none";
static cublasStatus_t ObserveMatmul(cublasLtHandle_t handle,cublasLtMatmulDesc_t operation,
    const void* alpha,const void* a,cublasLtMatrixLayout_t ad,const void* b,cublasLtMatrixLayout_t bd,
    const void* beta,const void* c,cublasLtMatrixLayout_t cd,void* d,cublasLtMatrixLayout_t dd,
    const cublasLtMatmulAlgo_t* proposed,void* workspace,size_t workspaceBytes,cudaStream_t stream){
    uint64_t m=0;size_t bytes=0;
    if(cublasLtMatrixLayoutGetAttribute(bd,CUBLASLT_MATRIX_LAYOUT_COLS,&m,sizeof(m),&bytes)!=CUBLAS_STATUS_SUCCESS||bytes!=sizeof(m))return CUBLAS_STATUS_INVALID_VALUE;
    const cublasLtMatmulAlgo_t* selected=proposed;
    if(!std::strcmp(mode,"shared16")){
        if(!referenceValid||m!=16)return CUBLAS_STATUS_INVALID_VALUE;
        cublasLtMatmulHeuristicResult_t check{};
        const auto status=cublasLtMatmulAlgoCheck(handle,operation,ad,bd,cd,dd,&referenceAlgorithm,&check);
        printf("ALGO_CHECK projection=%s step=%d repeat=%d m=%llu status=%d state=%d workspace=%zu capacity=%zu\n",projection,stepIndex,repeatIndex,(unsigned long long)m,int(status),int(check.state),check.workspaceSize,workspaceBytes);
        if(status!=CUBLAS_STATUS_SUCCESS||check.state!=CUBLAS_STATUS_SUCCESS||check.workspaceSize>workspaceBytes)return CUBLAS_STATUS_INVALID_VALUE; // No fallback.
        selected=&referenceAlgorithm;
    }
    const cublasLtMatmulAlgoConfigAttributes_t attrs[]={CUBLASLT_ALGO_CONFIG_ID,CUBLASLT_ALGO_CONFIG_TILE_ID,CUBLASLT_ALGO_CONFIG_STAGES_ID,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION};
    std::array<int32_t,7> config{};
    for(size_t i=0;i<config.size();++i){size_t written=0;const auto status=cublasLtMatmulAlgoConfigGetAttribute(selected,attrs[i],&config[i],sizeof(config[i]),&written);if(status!=CUBLAS_STATUS_SUCCESS||written!=sizeof(config[i]))return CUBLAS_STATUS_INVALID_VALUE;}
    const auto status=cublasLtMatmul(handle,operation,alpha,a,ad,b,bd,beta,c,cd,d,dd,selected,workspace,workspaceBytes,stream);
    if(!std::strcmp(mode,"m4")&&status==CUBLAS_STATUS_SUCCESS){
        if(m!=4||(referenceValid&&config!=referenceConfig))return CUBLAS_STATUS_INVALID_VALUE;
        referenceAlgorithm=*selected;referenceConfig=config;referenceValid=true;
    }
    printf("ALGO projection=%s step=%d repeat=%d row=%d mode=%s m=%llu status=%d id=%d tile=%d stages=%d split=%d reduction=%d swizzle=%d custom=%d workspace=%zu\n",projection,stepIndex,repeatIndex,rowIndex,mode,(unsigned long long)m,int(status),config[0],config[1],config[2],config[3],config[4],config[5],config[6],workspaceBytes);fflush(stdout);
    return status;
}
#define cublasLtMatmul ObserveMatmul
#include "modelopt_cublaslt.cpp"
#undef cublasLtMatmul
static void Check(cudaError_t s){if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s));}
static std::vector<unsigned char> Read(const std::string& path,size_t count){
    std::ifstream f(path,std::ios::binary|std::ios::ate);if(!f||f.tellg()!=static_cast<std::streamoff>(count))throw std::runtime_error("Wrong input size: "+path);
    f.seekg(0);std::vector<unsigned char> data(count);if(!f.read(reinterpret_cast<char*>(data.data()),count))throw std::runtime_error("Input read failed");return data;
}
static void Write(const std::string& path,const std::vector<unsigned char>& data){
    {std::ifstream prior(path,std::ios::binary);if(prior)throw std::runtime_error("Preserve prior output: "+path);}
    std::ofstream f(path,std::ios::binary);if(!f.write(reinterpret_cast<const char*>(data.data()),data.size()))throw std::runtime_error("Output write failed");
}
struct Device{void* p=nullptr;explicit Device(size_t bytes){Check(cudaMalloc(&p,bytes));}~Device(){if(p)cudaFree(p);}};
__global__ void Cast(const float* input,__nv_bfloat16* output,int count){const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)output[i]=__float2bfloat16_rn(input[i]);}
int main(int argc,char** argv){
    try{
        if(argc!=3)throw std::runtime_error("Usage: control FIXTURE OUTPUT_EXISTING_DIR");
        const std::string fixture=std::string(argv[1])+"/",output=std::string(argv[2])+"/";constexpr int k=5120;constexpr size_t cap=4ULL<<20;
        cudaDeviceProp properties{};Check(cudaGetDeviceProperties(&properties,0));if(properties.major!=12||properties.minor!=0)throw std::runtime_error("Control requires SM120");
        printf("DEVICE %s SM%d%d\n",properties.name,properties.major,properties.minor);
        cublasLtHandle_t handle=nullptr;if(cublasLtCreate(&handle)!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cuBLASLt create failed");
        if(cublasLtGetVersion()!=130400)throw std::runtime_error("Pinned cuBLASLt13.4 required");printf("CUBLASLT_VERSION %zu\n",cublasLtGetVersion());
        char loaded[MAX_PATH]{};const auto length=GetModuleFileNameA(GetModuleHandleA("cublasLt64_13.dll"),loaded,MAX_PATH);if(!length||length>=MAX_PATH)throw std::runtime_error("Loaded SDK path unavailable");for(DWORD i=0;i<length;++i)if(loaded[i]=='\\')loaded[i]='/';printf("CUBLASLT_DLL %s\n",loaded);
        for(const char* label:{"qkv","z"}){
            projection=label;const int n=!std::strcmp(label,"qkv")?10240:6144;referenceValid=false;
            const auto weights=Read(fixture+label+"-weight.e4m3",size_t(n)*k),ws=Read(fixture+label+"-weight-scale.f32",4),xs=Read(fixture+label+"-input-scale.f32",4);
            float weightScale=0,inputScale=0;std::memcpy(&weightScale,ws.data(),4);std::memcpy(&inputScale,xs.data(),4);if(!std::isfinite(weightScale)||!std::isfinite(inputScale)||weightScale<=0||inputScale<=0||!std::isfinite(weightScale*inputScale))throw std::runtime_error("Invalid calibration");
            Device w(weights.size()),workspace(cap),x(16*k*4),bf16(16*k*2),packed16(16*k),packed4(16*k),result(16*n*2);Check(cudaMemcpy(w.p,weights.data(),weights.size(),cudaMemcpyHostToDevice));
            for(int step=0;step<5;++step){
                stepIndex=step;const std::string stem=std::string(label)+"-step"+std::to_string(step);const auto input=Read(fixture+"step"+std::to_string(step)+"-normalized.f32",16*k*4);Check(cudaMemcpy(x.p,input.data(),input.size(),cudaMemcpyHostToDevice));
                Cast<<<(16*k+255)/256,256>>>(static_cast<const float*>(x.p),static_cast<__nv_bfloat16*>(bf16.p),16*k);Check(cudaGetLastError());
                Check(Garnet::QuantizeModelOptFP8Activation(bf16.p,static_cast<unsigned char*>(packed16.p),inputScale,16,k,nullptr));
                for(int row=0;row<4;++row)Check(Garnet::QuantizeModelOptFP8Activation(static_cast<const unsigned char*>(bf16.p)+size_t(row)*4*k*2,static_cast<unsigned char*>(packed4.p)+size_t(row)*4*k,inputScale,4,k,nullptr));
                for(const auto& pair:std::vector<std::pair<std::string,std::pair<void*,size_t>>>{{"cast.bf16",{bf16.p,size_t(16*k*2)}},{"packed16.e4m3",{packed16.p,size_t(16*k)}},{"packed4.e4m3",{packed4.p,size_t(16*k)}}}){
                    std::vector<unsigned char> data(pair.second.second);Check(cudaMemcpy(data.data(),pair.second.first,data.size(),cudaMemcpyDeviceToHost));Write(output+stem+"-"+pair.first,data);
                }
                for(const char* current:{"m4","default16","shared16"}){
                    mode=current;for(int repeat=0;repeat<2;++repeat){repeatIndex=repeat;Check(cudaMemset(result.p,0xcd,16*n*2));
                        const int groups=!std::strcmp(mode,"m4")?4:1,m=groups==4?4:16;
                        for(int row=0;row<groups;++row){rowIndex=row;const size_t first=size_t(row)*4;const auto status=Garnet::RunModelOptFP8Matmul(handle,w.p,static_cast<const unsigned char*>(packed16.p)+first*k,static_cast<unsigned char*>(result.p)+first*n*2,m,n,k,weightScale*inputScale,workspace.p,cap,nullptr,false);if(status!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("SDK matmul or explicit algorithm check failed");}
                        Check(cudaDeviceSynchronize());std::vector<unsigned char> data(16*n*2);Check(cudaMemcpy(data.data(),result.p,data.size(),cudaMemcpyDeviceToHost));Write(output+stem+"-"+mode+"-r"+std::to_string(repeat)+".bf16",data);
                    }
                }
            }
        }
        cublasLtDestroy(handle);puts("ACTUAL_M4_M16_SDK_CAPTURE_COMPLETE_NO_MODEL_OR_PERFORMANCE_ADMISSION");return 0;
    }catch(const std::exception& e){fprintf(stderr,"ERROR %s\n",e.what());return 2;}
}
