// CPU-only factory and operator stubs. Never link into Garnet or a GPU helper.
#include "modelopt_fp8_z128_canonical.h"
#include "modelopt_fp8_m128_canonical.h"
#include "modelopt_cublaslt.h"
#include "modelopt_fp8.cuh"
#include "modelopt_nvfp4.cuh"
#include <atomic>
#include <stdexcept>
namespace Garnet {
namespace {std::atomic<int> mode{0},factories{0},operators{0};}
void ModelOptInitCPUSetMode(int value){mode=value;factories=0;operators=0;}
int ModelOptInitCPUFactoryCalls(){return factories.load();}
int ModelOptInitCPUOperatorCalls(){return operators.load();}
template<class T> std::shared_ptr<T> fake(){
    ++factories;const int value=mode.load();
    if(value==2)throw std::runtime_error("synthetic factory exception");
    if(value==3)throw 17;
    if(value==1){auto value=std::make_shared<int>(1);return std::shared_ptr<T>(value,reinterpret_cast<T*>(value.get()));}
    return {};
}
std::shared_ptr<ModelOptFP8Z128Context> CreateModelOptFP8Z128Context(){return fake<ModelOptFP8Z128Context>();}
std::shared_ptr<ModelOptFP8M128Context> CreateModelOptFP8M128Context(){return fake<ModelOptFP8M128Context>();}
std::shared_ptr<void> CreateModelOptNVFP4Context(){return fake<void>();}
cublasStatus_t RunModelOptFP8Z128Matmul(const std::shared_ptr<ModelOptFP8Z128Context>&,const void*,const void*,void*,int,int,int,float,void*,size_t,cudaStream_t) noexcept{++operators;return CUBLAS_STATUS_NOT_INITIALIZED;}
cublasStatus_t RunModelOptFP8M128Matmul(const std::shared_ptr<ModelOptFP8M128Context>&,const void*,const void*,void*,int,int,int,float,void*,size_t,cudaStream_t) noexcept{++operators;return CUBLAS_STATUS_NOT_INITIALIZED;}
cublasStatus_t RunModelOptFP8MatmulContext(const std::shared_ptr<void>&,const void*,const void*,void*,int,int,int,float,void*,size_t,cudaStream_t,bool) noexcept{++operators;return CUBLAS_STATUS_NOT_INITIALIZED;}
cublasStatus_t RunModelOptNVFP4MatmulContext(const std::shared_ptr<void>&,const void*,const void*,const void*,const void*,void*,int,int,int,float,void*,size_t,cudaStream_t,bool) noexcept{++operators;return CUBLAS_STATUS_NOT_INITIALIZED;}
cudaError_t QuantizeModelOptFP8Activation(const void*,unsigned char*,float,int,int,cudaStream_t){++operators;return cudaErrorInvalidValue;}
cudaError_t QuantizeModelOptNVFP4Activation(const void*,unsigned char*,unsigned char*,float,int,int,cudaStream_t){++operators;return cudaErrorInvalidValue;}
size_t ModelOptNVFP4ScaleBytes(int rows,int columns){return rows>0&&columns>0&&columns%16==0?size_t((rows+127)/128)*128*size_t(((columns/16)+3)/4)*4:0;}
}
