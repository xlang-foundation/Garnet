// CPU-only factory/operator fixtures. Never link into Garnet or SDK controls.
#include "modelopt_fp8_k_m4x4_canonical.h"
#include "modelopt_fp8_qk_m16_canonical.h"
#include "modelopt_fp8_v_m16_canonical.h"
#include "modelopt_fp8_gdn_m16_canonical.h"
#include "modelopt_cublaslt.h"
#include "modelopt_fp8.cuh"
#include <stdexcept>
#include <string>
namespace Garnet {
struct ModelOptFP8KM4X4Context{int rows;explicit ModelOptFP8KM4X4Context(int n):rows(n){}};
namespace {std::string mode;int factories=0,unexpected=0;}
void KM4X4CPUFactoryMode(const char* value){mode=value;}
int KM4X4CPUFactoryCalls(){return factories;}
int KM4X4CPUUnexpectedCalls(){return unexpected;}
std::shared_ptr<ModelOptFP8KM4X4Context> CreateModelOptFP8KM4X4Context(int rows){
    ++factories;if(rows!=1024){++unexpected;return {};}
    if(mode=="null")return {};if(mode=="std")throw std::runtime_error("intentional_CPU_factory_failure");if(mode=="unknown")throw 9;
    return std::make_shared<ModelOptFP8KM4X4Context>(rows);
}
bool GetModelOptFP8KM4X4Info(const std::shared_ptr<ModelOptFP8KM4X4Context>&,ModelOptFP8KM4X4Info&) noexcept{++unexpected;return false;}
cublasStatus_t RunModelOptFP8KM4X4Matmul(const std::shared_ptr<ModelOptFP8KM4X4Context>&,const void*,const void*,void*,int,int,int,float,void*,size_t,cudaStream_t) noexcept{++unexpected;return CUBLAS_STATUS_NOT_INITIALIZED;}
struct ModelOptFP8QKM16Context{};
std::shared_ptr<ModelOptFP8QKM16Context> CreateModelOptFP8QKM16Context(int){++unexpected;return {};}
bool GetModelOptFP8QKM16Info(const std::shared_ptr<ModelOptFP8QKM16Context>&,ModelOptFP8QKM16Info&) noexcept{++unexpected;return false;}
cublasStatus_t RunModelOptFP8QKM16Matmul(const std::shared_ptr<ModelOptFP8QKM16Context>&,const void*,const void*,void*,int,int,int,float,void*,size_t,cudaStream_t) noexcept{++unexpected;return CUBLAS_STATUS_NOT_INITIALIZED;}
struct ModelOptFP8VM16Context{};
std::shared_ptr<ModelOptFP8VM16Context> CreateModelOptFP8VM16Context(int){++unexpected;return {};}
bool GetModelOptFP8VM16Info(const std::shared_ptr<ModelOptFP8VM16Context>&,ModelOptFP8VM16Info&) noexcept{++unexpected;return false;}
cublasStatus_t RunModelOptFP8VM16Matmul(const std::shared_ptr<ModelOptFP8VM16Context>&,const void*,const void*,void*,int,int,int,float,void*,size_t,cudaStream_t) noexcept{++unexpected;return CUBLAS_STATUS_NOT_INITIALIZED;}
struct ModelOptFP8GDNM16Context{};
std::shared_ptr<ModelOptFP8GDNM16Context> CreateModelOptFP8GDNM16Context(int){++unexpected;return {};}
bool GetModelOptFP8GDNM16Info(const std::shared_ptr<ModelOptFP8GDNM16Context>&,ModelOptFP8GDNM16Info&) noexcept{++unexpected;return false;}
cublasStatus_t RunModelOptFP8GDNM16Matmul(const std::shared_ptr<ModelOptFP8GDNM16Context>&,const void*,const void*,void*,int,int,int,float,void*,size_t,cudaStream_t) noexcept{++unexpected;return CUBLAS_STATUS_NOT_INITIALIZED;}
std::shared_ptr<void> CreateModelOptNVFP4Context(){++unexpected;return {};}
cublasStatus_t RunModelOptFP8MatmulContext(const std::shared_ptr<void>&,const void*,const void*,void*,int,int,int,float,void*,size_t,cudaStream_t,bool) noexcept{++unexpected;return CUBLAS_STATUS_NOT_INITIALIZED;}
cudaError_t QuantizeModelOptFP8Activation(const void*,unsigned char*,float,int,int,cudaStream_t){++unexpected;return cudaErrorInvalidValue;}
}
