// CPU-only factory/operator fixtures. Never link into Garnet or SDK controls.
#include "modelopt_fp8_gdn_m16_canonical.h"
#include "modelopt_cublaslt.h"
#include "modelopt_fp8.cuh"
#include <stdexcept>
#include <string>
namespace Garnet {
struct ModelOptFP8GDNM16Context{int rows;explicit ModelOptFP8GDNM16Context(int n):rows(n){}};
namespace {std::string mode;int factories=0,unexpected=0;}
void GDNM16CPUFactoryMode(const char* value){mode=value;}
int GDNM16CPUFactoryCalls(){return factories;}
int GDNM16CPUUnexpectedCalls(){return unexpected;}
std::shared_ptr<ModelOptFP8GDNM16Context> CreateModelOptFP8GDNM16Context(int rows){
    ++factories;if(rows!=10240&&rows!=6144){++unexpected;return {};}
    if(mode=="null")return {};if(mode=="std")throw std::runtime_error("intentional_CPU_factory_failure");if(mode=="unknown")throw 9;
    return std::make_shared<ModelOptFP8GDNM16Context>(rows);
}
bool GetModelOptFP8GDNM16Info(const std::shared_ptr<ModelOptFP8GDNM16Context>&,ModelOptFP8GDNM16Info&) noexcept{++unexpected;return false;}
cublasStatus_t RunModelOptFP8GDNM16Matmul(const std::shared_ptr<ModelOptFP8GDNM16Context>&,const void*,const void*,void*,int,int,int,float,void*,size_t,cudaStream_t) noexcept{++unexpected;return CUBLAS_STATUS_NOT_INITIALIZED;}
std::shared_ptr<void> CreateModelOptNVFP4Context(){++unexpected;return {};}
cublasStatus_t RunModelOptFP8MatmulContext(const std::shared_ptr<void>&,const void*,const void*,void*,int,int,int,float,void*,size_t,cudaStream_t,bool) noexcept{++unexpected;return CUBLAS_STATUS_NOT_INITIALIZED;}
cudaError_t QuantizeModelOptFP8Activation(const void*,unsigned char*,float,int,int,cudaStream_t){++unexpected;return cudaErrorInvalidValue;}
}
