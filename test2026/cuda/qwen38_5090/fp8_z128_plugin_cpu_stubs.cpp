// CPU-only link stubs. Never link this file into Garnet or a GPU helper.
#include "modelopt_fp8_z128_canonical.h"
#include "modelopt_fp8_m128_canonical.h"
#include "modelopt_cublaslt.h"
#include "modelopt_fp8.cuh"
namespace Garnet {
namespace {int calls=0;}
int FP8Z128CPUUnexpectedCalls() noexcept{return calls;}
std::shared_ptr<ModelOptFP8Z128Context> CreateModelOptFP8Z128Context(){++calls;return {};}
bool GetModelOptFP8Z128Info(const std::shared_ptr<ModelOptFP8Z128Context>&,ModelOptFP8Z128Info&) noexcept{++calls;return false;}
cublasStatus_t RunModelOptFP8Z128Matmul(const std::shared_ptr<ModelOptFP8Z128Context>&,const void*,const void*,void*,int,int,int,float,void*,size_t,cudaStream_t) noexcept{++calls;return CUBLAS_STATUS_NOT_INITIALIZED;}
std::shared_ptr<ModelOptFP8M128Context> CreateModelOptFP8M128Context(){++calls;return {};}
bool GetModelOptFP8M128Info(const std::shared_ptr<ModelOptFP8M128Context>&,ModelOptFP8M128Info&) noexcept{++calls;return false;}
cublasStatus_t RunModelOptFP8M128Matmul(const std::shared_ptr<ModelOptFP8M128Context>&,const void*,const void*,void*,int,int,int,float,void*,size_t,cudaStream_t) noexcept{++calls;return CUBLAS_STATUS_NOT_INITIALIZED;}
std::shared_ptr<void> CreateModelOptNVFP4Context(){++calls;return {};}
cublasStatus_t RunModelOptFP8MatmulContext(const std::shared_ptr<void>&,const void*,const void*,void*,int,int,int,float,void*,size_t,cudaStream_t,bool) noexcept{++calls;return CUBLAS_STATUS_NOT_INITIALIZED;}
cudaError_t QuantizeModelOptFP8Activation(const void*,unsigned char*,float,int,int,cudaStream_t){++calls;return cudaErrorInvalidValue;}
}
