// Observe the actual typed native helper in a separate TU without substitution.
#include "modelopt_fp8_z128_canonical.h"
cublasStatus_t ObserveMatmul(cublasLtHandle_t handle,
    cublasLtMatmulDesc_t operation,const void* alpha,const void* a,
    cublasLtMatrixLayout_t ad,const void* b,cublasLtMatrixLayout_t bd,
    const void* beta,const void* c,cublasLtMatrixLayout_t cd,
    void* d,cublasLtMatrixLayout_t dd,const cublasLtMatmulAlgo_t* algo,
    void* workspace,size_t workspaceBytes,cudaStream_t stream);
#define cublasLtMatmul ObserveMatmul
#include "modelopt_fp8_z128_canonical.cpp"
#undef cublasLtMatmul
