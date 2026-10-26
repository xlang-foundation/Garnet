// Actual native helper compiled in a separate TU; observe calls without changing them.
#include "modelopt_fp8_m128_canonical.h"
cublasStatus_t ObserveMatmul(cublasLtHandle_t handle,
    cublasLtMatmulDesc_t operation, const void* alpha, const void* a,
    cublasLtMatrixLayout_t ad, const void* b, cublasLtMatrixLayout_t bd,
    const void* beta, const void* c, cublasLtMatrixLayout_t cd,
    void* d, cublasLtMatrixLayout_t dd, const cublasLtMatmulAlgo_t* algo,
    void* workspace, size_t workspaceBytes, cudaStream_t stream);
#define cublasLtMatmul ObserveMatmul
#include "modelopt_fp8_m128_canonical.cpp"
#undef cublasLtMatmul
