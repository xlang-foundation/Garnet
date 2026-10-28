// CPU-only interface fixture. Never used in a Garnet/SDK build or linked to CUDA.
#pragma once
#include <cstddef>
#include <cstdint>
using cudaStream_t=void*;
using cublasLtHandle_t=void*;
using cublasLtMatmulDesc_t=void*;
using cublasLtMatrixLayout_t=void*;
using cublasLtMatmulPreference_t=void*;
using cublasStatus_t=int;
using cublasOperation_t=int;
constexpr int CUBLAS_STATUS_SUCCESS=0,CUBLAS_STATUS_NOT_INITIALIZED=1;
constexpr int CUBLAS_STATUS_INVALID_VALUE=7,CUBLAS_STATUS_INTERNAL_ERROR=14;
constexpr int CUBLAS_COMPUTE_32F=68,CUDA_R_32F=0,CUDA_R_8F_E4M3=28,CUDA_R_16BF=14;
constexpr int CUBLAS_OP_T=1,CUBLAS_OP_N=0,CUBLASLT_MATMUL_DESC_TRANSA=0,CUBLASLT_MATMUL_DESC_TRANSB=1;
constexpr int CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES=1,CUBLASLT_ALGO_CONFIG_ID=0;
struct cublasLtMatmulAlgo_t{unsigned char bytes[64];};
struct cublasLtMatmulHeuristicResult_t{cublasLtMatmulAlgo_t algo{};size_t workspaceSize=0;int state=0;float wavesCount=0;};
size_t cublasLtGetVersion();
cublasStatus_t cublasLtCreate(cublasLtHandle_t*);
cublasStatus_t cublasLtDestroy(cublasLtHandle_t);
cublasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t*,int,int);
cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t);
cublasStatus_t cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t,int,const void*,size_t);
cublasStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t*,int,uint64_t,uint64_t,int64_t);
cublasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t);
cublasStatus_t cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t*);
cublasStatus_t cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t);
cublasStatus_t cublasLtMatmulPreferenceSetAttribute(cublasLtMatmulPreference_t,int,const void*,size_t);
cublasStatus_t cublasLtMatmulAlgoGetHeuristic(cublasLtHandle_t,cublasLtMatmulDesc_t,cublasLtMatrixLayout_t,cublasLtMatrixLayout_t,cublasLtMatrixLayout_t,cublasLtMatrixLayout_t,cublasLtMatmulPreference_t,int,cublasLtMatmulHeuristicResult_t*,int*);
cublasStatus_t cublasLtMatmulAlgoConfigGetAttribute(const cublasLtMatmulAlgo_t*,int,void*,size_t,size_t*);
cublasStatus_t cublasLtMatmulAlgoCheck(cublasLtHandle_t,cublasLtMatmulDesc_t,cublasLtMatrixLayout_t,cublasLtMatrixLayout_t,cublasLtMatrixLayout_t,cublasLtMatrixLayout_t,const cublasLtMatmulAlgo_t*,cublasLtMatmulHeuristicResult_t*);
cublasStatus_t cublasLtMatmul(cublasLtHandle_t,cublasLtMatmulDesc_t,const void*,const void*,cublasLtMatrixLayout_t,const void*,cublasLtMatrixLayout_t,const void*,const void*,cublasLtMatrixLayout_t,void*,cublasLtMatrixLayout_t,const cublasLtMatmulAlgo_t*,void*,size_t,cudaStream_t);

using cublasLtMatmulAlgoConfigAttributes_t=int;
constexpr int CUBLASLT_ALGO_CONFIG_TILE_ID=1,CUBLASLT_ALGO_CONFIG_STAGES_ID=2,CUBLASLT_ALGO_CONFIG_SPLITK_NUM=3;
constexpr int CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME=4,CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING=5,CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION=6;
constexpr int CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID=7,CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID=8;
cublasStatus_t cublasLtMatmulAlgoInit(cublasLtHandle_t,int,int,int,int,int,int,int,cublasLtMatmulAlgo_t*);
cublasStatus_t cublasLtMatmulAlgoConfigSetAttribute(cublasLtMatmulAlgo_t*,int,const void*,size_t);

constexpr uint32_t CUBLASLT_MATMUL_TILE_64x32=13,CUBLASLT_MATMUL_STAGES_64x5=17;
constexpr uint32_t CUBLASLT_MATMUL_TILE_64x64=15,CUBLASLT_MATMUL_STAGES_64x6=18,CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE=2;
