// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cstdint>
#include <memory>
#include <new>

namespace Garnet { namespace {
// Experimental SM89-only MLP-down prefill configuration. No weight copies or
// caller scratch. All descriptors/algorithms are fixed before graph capture.
struct PrefillLtProgram {
    cublasLtMatmulDesc_t operation=nullptr;
    cublasLtMatrixLayout_t weight=nullptr,input=nullptr,output=nullptr;
    cublasLtMatmulAlgo_t algorithm{};
    ~PrefillLtProgram(){
        if(output)cublasLtMatrixLayoutDestroy(output);
        if(input)cublasLtMatrixLayoutDestroy(input);
        if(weight)cublasLtMatrixLayoutDestroy(weight);
        if(operation)cublasLtMatmulDescDestroy(operation);
    }
    bool initialize(cublasLtHandle_t handle,int rows) noexcept {
        if(rows!=256&&rows!=512)return false;
        if(cublasLtMatmulDescCreate(&operation,CUBLAS_COMPUTE_32F,CUDA_R_32F)!=CUBLAS_STATUS_SUCCESS)return false;
        cublasOperation_t trans=CUBLAS_OP_T;
        if(cublasLtMatmulDescSetAttribute(operation,CUBLASLT_MATMUL_DESC_TRANSA,&trans,sizeof(trans))!=CUBLAS_STATUS_SUCCESS)return false;
        if(cublasLtMatrixLayoutCreate(&weight,CUDA_R_16BF,9216,2560,9216)!=CUBLAS_STATUS_SUCCESS||
           cublasLtMatrixLayoutCreate(&input,CUDA_R_16BF,9216,rows,9216)!=CUBLAS_STATUS_SUCCESS||
           cublasLtMatrixLayoutCreate(&output,CUDA_R_16BF,2560,rows,2560)!=CUBLAS_STATUS_SUCCESS)return false;
        if(cublasLtMatmulAlgoInit(handle,CUBLAS_COMPUTE_32F,CUDA_R_32F,CUDA_R_16BF,CUDA_R_16BF,
                                CUDA_R_16BF,CUDA_R_16BF,21,&algorithm)!=CUBLAS_STATUS_SUCCESS)return false;
        uint64_t flags=0;size_t written=0;
        if(cublasLtMatmulAlgoCapGetAttribute(&algorithm,CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS,
                                            &flags,sizeof(flags),&written)!=CUBLAS_STATUS_SUCCESS||written!=sizeof(flags))return false;
        if((flags&CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_TYPE_MASK)!=CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32F||
           (flags&CUBLASLT_NUMERICAL_IMPL_FLAGS_OP_INPUT_TYPE_MASK)!=CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_16BF)return false;
        int tile=15,stages=12,split=1;uint32_t reduction=CUBLASLT_REDUCTION_SCHEME_NONE;
        if(cublasLtMatmulAlgoConfigSetAttribute(&algorithm,CUBLASLT_ALGO_CONFIG_TILE_ID,&tile,sizeof(tile))!=CUBLAS_STATUS_SUCCESS||
           cublasLtMatmulAlgoConfigSetAttribute(&algorithm,CUBLASLT_ALGO_CONFIG_STAGES_ID,&stages,sizeof(stages))!=CUBLAS_STATUS_SUCCESS||
           cublasLtMatmulAlgoConfigSetAttribute(&algorithm,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,&split,sizeof(split))!=CUBLAS_STATUS_SUCCESS||
           cublasLtMatmulAlgoConfigSetAttribute(&algorithm,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,&reduction,sizeof(reduction))!=CUBLAS_STATUS_SUCCESS)return false;
        cublasLtMatmulHeuristicResult_t result{};
        return cublasLtMatmulAlgoCheck(handle,operation,weight,input,output,output,&algorithm,&result)==CUBLAS_STATUS_SUCCESS&&
               result.state==CUBLAS_STATUS_SUCCESS&&result.workspaceSize==0;
    }
};
struct PrefillLtState {
    cublasLtHandle_t handle=nullptr;
    std::unique_ptr<PrefillLtProgram> programs[2];
    ~PrefillLtState(){
        programs[0].reset();programs[1].reset();
        if(handle)cublasLtDestroy(handle);
    }
    bool initialize() noexcept {
        int device=-1;cudaDeviceProp property{};
        return cudaGetDevice(&device)==cudaSuccess&&cudaGetDeviceProperties(&property,device)==cudaSuccess&&
               property.major==8&&property.minor==9&&cublasLtCreate(&handle)==CUBLAS_STATUS_SUCCESS;
    }
    int run(const void* x,const void* w,void* y,int rows,cudaStream_t stream) noexcept {
        if(!handle||(rows!=256&&rows!=512)||!x||!w||!y||
           ((reinterpret_cast<uintptr_t>(x)|reinterpret_cast<uintptr_t>(w)|reinterpret_cast<uintptr_t>(y))&15))return 1;
        auto& program=programs[rows==256?0:1];
        if(!program){
            cudaStreamCaptureStatus capture;
            if(cudaStreamIsCapturing(stream,&capture)!=cudaSuccess||capture!=cudaStreamCaptureStatusNone)return 1;
            auto candidate=std::unique_ptr<PrefillLtProgram>(new(std::nothrow) PrefillLtProgram);
            if(!candidate||!candidate->initialize(handle,rows))return 1;
            program=std::move(candidate);
        }
        const float alpha=1,beta=0;
        return cublasLtMatmul(handle,program->operation,&alpha,w,program->weight,x,program->input,
            &beta,y,program->output,y,program->output,&program->algorithm,nullptr,0,stream)==CUBLAS_STATUS_SUCCESS?0:1;
    }
};
}}
