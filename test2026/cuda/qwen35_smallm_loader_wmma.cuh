// Unwired component candidate: original BF16 inputs/weights, FP32 WMMA sum.
// Dedicated loader warps prefetch the next shared tile while compute warps
// consume the current tile. No codec, quantization, normalization or epilogue
// arithmetic is fused into this candidate. Correctness/performance UNPROVEN.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace Qwen35SmallM {

__device__ __forceinline__ void Copy16(void* destination,const void* source,int bytes) {
    unsigned shared=static_cast<unsigned>(__cvta_generic_to_shared(destination));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(shared),"l"(source),"r"(bytes):"memory");
}
__device__ __forceinline__ void CommitCopies() {
    asm volatile("cp.async.commit_group;":::"memory");
}
__device__ __forceinline__ void WaitCopies() {
    asm volatile("cp.async.wait_group 0;":::"memory");
}

template<int Tokens,int TileK>
__device__ void Load(const __nv_bfloat16* input,const __nv_bfloat16* weight,
                    __nv_bfloat16* stage,int rows,int inner,int columns,int nbase,int kbase,
                    int worker,int workers) {
    constexpr int aElements=16*TileK;
    constexpr int vectors=(16+Tokens)*TileK/8;
    for(int vector=worker;vector<vectors;vector+=workers) {
        bool isA=vector<aElements/8;
        int element=isA?vector*8:(vector-aElements/8)*8;
        int row=element/TileK,column=element%TileK;
        bool valid=isA?(nbase+row<columns):(row<rows);
        const __nv_bfloat16* source=valid?
            (isA?weight+static_cast<size_t>(nbase+row)*inner+kbase+column:
                 input+static_cast<size_t>(row)*inner+kbase+column):input;
        Copy16(stage+vector*8,source,valid?16:0);
    }
    CommitCopies();
}

template<int Tokens,int TileK,int Warps>
__global__ void Matrix(const __nv_bfloat16* input,const __nv_bfloat16* weight,
                       __nv_bfloat16* output,int rows,int inner,int columns) {
    static_assert(Tokens==16||Tokens==32);
    static_assert(TileK==32||TileK==64||TileK==128);
    constexpr int computeWarps=Tokens/16;
    static_assert(Warps>computeWarps);
    constexpr int stageElements=(16+Tokens)*TileK;
    extern __shared__ __align__(32) unsigned char memory[];
    auto* stages=reinterpret_cast<__nv_bfloat16*>(memory);
    auto* result=reinterpret_cast<float*>(stages+2*stageElements);
    int warp=threadIdx.x/32,nbase=blockIdx.x*16;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator,16,16,16,float> accumulator;
    if(warp<computeWarps)nvcuda::wmma::fill_fragment(accumulator,0.0f);
    Load<Tokens,TileK>(input,weight,stages,rows,inner,columns,nbase,0,threadIdx.x,Warps*32);
    WaitCopies();__syncthreads();
    int stageIndex=0;
    for(int kbase=0;kbase<inner;kbase+=TileK) {
        bool next=kbase+TileK<inner;
        if(warp>=computeWarps&&next)
            Load<Tokens,TileK>(input,weight,stages+(1-stageIndex)*stageElements,
                rows,inner,columns,nbase,kbase+TileK,threadIdx.x-computeWarps*32,(Warps-computeWarps)*32);
        if(warp<computeWarps) {
            auto* a=stages+stageIndex*stageElements;
            auto* b=a+16*TileK+warp*16*TileK;
            #pragma unroll
            for(int sub=0;sub<TileK;sub+=16) {
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,16,16,16,__nv_bfloat16,nvcuda::wmma::row_major> af;
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,16,16,16,__nv_bfloat16,nvcuda::wmma::col_major> bf;
                nvcuda::wmma::load_matrix_sync(af,a+sub,TileK);
                nvcuda::wmma::load_matrix_sync(bf,b+sub,TileK);
                nvcuda::wmma::mma_sync(accumulator,af,bf,accumulator);
            }
        }
        if(warp>=computeWarps&&next)WaitCopies();
        __syncthreads();stageIndex=1-stageIndex;
    }
    if(warp<computeWarps)
        nvcuda::wmma::store_matrix_sync(result+warp*16,accumulator,Tokens,nvcuda::wmma::mem_row_major);
    __syncthreads();
    for(int i=threadIdx.x;i<16*Tokens;i+=Warps*32) {
        int row=i/16,n=i%16;
        if(row<rows&&nbase+n<columns)
            output[static_cast<size_t>(row)*columns+nbase+n]=__float2bfloat16_rn(result[n*Tokens+row]);
    }
}

template<int Tokens,int TileK,int Warps>
inline cudaError_t Run(const __nv_bfloat16* input,const __nv_bfloat16* weight,
                      __nv_bfloat16* output,int rows,int inner,int columns,cudaStream_t stream) {
    if(rows<1||rows>Tokens||inner<1||inner%TileK||columns<1||columns%16)return cudaErrorInvalidValue;
    constexpr size_t shared=2*(16+Tokens)*TileK*sizeof(__nv_bfloat16)+16*Tokens*sizeof(float);
    Matrix<Tokens,TileK,Warps><<<(columns+15)/16,Warps*32,shared,stream>>>(input,weight,output,rows,inner,columns);
    return cudaPeekAtLastError();
}

} // namespace Qwen35SmallM
