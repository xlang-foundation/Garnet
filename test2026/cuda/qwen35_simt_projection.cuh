// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace Qwen35SIMT {
template<bool Streaming> __device__ __forceinline__ uint4 weightVector(const uint4* pointer){
    if constexpr(Streaming)return __ldcs(pointer);
    else return __ldcg(pointer);
}
__device__ __forceinline__ unsigned word(uint4 value,int index){
    return index==0?value.x:index==1?value.y:index==2?value.z:value.w;
}
// BF16-to-FP32 conversion is an exact bit expansion, with no intermediate cast.
__device__ __forceinline__ float low(unsigned packed){return __uint_as_float(packed<<16);}
__device__ __forceinline__ float high(unsigned packed){return __uint_as_float(packed&0xffff0000u);}

template<int TileRows,int Warps,int Accumulators,bool Streaming>
__global__ __launch_bounds__(Warps*32,2) void project(const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weights,__nv_bfloat16* __restrict__ output,int rows,int inner,int columns){
    int lane=threadIdx.x&31,column=blockIdx.x*Warps+threadIdx.x/32,base=blockIdx.y*TileRows;
    if(column>=columns)return; // Entire warp shares the same branch.
    float sums[TileRows][Accumulators]={};
    for(int k=lane*8;k<inner;k+=256){
        uint4 packedWeight=weightVector<Streaming>(reinterpret_cast<const uint4*>(weights+size_t(column)*inner+k));
        float w[8];
        #pragma unroll
        for(int p=0;p<4;++p){unsigned bits=word(packedWeight,p);w[2*p]=low(bits);w[2*p+1]=high(bits);}
        #pragma unroll
        for(int r=0;r<TileRows;++r){
            uint4 packedInput=__ldg(reinterpret_cast<const uint4*>(input+size_t(base+r)*inner+k));
            #pragma unroll
            for(int p=0;p<4;++p){
                unsigned bits=word(packedInput,p);constexpr int mask=Accumulators-1;
                sums[r][p&mask]=__fmaf_rn(low(bits),w[2*p],sums[r][p&mask]);
                sums[r][p&mask]=__fmaf_rn(high(bits),w[2*p+1],sums[r][p&mask]);
            }
        }
    }
    #pragma unroll
    for(int r=0;r<TileRows;++r){
        float value=sums[r][0];if constexpr(Accumulators==2)value=__fadd_rn(value,sums[r][1]);
        #pragma unroll
        for(int offset=16;offset;offset/=2)value=__fadd_rn(value,__shfl_down_sync(0xffffffffu,value,offset));
        if(lane==0)output[size_t(base+r)*columns+column]=__float2bfloat16_rn(value);
    }
}
inline bool overlap(std::uintptr_t a,size_t an,std::uintptr_t b,size_t bn){return a<=b?b-a<an:a-b<bn;}
inline cudaError_t Run(int configuration,const void* input,const void* weights,void* output,
    int rows,int inner,int columns,cudaStream_t stream){
    int tile=configuration>>16,warps=(configuration>>8)&255,accumulators=(configuration>>1)&127;
    bool streaming=(configuration&1)!=0;
    if(!input||!weights||!output||(rows!=4&&rows!=8&&rows!=16&&rows!=32)||inner<256||inner>16384||inner%256||
       columns<1||columns>32768||tile<4||tile>32||tile>rows||rows%tile||
       (reinterpret_cast<std::uintptr_t>(input)&15)||(reinterpret_cast<std::uintptr_t>(weights)&15)||(reinterpret_cast<std::uintptr_t>(output)&1))return cudaErrorInvalidValue;
    auto x=reinterpret_cast<std::uintptr_t>(input),w=reinterpret_cast<std::uintptr_t>(weights),y=reinterpret_cast<std::uintptr_t>(output);
    if(overlap(x,size_t(rows)*inner*2,y,size_t(rows)*columns*2)||overlap(w,size_t(columns)*inner*2,y,size_t(rows)*columns*2))return cudaErrorInvalidValue;
    auto* a=static_cast<const __nv_bfloat16*>(input);auto* b=static_cast<const __nv_bfloat16*>(weights);auto* c=static_cast<__nv_bfloat16*>(output);
    #define SIMT_CASE(T,W,A) if(tile==T&&warps==W&&accumulators==A){dim3 grid((columns+W-1)/W,rows/T); \
        if(streaming)project<T,W,A,true><<<grid,W*32,0,stream>>>(a,b,c,rows,inner,columns); \
        else project<T,W,A,false><<<grid,W*32,0,stream>>>(a,b,c,rows,inner,columns);return cudaGetLastError();}
    SIMT_CASE(4,4,1);SIMT_CASE(4,4,2);SIMT_CASE(4,8,1);SIMT_CASE(4,8,2);
    SIMT_CASE(8,4,1);SIMT_CASE(8,4,2);SIMT_CASE(8,8,1);SIMT_CASE(8,8,2);
    SIMT_CASE(16,4,1);SIMT_CASE(16,4,2);SIMT_CASE(16,8,1);SIMT_CASE(16,8,2);
    SIMT_CASE(32,4,1);SIMT_CASE(32,4,2);SIMT_CASE(32,8,1);SIMT_CASE(32,8,2);
    #undef SIMT_CASE
    return cudaErrorInvalidValue;
}
}
