// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace Garnet {
// Append after the baseline Q4 shader: preserve its coefficient semantics and
// per-output K accumulation order while amortizing B unpacking over more rows.
inline constexpr const char* kAffineInt4TallShader=R"METAL(
template<uint M>
inline void affine_q4_tall(device const float* a,device const uint* packed,
 device const ushort* scales,device const ushort* biases,device float* out,
 constant Q4Params& p,uint2 group,uint tid,uint simd,
 threadgroup float* atile,threadgroup float* btile,threadgroup float* result) {
    simdgroup_float8x8 acc[M/8];
    #pragma unroll
    for(uint r=0;r<M/8;++r)acc[r]=simdgroup_float8x8(0.0f);
    for(ulong base=0;base<p.inner;base+=32) {
        for(uint i=tid;i<M*32;i+=128) {
            const ulong row=ulong(group.y)*M+i/32,k=base+i%32;
            atile[i]=(row<p.rows&&k<p.inner)?a[row*p.aStride+k]:0;
        }
        const uint localN=tid/4,localK=(tid%4)*8;
        const ulong n=ulong(group.x)*32+localN,k=base+localK;
        uint values=0;float scale=0,bias=0;
        if(n<p.columns&&k<p.inner) {
            const ulong coefficient=n*(p.inner/64)+k/64;
            values=packed[n*(p.inner/8)+k/8];
            scale=q4_bf16(scales[coefficient]);bias=q4_bf16(biases[coefficient]);
        }
        #pragma unroll
        for(uint j=0;j<8;++j)btile[(localK+j)*33+localN]=(n<p.columns&&k+j<p.inner)?
            fma(float((values>>(4*j))&15u),scale,bias):0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint k=0;k<32;k+=8) {
            simdgroup_float8x8 w;simdgroup_load(w,btile+k*33+simd*8,33);
            #pragma unroll
            for(uint r=0;r<M/8;++r) {
                simdgroup_float8x8 x;simdgroup_load(x,atile+r*8*32+k,32);
                simdgroup_multiply_accumulate(acc[r],x,w,acc[r]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    #pragma unroll
    for(uint r=0;r<M/8;++r)simdgroup_store(acc[r],result+r*8*32+simd*8,32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<M*32;i+=128) {
        const ulong row=ulong(group.y)*M+i/32,n=ulong(group.x)*32+i%32;
        if(row<p.rows&&n<p.columns)out[row*p.oStride+n]=result[i];
    }
}
#define Q4_TALL_ENTRY(NAME,M) \
kernel void NAME(device const float* a [[buffer(0)]],device const uint* packed [[buffer(1)]], \
 device const ushort* scales [[buffer(2)]],device const ushort* biases [[buffer(3)]], \
 device float* out [[buffer(4)]],constant Q4Params& p [[buffer(5)]], \
 uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]], \
 uint simd [[simdgroup_index_in_threadgroup]]) { \
 threadgroup float atile[M*32],btile[32*33],result[M*32]; \
 affine_q4_tall<M>(a,packed,scales,biases,out,p,group,tid,simd,atile,btile,result); \
}
Q4_TALL_ENTRY(affine_q4_gemm32,32)
Q4_TALL_ENTRY(affine_q4_gemm64,64)
#undef Q4_TALL_ENTRY
)METAL";
}
