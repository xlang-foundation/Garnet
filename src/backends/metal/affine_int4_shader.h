// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace Garnet {
inline constexpr const char* kAffineInt4Shader=R"METAL(
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
struct Q4Params {ulong rows,inner,columns,aStride,oStride;};
inline float q4_bf16(ushort value){return as_type<float>(uint(value)<<16);}
kernel void affine_q4_decode(device const float* a [[buffer(0)]],
 device const uint* packed [[buffer(1)]],device const ushort* scales [[buffer(2)]],
 device const ushort* biases [[buffer(3)]],device float* out [[buffer(4)]],
 constant Q4Params& p [[buffer(5)]],uint group [[threadgroup_position_in_grid]],
 uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    const ulong n=ulong(group)*4+simd;
    if(n>=p.columns)return;
    float sum[8]={0};
    for(ulong word=lane;word<p.inner/8;word+=32) {
        const uint values=packed[n*(p.inner/8)+word];
        const ulong coefficient=n*(p.inner/64)+word/8;
        const float scale=q4_bf16(scales[coefficient]),bias=q4_bf16(biases[coefficient]);
        for(uint j=0;j<8;++j) {
            const float weight=fma(float((values>>(4*j))&15u),scale,bias);
            for(ulong row=0;row<p.rows;++row)
                sum[row]=fma(a[row*p.aStride+word*8+j],weight,sum[row]);
        }
    }
    for(ulong row=0;row<p.rows;++row){const float total=simd_sum(sum[row]);if(lane==0)out[row*p.oStride+n]=total;}
}
kernel void affine_q4_gemm(device const float* a [[buffer(0)]],
 device const uint* packed [[buffer(1)]],device const ushort* scales [[buffer(2)]],
 device const ushort* biases [[buffer(3)]],device float* out [[buffer(4)]],
 constant Q4Params& p [[buffer(5)]],uint2 group [[threadgroup_position_in_grid]],
 uint tid [[thread_index_in_threadgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float atile[16*32],btile[32*32],result[16*32];
    simdgroup_float8x8 acc0=simdgroup_float8x8(0.0f),acc1=simdgroup_float8x8(0.0f);
    for(ulong base=0;base<p.inner;base+=32) {
        for(uint i=tid;i<16*32;i+=128){const ulong row=ulong(group.y)*16+i/32,k=base+i%32;
            atile[i]=(row<p.rows&&k<p.inner)?a[row*p.aStride+k]:0;}
        for(uint i=tid;i<32*32;i+=128) {
            const ulong n=ulong(group.x)*32+i%32,k=base+i/32;
            float value=0;
            if(n<p.columns&&k<p.inner){const ulong coefficient=n*(p.inner/64)+k/64;
                value=fma(float((packed[n*(p.inner/8)+k/8]>>((k%8)*4))&15u),
                          q4_bf16(scales[coefficient]),q4_bf16(biases[coefficient]));}
            btile[i]=value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint k=0;k<32;k+=8) {
            simdgroup_float8x8 x0,x1,w;
            simdgroup_load(x0,atile+k,32);simdgroup_load(x1,atile+8*32+k,32);
            simdgroup_load(w,btile+k*32+simd*8,32);
            simdgroup_multiply_accumulate(acc0,x0,w,acc0);
            simdgroup_multiply_accumulate(acc1,x1,w,acc1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(acc0,result+simd*8,32);simdgroup_store(acc1,result+8*32+simd*8,32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<16*32;i+=128){const ulong row=ulong(group.y)*16+i/32,n=ulong(group.x)*32+i%32;
        if(row<p.rows&&n<p.columns)out[row*p.oStride+n]=result[i];}
}
)METAL";
}
