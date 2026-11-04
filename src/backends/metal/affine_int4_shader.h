// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace Garnet {
inline constexpr const char* kAffineInt4Shader=R"METAL(
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
struct Q4Params {ulong rows,inner,columns,aStride,oStride;};
constant uint affine_decode_rows [[function_constant(0)]];
inline float q4_bf16(ushort value){return as_type<float>(uint(value)<<16);}
kernel void affine_q4_decode(device const float* a [[buffer(0)]],
 device const uint* packed [[buffer(1)]],device const ushort* scales [[buffer(2)]],
 device const ushort* biases [[buffer(3)]],device float* out [[buffer(4)]],
 constant Q4Params& runtime [[buffer(5)]],uint group [[threadgroup_position_in_grid]],
 uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    GARNET_AFFINE_PARAMETERS(Q4Params);
    const ulong n=ulong(group)*4+simd;
    if(n>=p.columns)return;
    float sum0=0,sum1=0,sum2=0,sum3=0;
    for(ulong word=lane;word<p.inner/8;word+=32) {
        const uint values=packed[n*(p.inner/8)+word];
        const ulong coefficient=n*(p.inner/64)+word/8;
        const float scale=q4_bf16(scales[coefficient]),bias=q4_bf16(biases[coefficient]);
        for(uint j=0;j<8;++j) {
            const float weight=fma(float((values>>(4*j))&15u),scale,bias);
            const ulong k=word*8+j;
            sum0=fma(a[k],weight,sum0);
            if(affine_decode_rows>1)sum1=fma(a[p.aStride+k],weight,sum1);
            if(affine_decode_rows>2)sum2=fma(a[2*p.aStride+k],weight,sum2);
            if(affine_decode_rows>3)sum3=fma(a[3*p.aStride+k],weight,sum3);
        }
    }
    float total=simd_sum(sum0);if(lane==0)out[n]=total;
    if(affine_decode_rows>1){total=simd_sum(sum1);if(lane==0)out[p.oStride+n]=total;}
    if(affine_decode_rows>2){total=simd_sum(sum2);if(lane==0)out[2*p.oStride+n]=total;}
    if(affine_decode_rows>3){total=simd_sum(sum3);if(lane==0)out[3*p.oStride+n]=total;}
}
kernel void affine_q4_gemm(device const float* a [[buffer(0)]],
 device const uint* packed [[buffer(1)]],device const ushort* scales [[buffer(2)]],
 device const ushort* biases [[buffer(3)]],device float* out [[buffer(4)]],
 constant Q4Params& runtime [[buffer(5)]],uint2 group [[threadgroup_position_in_grid]],
 uint tid [[thread_index_in_threadgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    GARNET_AFFINE_PARAMETERS(Q4Params);
    // The padded B stride distributes transposed stores across memory banks.
    // One thread owns one packed word (eight K values) of one output column.
    threadgroup float atile[16*32],btile[32*33],result[16*32];
    simdgroup_float8x8 acc0=simdgroup_float8x8(0.0f),acc1=simdgroup_float8x8(0.0f);
    for(ulong base=0;base<p.inner;base+=32) {
        for(uint i=tid;i<16*32;i+=128){const ulong row=ulong(group.y)*16+i/32,k=base+i%32;
            atile[i]=(row<p.rows&&k<p.inner)?a[row*p.aStride+k]:0;}
        const uint localN=tid/4,localK=(tid%4)*8;
        const ulong n=ulong(group.x)*32+localN,k=base+localK;
        uint values=0;float scale=0,bias=0;
        if(n<p.columns&&k<p.inner) {
            const ulong coefficient=n*(p.inner/64)+k/64;
            values=packed[n*(p.inner/8)+k/8];
            scale=q4_bf16(scales[coefficient]);bias=q4_bf16(biases[coefficient]);
        }
        for(uint j=0;j<8;++j) {
            btile[(localK+j)*33+localN]=(n<p.columns&&k+j<p.inner)?
                fma(float((values>>(4*j))&15u),scale,bias):0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint k=0;k<32;k+=8) {
            simdgroup_float8x8 x0,x1,w;
            simdgroup_load(x0,atile+k,32);simdgroup_load(x1,atile+8*32+k,32);
            simdgroup_load(w,btile+k*33+simd*8,33);
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
