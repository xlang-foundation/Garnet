// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace Garnet {
// Preserve ordinary decode's FP32 operands, FMA order and SIMD reduction.
// Only logical row addresses change; selected rows retain their physical slots.
inline constexpr const char* kAffineInt4IndexedDecodeShader=R"METAL(
#include <metal_stdlib>
using namespace metal;
struct IndexedDecodeQ4Params {ulong rows,inner,columns,aStride,oStride;};
constant uint selected_decode_rows [[function_constant(0)]];
inline float indexed_decode_bf16(ushort value){return as_type<float>(uint(value)<<16);}
kernel void affine_q4_indexed_decode(device const float* a [[buffer(0)]],
 device const uint* packed [[buffer(1)]],device const ushort* scales [[buffer(2)]],
 device const ushort* biases [[buffer(3)]],device float* out [[buffer(4)]],
 constant IndexedDecodeQ4Params& p [[buffer(5)]],constant uint* indices [[buffer(6)]],
 uint group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],
 uint simd [[simdgroup_index_in_threadgroup]]) {
    const ulong n=ulong(group)*4+simd;
    if(n>=p.columns)return;
    const ulong r0=indices[0],r1=selected_decode_rows>1?indices[1]:0;
    const ulong r2=selected_decode_rows>2?indices[2]:0,r3=selected_decode_rows>3?indices[3]:0;
    float sum0=0,sum1=0,sum2=0,sum3=0;
    for(ulong word=lane;word<p.inner/8;word+=32) {
        const uint values=packed[n*(p.inner/8)+word];
        const ulong coefficient=n*(p.inner/64)+word/8;
        const float scale=indexed_decode_bf16(scales[coefficient]),bias=indexed_decode_bf16(biases[coefficient]);
        for(uint j=0;j<8;++j) {
            const float weight=fma(float((values>>(4*j))&15u),scale,bias);
            const ulong k=word*8+j;
            sum0=fma(a[r0*p.aStride+k],weight,sum0);
            if(selected_decode_rows>1)sum1=fma(a[r1*p.aStride+k],weight,sum1);
            if(selected_decode_rows>2)sum2=fma(a[r2*p.aStride+k],weight,sum2);
            if(selected_decode_rows>3)sum3=fma(a[r3*p.aStride+k],weight,sum3);
        }
    }
    float total=simd_sum(sum0);if(lane==0)out[r0*p.oStride+n]=total;
    if(selected_decode_rows>1){total=simd_sum(sum1);if(lane==0)out[r1*p.oStride+n]=total;}
    if(selected_decode_rows>2){total=simd_sum(sum2);if(lane==0)out[r2*p.oStride+n]=total;}
    if(selected_decode_rows>3){total=simd_sum(sum3);if(lane==0)out[r3*p.oStride+n]=total;}
}
)METAL";
}
