// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace Garnet {
inline constexpr const char* kFusedQ4DecodeShader=R"METAL(
#include <metal_stdlib>
using namespace metal;
struct IndexedDecodeQ4Params{ulong rows,inner,columns,aStride,oStride;};
constant uint selected_decode_rows [[function_constant(0)]];
// Experimental FP32 Q4 gate/up/SiLU decode. No half operand conversion.
constant bool fused_shadow [[function_constant(20)]];
constant uint fused_columns [[function_constant(21)]];
kernel void fused_q4_decode(device const float* a [[buffer(0)]],
 device const uint* gp [[buffer(1)]],device const ushort* gs [[buffer(2)]],
 device const ushort* gb [[buffer(3)]],device const uint* up [[buffer(4)]],
 device const ushort* us [[buffer(5)]],device const ushort* ub [[buffer(6)]],
 device float* output [[buffer(7)]],constant IndexedDecodeQ4Params& runtime [[buffer(8)]],
 constant uint* indices [[buffer(9)]],device float* shadow_g [[buffer(10)]],
 device float* shadow_u [[buffer(11)]],uint group [[threadgroup_position_in_grid]],
 uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    GARNET_AFFINE_PARAMETERS(IndexedDecodeQ4Params);
    const AffineIndex first=(AffineIndex(group)*4+simd)*fused_columns;
    if(first>=p.columns)return;
    // Maximum4 columns/4 rows. Function constants eliminate unused members.
    float gate[4][4],upper[4][4];AffineIndex physical[4];
#pragma clang loop unroll(full)
    for(uint row=0;row<4;++row) {
        physical[row]=row<selected_decode_rows?indices[row]:0;
#pragma clang loop unroll(full)
        for(uint column=0;column<4;++column){gate[row][column]=0;upper[row][column]=0;}
    }
    for(AffineIndex word=lane;word<p.inner/8;word+=32) {
        float4 low[4],high[4];
#pragma clang loop unroll(full)
        for(uint row=0;row<4;++row)if(row<selected_decode_rows) {
            const AffineIndex base=physical[row]*p.aStride+word*8;
            low[row]=float4(*reinterpret_cast<device const packed_float4*>(a+base));
            high[row]=float4(*reinterpret_cast<device const packed_float4*>(a+base+4));
        }
#pragma clang loop unroll(full)
        for(uint column=0;column<4;++column)if(column<fused_columns&&first+column<p.columns) {
            const AffineIndex n=first+column,coefficient=n*(p.inner/64)+word/8;
            const uint gateword=gp[n*(p.inner/8)+word],upword=up[n*(p.inner/8)+word];
            const float gscale=garnet_affine_coefficient(gs[coefficient]),gbias=garnet_affine_coefficient(gb[coefficient]);
            const float uscale=garnet_affine_coefficient(us[coefficient]),ubias=garnet_affine_coefficient(ub[coefficient]);
#pragma clang loop unroll(full)
            for(uint j=0;j<8;++j) {
                const float gw=fma(float((gateword>>(4*j))&15u),gscale,gbias);
                const float uw=fma(float((upword>>(4*j))&15u),uscale,ubias);
#pragma clang loop unroll(full)
                for(uint row=0;row<4;++row)if(row<selected_decode_rows) {
                    const float value=j<4?low[row][j]:high[row][j-4];
                    gate[row][column]=fma(value,gw,gate[row][column]);
                    upper[row][column]=fma(value,uw,upper[row][column]);
                }
            }
        }
    }
#pragma clang loop unroll(full)
    for(uint column=0;column<4;++column)if(column<fused_columns&&first+column<p.columns) {
#pragma clang loop unroll(full)
        for(uint row=0;row<4;++row)if(row<selected_decode_rows) {
            const float g=simd_sum(gate[row][column]),u=simd_sum(upper[row][column]);
            if(lane==0) {
                const AffineIndex at=physical[row]*p.oStride+first+column;
                output[at]=(g/(1.0f+FUSED_EXP(-g)))*u;
                if(fused_shadow){shadow_g[at]=g;shadow_u[at]=u;}
            }
        }
    }
}
)METAL";
}
