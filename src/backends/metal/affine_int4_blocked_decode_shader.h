// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace Garnet {
// Reuse each input vector across adjacent output channels. Every output keeps
// the original word/lane order, per-weight dequantization FMA and SIMD sum.
// The including library provides GARNET_BLOCKED_ROWS (function constant0).
inline constexpr const char* kAffineInt4BlockedDecodeShader=R"METAL(
struct BlockedQ4Params {ulong rows,inner,columns,aStride,oStride;};
kernel void affine_q4_blocked(device const float* a [[buffer(0)]],
 device const uint* packed [[buffer(1)]],device const ushort* scales [[buffer(2)]],
 device const ushort* biases [[buffer(3)]],device float* out [[buffer(4)]],
 constant BlockedQ4Params& runtime [[buffer(5)]],
#ifdef GARNET_BLOCKED_INDEXED
 constant uint* indices [[buffer(6)]],
#endif
 uint group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],
 uint simd [[simdgroup_index_in_threadgroup]]) {
    GARNET_AFFINE_PARAMETERS(BlockedQ4Params);
    const AffineIndex first=(AffineIndex(group)*4+simd)*GARNET_BLOCKED_COLUMNS;
    if(first>=p.columns)return;
    AffineIndex physical[4];float sums[4][GARNET_BLOCKED_COLUMNS];
#pragma clang loop unroll(full)
    for(uint row=0;row<4;++row) {
#ifdef GARNET_BLOCKED_INDEXED
        physical[row]=row<GARNET_BLOCKED_ROWS?indices[row]:0;
#else
        physical[row]=row;
#endif
#pragma clang loop unroll(full)
        for(uint column=0;column<GARNET_BLOCKED_COLUMNS;++column)sums[row][column]=0;
    }
    for(AffineIndex word=lane;word<p.inner/8;word+=32) {
        float4 low[4],high[4];
#pragma clang loop unroll(full)
        for(uint row=0;row<4;++row)if(row<GARNET_BLOCKED_ROWS) {
            const AffineIndex base=physical[row]*p.aStride+word*8;
            // packed_float4 has scalar alignment, including row-padded views.
            low[row]=float4(*reinterpret_cast<device const packed_float4*>(a+base));
            high[row]=float4(*reinterpret_cast<device const packed_float4*>(a+base+4));
        }
#pragma clang loop unroll(full)
        for(uint column=0;column<GARNET_BLOCKED_COLUMNS;++column)if(first+column<p.columns) {
            const AffineIndex n=first+column,coefficient=n*(p.inner/64)+word/8;
            const uint values=packed[n*(p.inner/8)+word];
            const float scale=garnet_affine_coefficient(scales[coefficient]);
            const float bias=garnet_affine_coefficient(biases[coefficient]);
#pragma clang loop unroll(full)
            for(uint j=0;j<8;++j) {
                const float weight=fma(float((values>>(4*j))&15u),scale,bias);
#pragma clang loop unroll(full)
                for(uint row=0;row<4;++row)if(row<GARNET_BLOCKED_ROWS)
                    sums[row][column]=fma(j<4?low[row][j]:high[row][j-4],weight,sums[row][column]);
            }
        }
    }
#pragma clang loop unroll(full)
    for(uint column=0;column<GARNET_BLOCKED_COLUMNS;++column)if(first+column<p.columns) {
#pragma clang loop unroll(full)
        for(uint row=0;row<4;++row)if(row<GARNET_BLOCKED_ROWS) {
            const float total=simd_sum(sums[row][column]);
            if(lane==0)out[physical[row]*p.oStride+first+column]=total;
        }
    }
}
)METAL";
}
