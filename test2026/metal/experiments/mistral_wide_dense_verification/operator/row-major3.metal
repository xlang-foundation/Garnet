#define GARNET_AFFINE_FP16_COEFFICIENTS 1
#define GARNET_AFFINE_PROFILE 1
#define GARNET_AFFINE_INDEX32 1

#include <metal_stdlib>
inline float garnet_affine_coefficient(ushort value) {
#ifdef GARNET_AFFINE_FP16_COEFFICIENTS
    return float(as_type<half>(value));
#else
    return as_type<float>(uint(value)<<16);
#endif
}
#if defined(GARNET_AFFINE_INDEX32) && !defined(GARNET_AFFINE_PROFILE)
#error Bounded_indexes_require_specialized_profiles
#endif
#ifdef GARNET_AFFINE_INDEX32
using AffineIndex=uint;
static_assert(sizeof(AffineIndex)==4,"Compiled affine index width differs");
struct BoundedAffineParams {uint rows,inner,columns,aStride,oStride;};
#else
using AffineIndex=ulong;
#endif
#ifdef GARNET_AFFINE_PROFILE
constant uint affine_profile_inner [[function_constant(10)]];
constant uint affine_profile_columns [[function_constant(11)]];
constant uint affine_profile_a_stride [[function_constant(12)]];
constant uint affine_profile_o_stride [[function_constant(13)]];
#ifdef GARNET_AFFINE_INDEX32
#define GARNET_AFFINE_PARAMETERS(TYPE) const BoundedAffineParams p{uint(runtime.rows),affine_profile_inner,affine_profile_columns,affine_profile_a_stride,affine_profile_o_stride}
#else
#define GARNET_AFFINE_PARAMETERS(TYPE) const TYPE p{runtime.rows,affine_profile_inner,affine_profile_columns,affine_profile_a_stride,affine_profile_o_stride}
#endif
#else
#define GARNET_AFFINE_PARAMETERS(TYPE) constant TYPE& p=runtime
#endif

using namespace metal;
#define GARNET_BLOCKED_COLUMNS 4
#define GARNET_BLOCKED_INDEXED 1
#define GARNET_WIDE_ROW_MAJOR 1

struct BlockedQ4Params {ulong rows,inner,columns,aStride,oStride;};
constant uint wide_selected_rows [[function_constant(0)]];
kernel void affine_q4_wide(device const float* a [[buffer(0)]],
 device const uint* packed [[buffer(1)]],device const ushort* scales [[buffer(2)]],
 device const ushort* biases [[buffer(3)]],device float* out [[buffer(4)]],
 constant BlockedQ4Params& runtime [[buffer(5)]],
 constant uint& selectedCount [[buffer(7)]],
#ifdef GARNET_BLOCKED_INDEXED
 constant uint* indices [[buffer(6)]],
#endif
 uint group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],
 uint simd [[simdgroup_index_in_threadgroup]]) {
    GARNET_AFFINE_PARAMETERS(BlockedQ4Params);
    const uint tiles=(wide_selected_rows+3)/4;
#ifdef GARNET_WIDE_ROW_MAJOR
    const uint columns=(p.columns+4*GARNET_BLOCKED_COLUMNS-1)/(4*GARNET_BLOCKED_COLUMNS);
    const uint rowTile=group/columns,columnTile=group%columns;
#else
    const uint rowTile=group%tiles,columnTile=group/tiles;
#endif
    const uint selectedBase=rowTile*4,live=(rowTile+1==tiles)?(wide_selected_rows-4*(tiles-1)):4;
    const AffineIndex first=(AffineIndex(columnTile)*4+simd)*GARNET_BLOCKED_COLUMNS;
    if(first>=p.columns)return;
    AffineIndex physical[4];float sums[4][GARNET_BLOCKED_COLUMNS];
#pragma clang loop unroll(full)
    for(uint row=0;row<4;++row) {
#ifdef GARNET_BLOCKED_INDEXED
        physical[row]=row<live?indices[selectedBase+row]:0;
#else
        physical[row]=row;
#endif
#pragma clang loop unroll(full)
        for(uint column=0;column<GARNET_BLOCKED_COLUMNS;++column)sums[row][column]=0;
    }
    for(AffineIndex word=lane;word<p.inner/8;word+=32) {
        float4 low[4],high[4];
#pragma clang loop unroll(full)
        for(uint row=0;row<4;++row)if(row<live) {
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
                for(uint row=0;row<4;++row)if(row<live)
                    sums[row][column]=fma(j<4?low[row][j]:high[row][j-4],weight,sums[row][column]);
            }
        }
    }
#pragma clang loop unroll(full)
    for(uint column=0;column<GARNET_BLOCKED_COLUMNS;++column)if(first+column<p.columns) {
#pragma clang loop unroll(full)
        for(uint row=0;row<4;++row)if(row<live) {
            const float total=simd_sum(sums[row][column]);
            if(lane==0)out[physical[row]*p.oStride+first+column]=total;
        }
    }
}
