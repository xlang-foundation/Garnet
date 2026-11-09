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

#include <metal_stdlib>
using namespace metal;
struct IndexedDecodeQ4Params {ulong rows,inner,columns,aStride,oStride;};
constant uint selected_decode_rows [[function_constant(0)]];
inline float indexed_decode_bf16(ushort value){return garnet_affine_coefficient(value);}
kernel void affine_q4_indexed_decode(device const float* a [[buffer(0)]],
 device const uint* packed [[buffer(1)]],device const ushort* scales [[buffer(2)]],
 device const ushort* biases [[buffer(3)]],device float* out [[buffer(4)]],
 constant IndexedDecodeQ4Params& runtime [[buffer(5)]],constant uint* indices [[buffer(6)]],
 uint group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],
 uint simd [[simdgroup_index_in_threadgroup]]) {
    GARNET_AFFINE_PARAMETERS(IndexedDecodeQ4Params);
    const AffineIndex n=AffineIndex(group)*4+simd;
    if(n>=p.columns)return;
    const AffineIndex r0=indices[0],r1=selected_decode_rows>1?indices[1]:0;
    const AffineIndex r2=selected_decode_rows>2?indices[2]:0,r3=selected_decode_rows>3?indices[3]:0;
    float sum0=0,sum1=0,sum2=0,sum3=0;
    for(AffineIndex word=lane;word<p.inner/8;word+=32) {
        const uint values=packed[n*(p.inner/8)+word];
        const AffineIndex coefficient=n*(p.inner/64)+word/8;
        const float scale=indexed_decode_bf16(scales[coefficient]),bias=indexed_decode_bf16(biases[coefficient]);
        for(uint j=0;j<8;++j) {
            const float weight=fma(float((values>>(4*j))&15u),scale,bias);
            const AffineIndex k=word*8+j;
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

#define GARNET_BLOCKED_COLUMNS 4
#define GARNET_BLOCKED_ROWS selected_decode_rows
#define GARNET_BLOCKED_INDEXED 1

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
