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
#include <metal_simdgroup_matrix>
using namespace metal;
struct Q4Params {ulong rows,inner,columns,aStride,oStride;};
constant uint affine_decode_rows [[function_constant(0)]];
inline float q4_bf16(ushort value){return garnet_affine_coefficient(value);}
kernel void affine_q4_decode(device const float* a [[buffer(0)]],
 device const uint* packed [[buffer(1)]],device const ushort* scales [[buffer(2)]],
 device const ushort* biases [[buffer(3)]],device float* out [[buffer(4)]],
 constant Q4Params& runtime [[buffer(5)]],uint group [[threadgroup_position_in_grid]],
 uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    GARNET_AFFINE_PARAMETERS(Q4Params);
    const AffineIndex n=AffineIndex(group)*4+simd;
    if(n>=p.columns)return;
    float sum0=0,sum1=0,sum2=0,sum3=0;
    for(AffineIndex word=lane;word<p.inner/8;word+=32) {
        const uint values=packed[n*(p.inner/8)+word];
        const AffineIndex coefficient=n*(p.inner/64)+word/8;
        const float scale=q4_bf16(scales[coefficient]),bias=q4_bf16(biases[coefficient]);
        for(uint j=0;j<8;++j) {
            const float weight=fma(float((values>>(4*j))&15u),scale,bias);
            const AffineIndex k=word*8+j;
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
