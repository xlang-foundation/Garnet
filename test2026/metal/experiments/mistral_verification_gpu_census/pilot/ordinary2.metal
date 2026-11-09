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

#define GARNET_BLOCKED_COLUMNS 4
#define GARNET_BLOCKED_ROWS affine_decode_rows

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


struct Geometry2Params {ulong rows,inner,columns,aStride,oStride;};
kernel void affine_q4_groups2(device const float* a [[buffer(0)]],
 device const uint* packed [[buffer(1)]],device const ushort* scales [[buffer(2)]],
 device const ushort* biases [[buffer(3)]],device float* out [[buffer(4)]],
 constant Geometry2Params& runtime [[buffer(5)]],
#ifdef GARNET_BLOCKED_INDEXED
 constant uint* indices [[buffer(6)]],
#endif
 uint group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],
 uint simd [[simdgroup_index_in_threadgroup]]) {
    GARNET_AFFINE_PARAMETERS(Geometry2Params);
    const AffineIndex first=(AffineIndex(group)*2+simd)*GARNET_BLOCKED_COLUMNS;
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


struct Geometry8Params {ulong rows,inner,columns,aStride,oStride;};
kernel void affine_q4_groups8(device const float* a [[buffer(0)]],
 device const uint* packed [[buffer(1)]],device const ushort* scales [[buffer(2)]],
 device const ushort* biases [[buffer(3)]],device float* out [[buffer(4)]],
 constant Geometry8Params& runtime [[buffer(5)]],
#ifdef GARNET_BLOCKED_INDEXED
 constant uint* indices [[buffer(6)]],
#endif
 uint group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],
 uint simd [[simdgroup_index_in_threadgroup]]) {
    GARNET_AFFINE_PARAMETERS(Geometry8Params);
    const AffineIndex first=(AffineIndex(group)*8+simd)*GARNET_BLOCKED_COLUMNS;
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


kernel void geometry_pressure(device uint* payload [[buffer(0)]],uint tid [[thread_position_in_grid]]) {
 payload[tid]=payload[tid]*1664525u+1013904223u;
}

#define GROUP64_INDEXED 0
#define GROUP64_ROWS affine_decode_rows
// Experimental FP32 centered affine-group dot. Separate arithmetic policy.
// Including library supplies the original affine prelude and function constants.
constant uint group64_columns [[function_constant(21)]];
struct Group64Params {ulong rows,inner,columns,aStride,oStride;};
kernel void centered_input_sums(device const float* a [[buffer(0)]],
 constant Group64Params& runtime [[buffer(5)]],
#if GROUP64_INDEXED
 constant uint* indices [[buffer(6)]],
#endif
 device float* sums [[buffer(7)]],uint id [[thread_position_in_grid]]) {
    GARNET_AFFINE_PARAMETERS(Group64Params);
    const uint groups=p.inner/64,logical=id/groups,g=id%groups;
    if(logical>=GROUP64_ROWS)return;
#if GROUP64_INDEXED
    const uint row=indices[logical];
#else
    const uint row=logical;
#endif
    float sum=0;
#pragma clang loop unroll(disable)
    for(uint j=0;j<64;++j)sum=sum+a[row*p.aStride+g*64+j];
    sums[row*groups+g]=sum;
}
kernel void centered_q4_group64(device const float* a [[buffer(0)]],
 device const uint* packed [[buffer(1)]],device const ushort* scales [[buffer(2)]],
 device const ushort* biases [[buffer(3)]],device float* out [[buffer(4)]],
 constant Group64Params& runtime [[buffer(5)]],
#if GROUP64_INDEXED
 constant uint* indices [[buffer(6)]],
#endif
 device const float* input_sums [[buffer(7)]],
 uint group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],
 uint simd [[simdgroup_index_in_threadgroup]]) {
    GARNET_AFFINE_PARAMETERS(Group64Params);
    const uint first=(group*4+simd)*group64_columns,groups=p.inner/64;
    if(first>=p.columns)return;
    uint physical[4];float total[4][4];
#pragma clang loop unroll(full)
    for(uint r=0;r<4;++r){
#if GROUP64_INDEXED
        physical[r]=r<GROUP64_ROWS?indices[r]:0;
#else
        physical[r]=r;
#endif
#pragma clang loop unroll(full)
        for(uint c=0;c<4;++c)total[r][c]=0;
    }
    for(uint g=lane;g<groups;g+=32){
        float scale[4],residual[4],dot[4][4];int zero[4];uint4 wordsLow[4],wordsHigh[4];
#pragma clang loop unroll(full)
        for(uint c=0;c<4;++c)if(c<group64_columns&&first+c<p.columns){
            const uint coefficient=(first+c)*groups+g;
            scale[c]=garnet_affine_coefficient(scales[coefficient]);
            const float bias=garnet_affine_coefficient(biases[coefficient]);
            const float z=scale[c]!=0?-bias/scale[c]:-1;
            zero[c]=(z>=0&&z<=15)?int(rint(z)):0;
            residual[c]=fma(float(zero[c]),scale[c],bias);
            const device uint* wp=packed+(first+c)*(p.inner/8)+g*8;
            wordsLow[c]=uint4(*reinterpret_cast<device const packed_uint4*>(wp));
            wordsHigh[c]=uint4(*reinterpret_cast<device const packed_uint4*>(wp+4));
#pragma clang loop unroll(full)
            for(uint r=0;r<4;++r)dot[r][c]=0;
        }
#pragma clang loop unroll(full)
        for(uint word=0;word<8;++word){
            float4 low[4],high[4];
#pragma clang loop unroll(full)
            for(uint r=0;r<4;++r)if(r<GROUP64_ROWS){
                const uint base=physical[r]*p.aStride+g*64+word*8;
                low[r]=float4(*reinterpret_cast<device const packed_float4*>(a+base));
                high[r]=float4(*reinterpret_cast<device const packed_float4*>(a+base+4));
            }
#pragma clang loop unroll(full)
            for(uint c=0;c<4;++c)if(c<group64_columns&&first+c<p.columns){
                const uint values=word<4?wordsLow[c][word]:wordsHigh[c][word-4];
#pragma clang loop unroll(full)
                for(uint j=0;j<8;++j){
                    const float q=float(int((values>>(j*4))&15u)-zero[c]);
#pragma clang loop unroll(full)
                    for(uint r=0;r<4;++r)if(r<GROUP64_ROWS)
                        dot[r][c]=fma(j<4?low[r][j]:high[r][j-4],q,dot[r][c]);
                }
            }
        }
#pragma clang loop unroll(full)
        for(uint c=0;c<4;++c)if(c<group64_columns&&first+c<p.columns){
#pragma clang loop unroll(full)
            for(uint r=0;r<4;++r)if(r<GROUP64_ROWS)
                total[r][c]=fma(dot[r][c],scale[c],fma(input_sums[physical[r]*groups+g],residual[c],total[r][c]));
        }
    }
#pragma clang loop unroll(full)
    for(uint c=0;c<4;++c)if(c<group64_columns&&first+c<p.columns){
#pragma clang loop unroll(full)
        for(uint r=0;r<4;++r)if(r<GROUP64_ROWS){
            const float result=simd_sum(total[r][c]);
            if(lane==0)out[physical[r]*p.oStride+first+c]=result;
        }
    }
}
