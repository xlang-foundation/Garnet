#define GARNET_VERIFY_VALUE_VECTOR4 1
#define GARNET_VERIFY_MISTRAL_LAYOUT 1

#include <metal_stdlib>
using namespace metal;
struct VerifyAttentionParams {
    ulong batch,chunk,heads,kvHeads,dim,capacity,qStride,kStride,vStride,
          kcStride,vcStride,sStride,aStride,gStride,pStride,oStride,gated;
};
#ifdef GARNET_VERIFY_MISTRAL_LAYOUT
// One precompiled up-to4096 layout. Slots/flags and all storage strides remain
// runtime values. Reductions visit the visible prefix; the existing contract
// still clears the probability suffix through the prepared capacity.
#define GARNET_VERIFY_PARAMS const VerifyAttentionParams p{1,4,32,8,128,4096,runtime.qStride,runtime.kStride,runtime.vStride,runtime.kcStride,runtime.vcStride,runtime.sStride,runtime.aStride,runtime.gStride,runtime.pStride,runtime.oStride,0}
#else
#define GARNET_VERIFY_PARAMS constant VerifyAttentionParams& p=runtime
#endif
kernel void verification_attention_scores(device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],device const float* keyCache [[buffer(2)]],
    device const float* slots [[buffer(3)]],device const float* active [[buffer(4)]],
    device float* output [[buffer(5)]],constant VerifyAttentionParams& runtime [[buffer(6)]],
    threadgroup float* scratch [[threadgroup(0)]],uint id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    GARNET_VERIFY_PARAMS;
    if(id>=p.batch*p.chunk*p.heads)return;
    ulong flatRow=id/p.heads,head=id%p.heads,row=flatRow/p.chunk,token=flatRow%p.chunk;
    ulong kv=head/(p.heads/p.kvHeads),base=flatRow*p.pStride+head*p.capacity;
    if(active[row*p.aStride+token]==0.0f){for(ulong s=tid;s<p.capacity;s+=128)output[base+s]=0.0f;return;}
    const ulong first=ulong(slots[row*p.sStride]);
    const ulong visible=min(p.capacity,ulong(slots[row*p.sStride+token])+1);
    threadgroup float partial[4];
    for(ulong s=simd;s<visible;s+=4) {
        float score=-1e30f;
        if(float(s)<=slots[row*p.sStride+token]) {
            score=0.0f;
            for(ulong j=lane;j<p.dim;j+=32) {
                float k=s<first?keyCache[row*p.kcStride+(kv*p.capacity+s)*p.dim+j]:key[(row*p.chunk+s-first)*p.kStride+kv*p.dim+j];
                score+=query[flatRow*p.qStride+head*p.dim+j]*k;
            }
            score=simd_sum(score)*rsqrt(float(p.dim));
        }
        if(lane==0)scratch[s]=score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maximum=-INFINITY;for(ulong s=tid;s<visible;s+=128)maximum=max(maximum,scratch[s]);
    maximum=simd_max(maximum);if(lane==0)partial[simd]=maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maximum=simd_max(lane<4?partial[lane]:-INFINITY);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float denominator=0.0f;
    for(ulong s=tid;s<visible;s+=128){float probability=exp(scratch[s]-maximum);scratch[s]=probability;denominator+=probability;}
    denominator=simd_sum(denominator);if(lane==0)partial[simd]=denominator;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    denominator=simd_sum(lane<4?partial[lane]:0.0f);
    for(ulong s=tid;s<p.capacity;s+=128)output[base+s]=s<visible?scratch[s]/denominator:0.0f;
}
kernel void verification_attention_context(device const float* probabilities [[buffer(0)]],
    device const float* value [[buffer(1)]],device const float* valueCache [[buffer(2)]],
    device const float* gate [[buffer(3)]],device const float* slots [[buffer(4)]],
    device const float* active [[buffer(5)]],device float* output [[buffer(6)]],
    constant VerifyAttentionParams& runtime [[buffer(7)]],uint id [[thread_position_in_grid]]) {
    GARNET_VERIFY_PARAMS;
    if(id>=p.batch*p.chunk*p.heads*p.dim)return;
    ulong flatRow=id/(p.heads*p.dim),head=(id/p.dim)%p.heads,col=id%p.dim;
    ulong row=flatRow/p.chunk,token=flatRow%p.chunk,kv=head/(p.heads/p.kvHeads);
    if(active[row*p.aStride+token]==0.0f){output[flatRow*p.oStride+head*p.dim+col]=0.0f;return;}
    const ulong first=ulong(slots[row*p.sStride]);
    const ulong visible=ulong(slots[row*p.sStride+token])+1;
    float result=0.0f;
    for(ulong s=0;s<visible;++s) {
        float v=s<first?valueCache[row*p.vcStride+(kv*p.capacity+s)*p.dim+col]:value[(row*p.chunk+s-first)*p.vStride+kv*p.dim+col];
        result+=probabilities[flatRow*p.pStride+head*p.capacity+s]*v;
    }
    output[flatRow*p.oStride+head*p.dim+col]=p.gated?result/(1.0f+exp(-gate[flatRow*p.gStride+head*p.dim+col])):result;
}

#ifdef GARNET_VERIFY_VALUE_VECTOR4
kernel void verification_attention_context_vector4(device const float* probabilities [[buffer(0)]],
    device const float* value [[buffer(1)]],device const float* valueCache [[buffer(2)]],
    device const float* gate [[buffer(3)]],device const float* slots [[buffer(4)]],
    device const float* active [[buffer(5)]],device float* output [[buffer(6)]],
    constant VerifyAttentionParams& runtime [[buffer(7)]],uint id [[thread_position_in_grid]]) {
    GARNET_VERIFY_PARAMS;
    const ulong scalar=ulong(id)*4;
    if(scalar>=p.batch*p.chunk*p.heads*p.dim)return;
    const ulong flatRow=scalar/(p.heads*p.dim),head=(scalar/p.dim)%p.heads,col=scalar%p.dim;
    const ulong row=flatRow/p.chunk,token=flatRow%p.chunk,kv=head/(p.heads/p.kvHeads);
    auto out=reinterpret_cast<device packed_float4*>(output+flatRow*p.oStride+head*p.dim+col);
    if(active[row*p.aStride+token]==0.0f){*out=packed_float4(0.0f);return;}
    const ulong first=ulong(slots[row*p.sStride]),visible=ulong(slots[row*p.sStride+token])+1;
    float4 result=float4(0.0f);
    for(ulong i=0;i<visible;++i) {
        float4 v;
        if(i<first)v=float4(*reinterpret_cast<device const packed_float4*>(valueCache+row*p.vcStride+(kv*p.capacity+i)*p.dim+col));
        else v=float4(*reinterpret_cast<device const packed_float4*>(value+(row*p.chunk+i-first)*p.vStride+kv*p.dim+col));
        result+=probabilities[flatRow*p.pStride+head*p.capacity+i]*v;
    }
    if(p.gated)result/=1.0f+exp(-float4(*reinterpret_cast<device const packed_float4*>(gate+flatRow*p.gStride+head*p.dim+col)));
    *out=packed_float4(result);
}
#endif

constant uint verify_score_chunk [[function_constant(1)]];
kernel void verification_split_score_dots(device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],device const float* keyCache [[buffer(2)]],
    device const float* slots [[buffer(3)]],device const float* active [[buffer(4)]],
    device float* output [[buffer(5)]],constant VerifyAttentionParams& runtime [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    GARNET_VERIFY_PARAMS;
    const ulong id=group.x;if(id>=p.batch*p.chunk*p.heads)return;
    ulong flatRow=id/p.heads,head=id%p.heads,row=flatRow/p.chunk,token=flatRow%p.chunk;
    ulong kv=head/(p.heads/p.kvHeads),base=flatRow*p.pStride+head*p.capacity;
    if(active[row*p.aStride+token]==0.0f)return;
    const ulong first=ulong(slots[row*p.sStride]);
    const ulong visible=min(p.capacity,ulong(slots[row*p.sStride+token])+1);
    const ulong begin=ulong(group.y)*verify_score_chunk,end=min(visible,begin+verify_score_chunk);
    for(ulong s=begin+simd;s<end;s+=4) {
        float score=-1e30f;
        if(float(s)<=slots[row*p.sStride+token]) {
            score=0.0f;
            for(ulong j=lane;j<p.dim;j+=32) {
                float k=s<first?keyCache[row*p.kcStride+(kv*p.capacity+s)*p.dim+j]:key[(row*p.chunk+s-first)*p.kStride+kv*p.dim+j];
                score+=query[flatRow*p.qStride+head*p.dim+j]*k;
            }
            score=simd_sum(score)*rsqrt(float(p.dim));
        }
        if(lane==0)output[base+s]=score;
    }
}
kernel void verification_split_score_softmax(device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],device const float* keyCache [[buffer(2)]],
    device const float* slots [[buffer(3)]],device const float* active [[buffer(4)]],
    device float* output [[buffer(5)]],constant VerifyAttentionParams& runtime [[buffer(6)]],
    threadgroup float* scratch [[threadgroup(0)]],uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    GARNET_VERIFY_PARAMS;
    const ulong id=group.x;if(id>=p.batch*p.chunk*p.heads)return;
    ulong flatRow=id/p.heads,head=id%p.heads,row=flatRow/p.chunk,token=flatRow%p.chunk;
    ulong kv=head/(p.heads/p.kvHeads),base=flatRow*p.pStride+head*p.capacity;
    if(active[row*p.aStride+token]==0.0f){for(ulong s=tid;s<p.capacity;s+=128)output[base+s]=0.0f;return;}
    const ulong visible=min(p.capacity,ulong(slots[row*p.sStride+token])+1);
    threadgroup float partial[4];
    for(ulong s=tid;s<visible;s+=128)scratch[s]=output[base+s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maximum=-INFINITY;for(ulong s=tid;s<visible;s+=128)maximum=max(maximum,scratch[s]);
    maximum=simd_max(maximum);if(lane==0)partial[simd]=maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maximum=simd_max(lane<4?partial[lane]:-INFINITY);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float denominator=0.0f;
    for(ulong s=tid;s<visible;s+=128){float probability=exp(scratch[s]-maximum);scratch[s]=probability;denominator+=probability;}
    denominator=simd_sum(denominator);if(lane==0)partial[simd]=denominator;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    denominator=simd_sum(lane<4?partial[lane]:0.0f);
    for(ulong s=tid;s<p.capacity;s+=128)output[base+s]=s<visible?scratch[s]/denominator:0.0f;
}
