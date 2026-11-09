
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
// Fixed Mistral B1/B4,T4,H32,KV8,D128,C4096. Shared K tile,16 Q rows.
// Raw dot products retain lane order; the original full-row softmax loads scale.
kernel void verification_gqa16_dots(device const float* query [[buffer(0)]],
 device const float* key [[buffer(1)]],device const float* keyCache [[buffer(2)]],
 device const float* slots [[buffer(3)]],device const float* active [[buffer(4)]],
 device float* output [[buffer(5)]],constant VerifyAttentionParams& runtime [[buffer(6)]],
 uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
 uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
 GARNET_VERIFY_PARAMS;
 const uint row=group.x/8,kv=group.x%8,tile=group.y*32;
 if(row>=p.batch)return;
 const uint first=uint(slots[row*uint(p.sStride)]);
 uint visible[4];bool enabled[4];uint maximum=0;
 #pragma clang loop unroll(full)
 for(uint t=0;t<4;++t){enabled[t]=active[row*uint(p.aStride)+t]!=0.0f;visible[t]=enabled[t]?min(4096u,uint(slots[row*uint(p.sStride)+t])+1):0;maximum=max(maximum,visible[t]);}
 threadgroup float keys[32*128];
 for(uint index=tid;index<32*128;index+=128){
  const uint s=tile+index/128,j=index%128;
  keys[index]=s<maximum?(s<first?keyCache[row*uint(p.kcStride)+(kv*4096+s)*128+j]:key[(row*4+s-first)*uint(p.kStride)+kv*128+j]):0.0f;
 }
 threadgroup_barrier(mem_flags::mem_threadgroup);
 float4 queries[16];
 #pragma clang loop unroll(full)
 for(uint q=0;q<16;++q){const uint t=q/4,head=kv*4+q%4,base=(row*4+t)*uint(p.qStride)+head*128+lane;queries[q]=enabled[t]?float4(query[base],query[base+32],query[base+64],query[base+96]):float4(0.0f);}
 for(uint local=simd;local<32;local+=4){
  const uint s=tile+local;
  const float4 k=float4(keys[local*128+lane],keys[local*128+lane+32],keys[local*128+lane+64],keys[local*128+lane+96]);
  #pragma clang loop unroll(full)
  for(uint q=0;q<16;++q){
   const uint t=q/4,head=kv*4+q%4;
   if(s<visible[t]){
    float score=0.0f;
    score+=queries[q].x*k.x;score+=queries[q].y*k.y;
    score+=queries[q].z*k.z;score+=queries[q].w*k.w;
    score=simd_sum(score);
    if(lane==0)output[(row*4+t)*uint(p.pStride)+head*4096+s]=score;
   }
  }
 }
}
kernel void verification_gqa16_softmax(device const float* query [[buffer(0)]],
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
    for(ulong s=tid;s<visible;s+=128)scratch[s]=output[base+s]*rsqrt(float(p.dim));
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
