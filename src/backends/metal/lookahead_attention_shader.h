// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace Garnet {
inline constexpr const char* kLookaheadAttentionShader=R"METAL(
#include <metal_stdlib>
using namespace metal;
constant uint la_batch [[function_constant(0)]];
constant uint la_capacity [[function_constant(1)]];
constant uint la_window [[function_constant(2)]];
constant uint la_levels [[function_constant(3)]];
constant uint la_guesses [[function_constant(4)]];
struct LookaheadAttentionParams {
 ulong qStride,kStride,vStride,kcStride,vcStride,pStride,oStride,dim;
};
// Identical physical geometry to the prepared native LookaheadLayout. The
// active mask clips work at request/capacity tails; it never joins branches.
inline uint lookahead_local_mask(uint token) {
 const uint windowRows=(la_levels-1)*la_window;
 const uint used=windowRows+(la_levels-1)*la_guesses;
 if(token>=used)return 0;
 uint mask=0;
 if(token<windowRows) {
  const uint level=token/la_window,column=token%la_window;
  for(uint i=0;i<=column;++i)mask|=1u<<i;
  for(uint l=1;l<=level;++l)mask|=1u<<(l*la_window+column);
 }else {
  const uint relative=token-windowRows,branch=relative/(la_levels-1),depth=relative%(la_levels-1);
  mask=1;
  for(uint i=0;i<=depth;++i)mask|=1u<<(windowRows+branch*(la_levels-1)+i);
 }
 return mask;
}
inline uint lookahead_local_key(uint mask,uint ordinal) {
 for(uint key=0;key<16;++key)if(mask&(1u<<key)){if(ordinal==0)return key;--ordinal;}
 return 16; // Host validates full causal ancestor closure before encoding.
}
kernel void lookahead_attention_scores(device const float* query [[buffer(0)]],
 device const float* key [[buffer(1)]],device const float* keyCache [[buffer(2)]],
 device float* probabilities [[buffer(3)]],constant LookaheadAttentionParams& p [[buffer(4)]],
 constant uint* prefix [[buffer(5)]],constant ushort* active [[buffer(6)]],
 threadgroup float* scratch [[threadgroup(0)]],uint id [[threadgroup_position_in_grid]],
 uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],
 uint simd [[simdgroup_index_in_threadgroup]]) {
 if(id>=la_batch*16*32)return;
 const uint flat=id/32,head=id%32,row=flat/16,token=flat%16,kv=head/4;
 const ulong base=ulong(flat)*p.pStride+head*la_capacity;
 if((uint(active[row])&(1u<<token))==0){
  for(uint s=tid;s<la_capacity;s+=128)probabilities[base+s]=0.0f;
  return;
 }
 const uint mask=lookahead_local_mask(token)&uint(active[row]);
 const uint first=prefix[row],visible=first+popcount(mask);
 threadgroup float partial[4];
 // Compact each branch's actual causal keys into ordinary logical key order.
 // The FP32 score/softmax boundary and reduction lane order are unchanged.
 for(uint s=simd;s<visible;s+=4) {
  const uint local=s<first?0:lookahead_local_key(mask,s-first);
  float score=0.0f;
  for(uint j=lane;j<128;j+=32) {
   const float k=s<first?keyCache[ulong(row)*p.kcStride+(kv*la_capacity+s)*128+j]
       :key[(row*16+local)*p.kStride+kv*128+j];
   score+=query[ulong(flat)*p.qStride+head*128+j]*k;
  }
  // Keep the reference's runtime scale evaluation boundary even though the
  // fixed layout separately validates D128 before dispatch.
  score=simd_sum(score)*rsqrt(float(p.dim));
  if(lane==0)scratch[s]=score;
 }
 threadgroup_barrier(mem_flags::mem_threadgroup);
 float maximum=-INFINITY;for(uint s=tid;s<visible;s+=128)maximum=max(maximum,scratch[s]);
 maximum=simd_max(maximum);if(lane==0)partial[simd]=maximum;
 threadgroup_barrier(mem_flags::mem_threadgroup);
 maximum=simd_max(lane<4?partial[lane]:-INFINITY);
 threadgroup_barrier(mem_flags::mem_threadgroup);
 float denominator=0.0f;
 for(uint s=tid;s<visible;s+=128){const float probability=exp(scratch[s]-maximum);scratch[s]=probability;denominator+=probability;}
 denominator=simd_sum(denominator);if(lane==0)partial[simd]=denominator;
 threadgroup_barrier(mem_flags::mem_threadgroup);
 denominator=simd_sum(lane<4?partial[lane]:0.0f);
 for(uint s=tid;s<la_capacity;s+=128)probabilities[base+s]=s<visible?scratch[s]/denominator:0.0f;
}
kernel void lookahead_attention_context(device const float* probabilities [[buffer(0)]],
 device const float* value [[buffer(1)]],device const float* valueCache [[buffer(2)]],
 device float* output [[buffer(3)]],constant LookaheadAttentionParams& p [[buffer(4)]],
 constant uint* prefix [[buffer(5)]],constant ushort* active [[buffer(6)]],
 uint id [[thread_position_in_grid]]) {
 const ulong scalar=ulong(id)*4;
 if(scalar>=ulong(la_batch)*16*32*128)return;
 const uint flat=uint(scalar/(32*128)),head=uint((scalar/128)%32),column=uint(scalar%128);
 const uint row=flat/16,token=flat%16,kv=head/4;
 auto out=reinterpret_cast<device packed_float4*>(output+ulong(flat)*p.oStride+head*128+column);
 if((uint(active[row])&(1u<<token))==0){*out=packed_float4(0.0f);return;}
 const uint mask=lookahead_local_mask(token)&uint(active[row]);
 const uint first=prefix[row],visible=first+popcount(mask);
 float4 result=float4(0.0f);
 for(uint s=0;s<visible;++s) {
  float4 v;
  if(s<first)v=float4(*reinterpret_cast<device const packed_float4*>(valueCache+ulong(row)*p.vcStride+(kv*la_capacity+s)*128+column));
  else {
   const uint local=lookahead_local_key(mask,s-first);
   v=float4(*reinterpret_cast<device const packed_float4*>(value+(row*16+local)*p.vStride+kv*128+column));
  }
  result+=probabilities[ulong(flat)*p.pStride+head*la_capacity+s]*v;
 }
 *out=packed_float4(result);
}
)METAL";
}
