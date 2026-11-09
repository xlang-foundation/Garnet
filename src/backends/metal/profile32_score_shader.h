// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace Garnet {
// Compile the bounded Mistral layout; retain runtime rsqrt and dot order.
inline constexpr const char* kProfile32ScoreShader=R"METAL(
#include <metal_stdlib>
using namespace metal;
constant uint score_chunk [[function_constant(1)]];
struct ScoresParams {ulong batch,heads,kvHeads,dim,capacity,qStride,kStride,sStride,aStride,oStride;};
struct Score32Params {uint batch,heads,kvHeads,dim,capacity,qStride,kStride,sStride,aStride,oStride;};
kernel void score_dots_profile32(device const float* query [[buffer(0)]],device const float* key [[buffer(1)]],
 device const float* slots [[buffer(2)]],device const float* active [[buffer(3)]],device float* output [[buffer(4)]],
 constant ScoresParams& runtime [[buffer(5)]],
 uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
 const Score32Params p{uint(runtime.batch),32u,8u,128u,4096u,4096u,4194304u,uint(runtime.sStride),uint(runtime.aStride),131072u};
 const uint id=group.x;if(id>=p.batch*p.heads)return;
 const uint row=id/p.heads,head=id%p.heads,kv=head/(p.heads/p.kvHeads),base=row*p.oStride+head*p.capacity;
 if(active[row*p.aStride]==0.0f)return;
 const uint visible=min(p.capacity,uint(slots[row*p.sStride])+1);
 const uint begin=uint(group.y)*score_chunk,end=min(visible,begin+score_chunk);
 for(uint s=begin+simd;s<end;s+=4){
  float score=-1e30f;
  if(float(s)<=slots[row*p.sStride]){
   score=0.0f;
   #pragma clang loop unroll(disable)
   for(uint j=lane;j<p.dim;j+=32)score+=query[row*p.qStride+head*p.dim+j]*key[row*p.kStride+(kv*p.capacity+s)*p.dim+j];
   score=simd_sum(score)*rsqrt(float(runtime.dim));
  }
  if(lane==0){output[base+s]=score;}
 }
}
kernel void score_softmax_profile32(device const float* slots [[buffer(2)]],device const float* active [[buffer(3)]],device float* output [[buffer(4)]],
 constant ScoresParams& runtime [[buffer(5)]],threadgroup float* scratch [[threadgroup(0)]],
 uint id [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
 const Score32Params p{uint(runtime.batch),32u,8u,128u,4096u,4096u,4194304u,uint(runtime.sStride),uint(runtime.aStride),131072u};
 if(id>=p.batch*p.heads)return;
 uint row=id/p.heads,head=id%p.heads,base=row*p.oStride+head*p.capacity;
 if(active[row*p.aStride]==0.0f){for(uint s=tid;s<p.capacity;s+=128){output[base+s]=0.0f;}return;}
 const uint visible=min(p.capacity,uint(slots[row*p.sStride])+1);
 threadgroup float partial[4];
 for(uint s=tid;s<visible;s+=128)scratch[s]=output[base+s];
 threadgroup_barrier(mem_flags::mem_threadgroup);
 float maximum=-INFINITY;for(uint s=tid;s<visible;s+=128)maximum=max(maximum,scratch[s]);
 maximum=simd_max(maximum);if(lane==0)partial[simd]=maximum;
 threadgroup_barrier(mem_flags::mem_threadgroup);
 maximum=simd_max(lane<4?partial[lane]:-INFINITY);
 threadgroup_barrier(mem_flags::mem_threadgroup);
 float denominator=0.0f;
 for(uint s=tid;s<visible;s+=128){float probability=exp(scratch[s]-maximum);scratch[s]=probability;denominator+=probability;}
 denominator=simd_sum(denominator);if(lane==0)partial[simd]=denominator;
 threadgroup_barrier(mem_flags::mem_threadgroup);
 denominator=simd_sum(lane<4?partial[lane]:0.0f);
 for(uint s=tid;s<p.capacity;s+=128){output[base+s]=s<visible?scratch[s]/denominator:0.0f;}
}
)METAL";
}
