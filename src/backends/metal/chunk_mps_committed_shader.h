// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace Garnet {
inline constexpr const char* kChunkMPSCommittedShader=R"METAL(
#if defined(GARNET_CHUNK_ATTENTION_INDEX32) && defined(GARNET_CHUNK_ATTENTION_BUCKETS)
kernel void chunk_mps_committed_pack(device const float* q [[buffer(0)]],device const float* active [[buffer(1)]],
 device float* qp [[buffer(2)]],constant ChunkAttentionParams& runtime [[buffer(3)]],constant ChunkMPSPitch& pitch [[buffer(4)]],uint id [[thread_position_in_grid]]) {
 const auto p=chunk_mps_params(runtime);uint d=id%p.dim,r=id/p.dim,t=r%p.chunk,h=(r/p.chunk)%p.heads,b=r/(p.chunk*p.heads);
 if(b<p.batch)qp[r*pitch.packed+d]=active[b*p.aStride+t]!=0?q[(b*p.chunk+t)*p.qStride+h*p.dim+d]:0;
}
kernel void chunk_mps_committed_softmax(device float* scores [[buffer(0)]],device const float* slots [[buffer(1)]],
 device const float* active [[buffer(2)]],constant ChunkAttentionParams& runtime [[buffer(3)]],constant ChunkMPSPitch& pitch [[buffer(4)]],
 threadgroup float* scratch [[threadgroup(0)]],uint query [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
 uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
 const auto p=chunk_mps_params(runtime);const uint extent=chunk_mps_extent;uint t=query%p.chunk,b=query/(p.chunk*p.heads);
 if(active[b*p.aStride+t]==0){for(uint s=tid;s<extent;s+=128)scores[query*pitch.cacheScore+s]=0;return;}
 const uint visible=uint(min(ulong(extent),ulong(slots[b*p.sStride+t])+1));threadgroup float partial[4];
 for(uint s=tid;s<visible;s+=128)scratch[s]=scores[query*pitch.cacheScore+s]*rsqrt(float(p.dim));
 threadgroup_barrier(mem_flags::mem_threadgroup);
 float maximum=-INFINITY;for(uint s=tid;s<visible;s+=128)maximum=max(maximum,scratch[s]);
 maximum=simd_max(maximum);if(lane==0)partial[simd]=maximum;threadgroup_barrier(mem_flags::mem_threadgroup);
 maximum=simd_max(lane<4?partial[lane]:-INFINITY);threadgroup_barrier(mem_flags::mem_threadgroup);
 float denominator=0;for(uint s=tid;s<visible;s+=128){float probability=exp(scratch[s]-maximum);scratch[s]=probability;denominator+=probability;}
 denominator=simd_sum(denominator);if(lane==0)partial[simd]=denominator;threadgroup_barrier(mem_flags::mem_threadgroup);
 denominator=simd_sum(lane<4?partial[lane]:0.0f);
 for(uint s=tid;s<extent;s+=128)scores[query*pitch.cacheScore+s]=s<visible?scratch[s]/denominator:0;
}
kernel void chunk_mps_committed_unpack(device const float* part [[buffer(0)]],device const float* gate [[buffer(1)]],
 device const float* active [[buffer(2)]],device float* output [[buffer(3)]],constant ChunkAttentionParams& runtime [[buffer(4)]],constant ChunkMPSPitch& pitch [[buffer(5)]],uint id [[thread_position_in_grid]]) {
 const auto p=chunk_mps_params(runtime);uint d=id%p.dim,r=id/p.dim,t=r%p.chunk,h=(r/p.chunk)%p.heads,b=r/(p.chunk*p.heads);
 if(b>=p.batch)return;uint flat=b*p.chunk+t,index=flat*p.oStride+h*p.dim+d;
 if(active[b*p.aStride+t]==0){output[index]=0;return;}
 float sum=part[r*pitch.packed+d];output[index]=p.gated?sum/(1+exp(-gate[flat*p.gStride+h*p.dim+d])):sum;
}

#endif
)METAL";
}
