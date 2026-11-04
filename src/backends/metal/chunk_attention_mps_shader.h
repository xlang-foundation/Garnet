// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace Garnet {
inline constexpr const char* kChunkAttentionMPSShader=R"METAL(
#ifdef GARNET_CHUNK_ATTENTION_INDEX32
// FP32 matrix products consume separate old-cache and live-token operands.
// Causal selection sends each probability to exactly one operand set, so a
// later overwrite never changes the cache value seen by an earlier query.
struct ChunkMPSPitch {uint packed,cacheScore,liveScore;};
inline BoundedChunkAttentionParams chunk_mps_params(constant ChunkAttentionParams& r) {
    return {uint(r.batch),uint(r.chunk),uint(r.heads),uint(r.kvHeads),uint(r.dim),uint(r.capacity),
        uint(r.qStride),uint(r.kStride),uint(r.vStride),uint(r.kcStride),uint(r.vcStride),uint(r.sStride),
        uint(r.aStride),uint(r.gStride),uint(r.oStride),uint(r.gated)};
}
kernel void chunk_mps_pack(device const float* q [[buffer(0)]],device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],device const float* active [[buffer(3)]],
    device float* qp [[buffer(4)]],device float* kp [[buffer(5)]],device float* vp [[buffer(6)]],
    constant ChunkAttentionParams& runtime [[buffer(7)]],constant ChunkMPSPitch& pitch [[buffer(8)]],
    uint id [[thread_position_in_grid]]) {
    const auto p=chunk_mps_params(runtime);
    uint d=id%p.dim,r=id/p.dim,t=r%p.chunk,h=(r/p.chunk)%p.heads,b=r/(p.chunk*p.heads);
    if(b<p.batch)qp[r*pitch.packed+d]=active[b*p.aStride+t]!=0?q[(b*p.chunk+t)*p.qStride+h*p.dim+d]:0;
    uint count=p.batch*p.kvHeads*p.chunk*p.dim;
    if(id<count) {
        h=(r/p.chunk)%p.kvHeads;b=r/(p.chunk*p.kvHeads);
        bool valid=active[b*p.aStride+t]!=0;
        kp[r*pitch.packed+d]=valid?k[(b*p.chunk+t)*p.kStride+h*p.dim+d]:0;
        vp[r*pitch.packed+d]=valid?v[(b*p.chunk+t)*p.vStride+h*p.dim+d]:0;
    }
}
kernel void chunk_mps_softmax(device float* cacheScores [[buffer(0)]],device const float* liveScores [[buffer(1)]],
    device float* liveProb [[buffer(2)]],device const float* slots [[buffer(3)]],device const float* active [[buffer(4)]],
    constant ChunkAttentionParams& runtime [[buffer(5)]],constant ChunkMPSPitch& pitch [[buffer(6)]],
    threadgroup float* scratch [[threadgroup(0)]],threadgroup short* selected [[threadgroup(1)]],
    uint query [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    const auto p=chunk_mps_params(runtime);
    uint t=query%p.chunk,b=query/(p.chunk*p.heads);
    if(active[b*p.aStride+t]==0) {
        for(uint s=tid;s<p.capacity;s+=128)cacheScores[query*pitch.cacheScore+s]=0;
        for(uint j=tid;j<p.chunk;j+=128)liveProb[query*pitch.liveScore+j]=0;
        return;
    }
    uint visible=uint(min(ulong(p.capacity),ulong(slots[b*p.sStride+t])+1));
    threadgroup int contiguous;threadgroup float partial[4];
    if(tid==0) {
        float first=slots[b*p.sStride];int valid=isfinite(first)&&first>=0&&first==floor(first);
        for(uint j=0;j<=t;++j)if(active[b*p.aStride+j]==0||slots[b*p.sStride+j]!=first+float(j))valid=0;
        contiguous=valid;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint s=tid;s<visible;s+=128) {
        int latest=-1;
        if(contiguous) {int candidate=int(s)-int(slots[b*p.sStride]);if(candidate>=0&&candidate<=int(t))latest=candidate;}
        else for(uint j=0;j<=t;++j)if(active[b*p.aStride+j]!=0&&slots[b*p.sStride+j]==float(s))latest=int(j);
        selected[s]=short(latest);
        scratch[s]=(latest<0?cacheScores[query*pitch.cacheScore+s]:liveScores[query*pitch.liveScore+uint(latest)])*rsqrt(float(p.dim));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maximum=-INFINITY;for(uint s=tid;s<visible;s+=128)maximum=max(maximum,scratch[s]);
    maximum=simd_max(maximum);if(lane==0)partial[simd]=maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maximum=simd_max(lane<4?partial[lane]:-INFINITY);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float denominator=0;
    for(uint s=tid;s<visible;s+=128){float probability=exp(scratch[s]-maximum);scratch[s]=probability;denominator+=probability;}
    denominator=simd_sum(denominator);if(lane==0)partial[simd]=denominator;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    denominator=simd_sum(lane<4?partial[lane]:0.0f);
    // All scratch reads below follow the last writes before the reduction's
    // threadgroup barrier. Each output probability has one owning thread.
    for(uint s=tid;s<p.capacity;s+=128)
        cacheScores[query*pitch.cacheScore+s]=s<visible&&selected[s]<0?scratch[s]/denominator:0;
    for(uint j=tid;j<p.chunk;j+=128) {
        float position=slots[b*p.sStride+j];
        bool valid=j<=t&&active[b*p.aStride+j]!=0&&isfinite(position)&&position>=0&&position<float(visible)&&position==floor(position);
        liveProb[query*pitch.liveScore+j]=valid&&selected[uint(position)]==short(j)?scratch[uint(position)]/denominator:0;
    }
}
kernel void chunk_mps_unpack(device const float* oldPart [[buffer(0)]],device const float* livePart [[buffer(1)]],
    device const float* gate [[buffer(2)]],device const float* active [[buffer(3)]],device float* output [[buffer(4)]],
    constant ChunkAttentionParams& runtime [[buffer(5)]],constant ChunkMPSPitch& pitch [[buffer(6)]],uint id [[thread_position_in_grid]]) {
    const auto p=chunk_mps_params(runtime);
    uint d=id%p.dim,r=id/p.dim,t=r%p.chunk,h=(r/p.chunk)%p.heads,b=r/(p.chunk*p.heads);
    if(b>=p.batch)return;
    uint flat=b*p.chunk+t,index=flat*p.oStride+h*p.dim+d;
    if(active[b*p.aStride+t]==0){output[index]=0;return;}
    float sum=oldPart[r*pitch.packed+d]+livePart[r*pitch.packed+d];
    output[index]=p.gated?sum/(1+exp(-gate[flat*p.gStride+h*p.dim+d])):sum;
}
#endif
)METAL";
}
