// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace Garnet {
// The caller proves identical cache views and a contiguous active prefix per
// owner. Each token then has a unique destination; no cache read/copy is needed.
inline constexpr const char* kChunkKVScatterShader=R"METAL(
kernel void kv_chunk_scatter4(device const float* token [[buffer(0)]],
 device const float* slots [[buffer(2)]],device const float* active [[buffer(3)]],
 device float* output [[buffer(4)]],constant KVChunkParams& p [[buffer(5)]],
 uint id [[thread_position_in_grid]]) {
    const uint vectors=uint(p.heads)*uint(p.dim/4),chunk=uint(p.chunk);
    if(id>=uint(p.batch)*chunk*vectors)return;
    const uint flat=id/vectors,row=flat/chunk,t=flat%chunk,col=id%vectors;
    if(active[row*uint(p.aStride)+t]==0.0f)return;
    const float slot=slots[row*uint(p.sStride)+t];
    if(!isfinite(slot)||slot<0.0f||slot>=float(p.capacity)||slot!=floor(slot))return;
    const uint head=col/uint(p.dim/4),d=col%uint(p.dim/4);
    device const float4* source=(device const float4*)token;
    device float4* destination=(device float4*)output;
    destination[row*uint(p.oStride/4)+(head*uint(p.capacity)+uint(slot))*uint(p.dim/4)+d]=
        source[flat*uint(p.tStride/4)+col];
}
)METAL";
}
