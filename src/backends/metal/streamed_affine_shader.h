// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace Garnet {
// Bounded per-context workspace. No model weight cache and no API operators.
inline constexpr const char* kStreamedAffineShader=R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void streamed_affine_unpack(device const uint* w [[buffer(0)]],device const half* s [[buffer(1)]],device const half* b [[buffer(2)]],device half* h [[buffer(3)]],constant uint2& shape [[buffer(4)]],uint i [[thread_position_in_grid]]) {
 if(i>=shape.x*shape.y)return;
 uint n=i/shape.y,k=i%shape.y,c=n*(shape.y/64)+k/64;
 h[i]=half(fma(float((w[n*(shape.y/8)+k/8]>>(4*(k%8)))&15u),float(s[c]),float(b[c])));
}
kernel void streamed_affine_gather(device const float* a [[buffer(0)]],device half* h [[buffer(1)]],constant uint* ids [[buffer(2)]],constant uint2& shape [[buffer(3)]],uint i [[thread_position_in_grid]]) {
 if(i>=512u*shape.x)return;
 h[i]=half(a[ids[i/shape.x]*shape.y+i%shape.x]);
}
kernel void streamed_affine_scatter(device const float* h [[buffer(0)]],device float* o [[buffer(1)]],constant uint* ids [[buffer(2)]],constant uint2& shape [[buffer(3)]],uint i [[thread_position_in_grid]]) {
 if(i>=512u*shape.x)return;
 o[ids[i/shape.x]*shape.y+i%shape.x]=h[i];
}
)MSL";
}
