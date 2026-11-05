// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace Garnet {
// LayoutParams is declared by the common state-kernel source. Host eligibility
// proves aligned rows, one mask token per vector and bounded uint addressing.
inline constexpr const char* kChunkLayoutVectorShader=R"METAL(
kernel void chunk_layout_vector4(device const float4* a [[buffer(0)]],
 device const float* flags [[buffer(1)]],device float4* output [[buffer(2)]],
 constant LayoutParams& p [[buffer(3)]],uint id [[thread_position_in_grid]]) {
    const uint sourceColumns=uint(p.aColumns/4),outputColumns=uint(p.oColumns/4);
    if(id>=uint(p.oRows)*outputColumns)return;
    const uint sourceRow=id/sourceColumns,sourceColumn=id%sourceColumns;
    float4 value=a[sourceRow*uint(p.aStride/4)+sourceColumn];
    if(p.tokenWidth&&flags[sourceRow*uint(p.bStride)+sourceColumn/uint(p.tokenWidth/4)]==0.0f)
        value=float4(0.0f);
    output[(id/outputColumns)*uint(p.oStride/4)+id%outputColumns]=value;
}
)METAL";
}
