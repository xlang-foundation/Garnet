// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace Garnet {
inline constexpr const char* kRowFinalHiddenShader=R"METAL(
struct FinalHiddenParams {ulong rows,width,chunk,iStride,aStride,oStride;};
kernel void gather_last_active_hidden(device const float* input [[buffer(0)]],
 device const float* active [[buffer(1)]],device float* output [[buffer(2)]],
 constant FinalHiddenParams& p [[buffer(3)]],uint id [[thread_position_in_grid]]) {
    if(ulong(id)>=p.rows*p.width)return;
    const ulong row=ulong(id)/p.width,column=ulong(id)%p.width;
    for(ulong token=p.chunk;token>0;--token)if(active[row*p.aStride+token-1]!=0.0f) {
        output[row*p.oStride+column]=input[row*p.iStride+(token-1)*p.width+column];return;
    }
    // Inactive rows keep their earlier completed token, not padded zeros.
}
)METAL";
}
