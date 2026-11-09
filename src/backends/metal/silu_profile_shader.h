// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace Garnet {
// Same FP32 expression and native compilation options. Host proof bounds every
// complete index extent before using this uint32 address/dispatch profile.
inline constexpr const char* kSiluProfile32Shader=R"METAL(
kernel void silu_profile32(device const float* a [[buffer(0)]],device const float* b [[buffer(1)]],
 device float* c [[buffer(2)]],constant Params& p [[buffer(3)]],uint id [[thread_position_in_grid]]) {
 if(id>=uint(p.rows)*uint(p.columns))return;
 uint row=id/uint(p.columns),col=id%uint(p.columns);
 float gate=a[row*uint(p.aStride)+col];
 c[row*uint(p.cStride)+col]=(gate/(1.0f+exp(-gate)))*b[row*uint(p.bStride)+col];
}
)METAL";
}
