// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace Garnet {
// Same FP32 expression and ordinary main-library math options, followed by the
// same explicit half conversion as the separate native bulk-input cast.
inline constexpr const char* kBulkSiluHalfShader=R"METAL(
struct BulkSiluHalfParams {uint rows,columns,aStride,bStride,hStride;};
kernel void bulk_silu_half(device const float* a [[buffer(0)]],device const float* b [[buffer(1)]],device half* h [[buffer(2)]],constant BulkSiluHalfParams& p [[buffer(3)]],uint id [[thread_position_in_grid]]){
 if(id>=p.rows*p.columns)return;
 uint row=id/p.columns,col=id%p.columns;float gate=a[row*p.aStride+col];
 float value=(gate/(1.0f+exp(-gate)))*b[row*p.bStride+col];
 h[row*p.hStride+col]=half(value);
}
)METAL";
}
