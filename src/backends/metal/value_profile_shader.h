// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace Garnet {
// Only bounded address generation and dispatch width change. Every column
// retains the original ascending-token FP32 accumulation sequence.
inline constexpr const char* kValueProfileShader=R"METAL(
#include <metal_stdlib>
using namespace metal;
struct AttentionParams {ulong batch,heads,kvHeads,dim,capacity,pStride,vStride,gStride,sStride,oStride,gated;};
struct Attention32Params {uint batch,heads,kvHeads,dim,capacity,pStride,vStride,gStride,sStride,oStride,gated;};
kernel void value_profile32(device const float* probabilities [[buffer(0)]],device const float* value [[buffer(1)]],
    device const float* gate [[buffer(2)]],device const float* slots [[buffer(3)]],device float* output [[buffer(4)]],constant AttentionParams& runtime [[buffer(5)]],
    uint id [[thread_position_in_grid]]) {
    const Attention32Params p{uint(runtime.batch),32u,8u,128u,4096u,131072u,4194304u,4096u,uint(runtime.sStride),4096u,0u};
    if(id>=p.batch*p.heads*p.dim)return;
    uint row=id/(p.heads*p.dim),head=(id/p.dim)%p.heads,col=id%p.dim,kv=head/(p.heads/p.kvHeads);
    const float slot=slots[row*p.sStride];
    const uint visible=isfinite(slot)&&slot>=0.0f&&slot<float(p.capacity)?uint(slot)+1:p.capacity;
    float result=0.0f;
    for(uint s=0;s<visible;++s)result+=probabilities[row*p.pStride+head*p.capacity+s]*value[row*p.vStride+(kv*p.capacity+s)*p.dim+col];
    output[row*p.oStride+head*p.dim+col]=p.gated?result/(1.0f+exp(-gate[row*p.gStride+head*p.dim+col])):result;
}
)METAL";
}
