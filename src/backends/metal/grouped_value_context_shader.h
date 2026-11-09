// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace Garnet {
inline constexpr const char* kGroupedValueContextShader=R"METAL(
kernel void grouped_context2(device const float* probabilities [[buffer(0)]],device const float* value [[buffer(1)]],
 device const float* gate [[buffer(2)]],device const float* slots [[buffer(3)]],device float* output [[buffer(4)]],constant AttentionParams& p [[buffer(5)]],
 uint tid [[thread_index_in_threadgroup]],uint tg [[threadgroup_position_in_grid]]) {
 ulong row=tg/(p.heads/2),head=(tg%(p.heads/2))*2,col=tid,kv=head/(p.heads/p.kvHeads);
 const float slot=slots[row*p.sStride];
 const ulong visible=isfinite(slot)&&slot>=0.0f&&slot<float(p.capacity)?ulong(slot)+1:p.capacity;
 float r0=0.0f;
 float r1=0.0f;
 for(ulong s=0;s<visible;++s){float v=value[row*p.vStride+(kv*p.capacity+s)*p.dim+col];
 r0+=probabilities[row*p.pStride+(head+0)*p.capacity+s]*v;
 r1+=probabilities[row*p.pStride+(head+1)*p.capacity+s]*v;
 }
 output[row*p.oStride+(head+0)*p.dim+col]=p.gated?r0/(1.0f+exp(-gate[row*p.gStride+(head+0)*p.dim+col])):r0;
 output[row*p.oStride+(head+1)*p.dim+col]=p.gated?r1/(1.0f+exp(-gate[row*p.gStride+(head+1)*p.dim+col])):r1;
}

)METAL";
}
