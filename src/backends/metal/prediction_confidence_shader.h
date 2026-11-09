// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace Garnet {
// The first two values preserve argmax_rows' ID/nonfinite-flag contract.
// The two scores correspond to distinct vocabulary indices; ties have gap0.
inline constexpr const char* kPredictionConfidenceShader=R"METAL(
#include <metal_stdlib>
using namespace metal;
struct ConfidenceParams {ulong rows,columns,iStride,oStride;};
kernel void prediction_confidence_rows(device const float* input [[buffer(0)]],
 device float* output [[buffer(1)]],constant ConfidenceParams& p [[buffer(2)]],
 uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
 uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
 if(row>=p.rows)return;
 threadgroup float maximum[4],runnerUp[4];
 threadgroup uint indices[4],invalid[4];
 float best=-INFINITY,second=-INFINITY;uint index=0xffffffff,bad=0;
 for(ulong j=tid;j<p.columns;j+=128) {
  float value=input[row*p.iStride+j];bad|=!isfinite(value);
  if(value>best||(value==best&&j<index)) {
   second=max(second,best);best=value;index=uint(j);
  } else second=max(second,value);
 }
 float groupBest=simd_max(best);
 uint groupIndex=simd_min(best==groupBest?index:0xffffffff);
 float groupSecond=simd_max(max(second,index!=groupIndex?best:-INFINITY));
 uint groupBad=simd_max(bad);
 if(lane==0){maximum[simd]=groupBest;runnerUp[simd]=groupSecond;indices[simd]=groupIndex;invalid[simd]=groupBad;}
 threadgroup_barrier(mem_flags::mem_threadgroup);
 best=lane<4?maximum[lane]:-INFINITY;second=lane<4?runnerUp[lane]:-INFINITY;
 index=lane<4?indices[lane]:0xffffffff;bad=lane<4?invalid[lane]:0;
 groupBest=simd_max(best);groupIndex=simd_min(best==groupBest?index:0xffffffff);
 groupSecond=simd_max(max(second,index!=groupIndex?best:-INFINITY));groupBad=simd_max(bad);
 if(tid==0){output[row*p.oStride]=float(groupIndex);output[row*p.oStride+1]=float(groupBad!=0);
  output[row*p.oStride+2]=groupBest;output[row*p.oStride+3]=groupSecond;}
}
)METAL";
}
