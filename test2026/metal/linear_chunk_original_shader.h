// SPDX-License-Identifier: Apache-2.0
// Original kernels from 527357dc2, before inactive-token fast paths.
#pragma once
static constexpr const char* kOriginalLinearChunkShader=R"ORIGINAL(
#include <metal_stdlib>
using namespace metal;
struct ConvChunkParams {ulong batch,chunk,channels,kSize,tStride,wStride,sStride,aStride,oStride,nStride;};
kernel void causal_conv_chunk(device const float* tokens [[buffer(0)]],device const float* weight [[buffer(1)]],
    device const float* state [[buffer(2)]],device const float* active [[buffer(3)]],
    device float* output [[buffer(4)]],device float* nextState [[buffer(5)]],
    constant ConvChunkParams& p [[buffer(6)]],uint id [[thread_position_in_grid]]) {
    if(id>=p.batch*p.channels)return;
    ulong row=id/p.channels,col=id%p.channels;
    float history[8];
    for(ulong j=0;j<p.kSize;++j)history[j]=state[row*p.sStride+col*p.kSize+j];
    for(ulong token=0;token<p.chunk;++token) {
        bool enabled=active[row*p.aStride+token]!=0.0f;
        float sum=0.0f;
        for(ulong j=0;j<p.kSize;++j) {
            float next=j+1==p.kSize?tokens[(row*p.chunk+token)*p.tStride+col]:history[j+1];
            if(enabled)history[j]=next;
            sum+=next*weight[col*p.wStride+j];
        }
        output[(row*p.chunk+token)*p.oStride+col]=enabled?sum/(1.0f+exp(-sum)):0.0f;
    }
    for(ulong j=0;j<p.kSize;++j)nextState[row*p.nStride+col*p.kSize+j]=history[j];
}
struct DeltaChunkParams {ulong batch,chunk,heads,keyDim,valueDim,pStride,sStride,aStride,oStride,nextStride;};
kernel void gated_delta_chunk(device const float* prepared [[buffer(0)]],device const float* state [[buffer(1)]],
    device const float* active [[buffer(2)]],device float* output [[buffer(3)]],device float* nextState [[buffer(4)]],
    constant DeltaChunkParams& p [[buffer(5)]],uint id [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    ulong tiles=p.valueDim/16,headId=ulong(id)/tiles;
    if(headId>=p.batch*p.heads)return;
    ulong head=headId%p.heads,row=headId/p.heads,column=(ulong(id)%tiles)*16+simd*4;
    ulong stateBase=row*p.sStride+head*p.keyDim*p.valueDim,nextBase=row*p.nextStride+head*p.keyDim*p.valueDim;
    float4 memories[4];
    for(uint j=0;j<4;++j)memories[j]=*(device const float4*)(state+stateBase+(ulong(lane)+32*j)*p.valueDim+column);
    threadgroup float qPartial[4],kPartial[4];
    ulong qw=p.heads*p.keyDim,vw=p.heads*p.valueDim;
    for(ulong token=0;token<p.chunk;++token) {
        ulong base=(row*p.chunk+token)*p.pStride,outputBase=(row*p.chunk+token)*p.oStride+head*p.valueDim;
        if(active[row*p.aStride+token]!=0.0f) {
            float q=prepared[base+head*p.keyDim+tid],k=prepared[base+qw+head*p.keyDim+tid];
            float qs=simd_sum(q*q),ks=simd_sum(k*k);
            if(lane==0){qPartial[simd]=qs;kPartial[simd]=ks;}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            qs=simd_sum(lane<4?qPartial[lane]:0.0f)+1e-6f;ks=simd_sum(lane<4?kPartial[lane]:0.0f)+1e-6f;
            float qi=rsqrt(qs)*rsqrt(float(p.keyDim)),ki=rsqrt(ks),factor=exp(prepared[base+2*qw+vw+head]);
            float4 memory=0.0f;
            for(uint j=0;j<4;++j)memory+=memories[j]*factor*prepared[base+qw+head*p.keyDim+ulong(lane)+32*j]*ki;
            memory=float4(simd_sum(memory.x),simd_sum(memory.y),simd_sum(memory.z),simd_sum(memory.w));
            float4 change=(*(device const float4*)(prepared+base+2*qw+head*p.valueDim+column)-memory)*prepared[base+2*qw+vw+p.heads+head];
            float4 result=0.0f;
            for(uint j=0;j<4;++j) {
                ulong keyIndex=ulong(lane)+32*j;
                memories[j]=memories[j]*factor+prepared[base+qw+head*p.keyDim+keyIndex]*ki*change;
                result+=memories[j]*prepared[base+head*p.keyDim+keyIndex]*qi;
            }
            result=float4(simd_sum(result.x),simd_sum(result.y),simd_sum(result.z),simd_sum(result.w));
            if(lane==0)*(device float4*)(output+outputBase+column)=result;
        }else if(lane==0)*(device float4*)(output+outputBase+column)=float4(0.0f);
        // Every SIMD group must finish reading the partial norms before any
        // group can replace them for the next token.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint j=0;j<4;++j)*(device float4*)(nextState+nextBase+(ulong(lane)+32*j)*p.valueDim+column)=memories[j];
}
)ORIGINAL";
