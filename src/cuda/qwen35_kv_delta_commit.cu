// SPDX-License-Identifier: Apache-2.0
#include "qwen35_kv_delta_commit.cuh"
#include <cuda_runtime.h>
#include <limits>
namespace Garnet { namespace {
__global__ void copyLinear(const float* source,int64_t sourceWidth,float* state,
    const Qwen35KVDeltaLinearLayout* layout,Qwen35KVDeltaSpans spans){
    const int b=blockIdx.y/48,layer=blockIdx.y%48;
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=Qwen35KVLinearWidth||!spans.count[b])return;
    const auto p=layout[layer];
    const int64_t offset=sourceWidth==Qwen35KVTargetWidth?p.target:p.replay;
    state[int64_t(b)*Qwen35KVCanonicalWidth+p.canonical+i]=source[int64_t(b)*sourceWidth+offset+i];
}
__global__ void scatterKV(const float* target,float* state,const Qwen35KVDeltaFullLayout* layout,Qwen35KVDeltaSpans spans){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=4*16*2*4*4*256)return;
    const int d=i%256;i/=256;const int head=i%4;i/=4;const int t=i%4;i/=4;
    const int which=i%2;i/=2;const int layer=i%16,b=i/16;
    if(t>=spans.count[b])return;
    const auto p=layout[layer];
    const int64_t destination=p.canonical+((int64_t(which)*4+head)*4096+spans.start[b]+t)*256+d;
    const int64_t source=p.target+int64_t(which)*4096+(t*4+head)*256+d;
    state[int64_t(b)*Qwen35KVCanonicalWidth+destination]=target[int64_t(b)*Qwen35KVTargetWidth+source];
}
bool overlap(const void* a,size_t n,const void* b,size_t m){
    const auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b),maximum=(std::numeric_limits<uintptr_t>::max)();
    return !a||!b||x>maximum-n||y>maximum-m||(x<y+m&&y<x+n);
}
}
int RunQwen35KVDeltaCommit(const float* target,const float* linear,int64_t linearWidth,float* canonical,
    const Qwen35KVDeltaLinearLayout* layout,const Qwen35KVDeltaFullLayout* full,Qwen35KVDeltaSpans spans,cudaStream_t stream) noexcept {
    if(!target||!linear||!canonical||!layout||!full||(linearWidth!=Qwen35KVTargetWidth&&linearWidth!=Qwen35KVReplayWidth)||
        (linearWidth==Qwen35KVTargetWidth&&linear!=target))return 1;
    for(int b=0;b<4;++b)if(spans.start[b]<0||spans.start[b]>4096||spans.count[b]<0||spans.count[b]>4||spans.count[b]>4096-spans.start[b])return 1;
    const size_t stateBytes=size_t(4)*Qwen35KVCanonicalWidth*sizeof(float);
    const void* reads[]={target,linear,layout,full};
    const size_t sizes[]={size_t(4)*Qwen35KVTargetWidth*sizeof(float),size_t(4)*linearWidth*sizeof(float),48*sizeof(*layout),16*sizeof(*full)};
    for(int i=0;i<4;++i)if(overlap(canonical,stateBytes,reads[i],sizes[i]))return 1;
    copyLinear<<<dim3((Qwen35KVLinearWidth+255)/256,4*48),256,0,stream>>>(linear,linearWidth,canonical,layout,spans);
    if(cudaPeekAtLastError()!=cudaSuccess)return 1;
    scatterKV<<<(4*16*2*4*4*256+255)/256,256,0,stream>>>(target,canonical,full,spans);
    return cudaPeekAtLastError()==cudaSuccess?0:1;
}
}
