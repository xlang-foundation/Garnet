// SPDX-License-Identifier: Apache-2.0
#include "qwen38_bf16_prefill128_commit.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <limits>
namespace Garnet { namespace {
constexpr int64_t linearWords=827392,kvWords=2097152;
constexpr int64_t canonicalGroup=3*linearWords+2*kvWords;
constexpr int64_t targetGroup=3*linearWords+2*128*1024;
__global__ void copyLinear(const float* target,float* canonical,Qwen38BF16KVSpans spans){
    const int layer=blockIdx.y;const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=linearWords||!spans.count[0])return;
    const int group=layer/3,sub=layer%3;
    canonical[group*canonicalGroup+sub*linearWords+i]=target[128*5120+group*targetGroup+sub*linearWords+i];
}
__global__ void scatterKV(const float* target,float* canonical,Qwen38BF16KVSpans spans){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=16*2*128*4*256)return;
    const int d=i%256;i/=256;const int head=i%4;i/=4;
    const int t=i%128;i/=128;const int which=i%2,layer=i/2;
    if(t>=spans.count[0])return;
    const int64_t cache=layer*canonicalGroup+3*linearWords+which*kvWords;
    const int64_t destination=2*cache+(int64_t(head)*4096+spans.start[0]+t)*256+d;
    const int64_t source=128*5120+layer*targetGroup+3*linearWords+which*128*1024+(t*4+head)*256+d;
    reinterpret_cast<__nv_bfloat16*>(canonical)[destination]=__float2bfloat16_rn(target[source]);
}
bool overlap(const void* a,size_t n,const void* b,size_t m){
    const auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b),maximum=(std::numeric_limits<uintptr_t>::max)();
    return !a||!b||x>maximum-n||y>maximum-m||(x<y+m&&y<x+n);
}
}
int RunQwen38BF16Prefill128Commit(const float* target,float* canonical,int batch,
    int chunk,Qwen38BF16KVSpans spans,cudaStream_t stream) noexcept {
    if(!BF16Prefill128CommitProfile(batch,chunk)||!target||!canonical)return 1;
    if(reinterpret_cast<uintptr_t>(target)%alignof(float)||reinterpret_cast<uintptr_t>(canonical)%alignof(float))return 1;
    if(spans.start[0]<0||spans.start[0]>4096||spans.count[0]<0||spans.count[0]>128||spans.count[0]>4096-spans.start[0])return 1;
    for(int b=1;b<8;++b)if(spans.start[b]||spans.count[b])return 1;
    if(overlap(canonical,size_t(Qwen38BF16KVCanonicalWidth)*4,target,size_t(Qwen38BF16Prefill128TargetWidth)*4))return 1;
    if(!spans.count[0])return 0;
    copyLinear<<<dim3((linearWords+255)/256,48),256,0,stream>>>(target,canonical,spans);
    if(cudaPeekAtLastError()!=cudaSuccess)return 1;
    scatterKV<<<(16*2*128*4*256+255)/256,256,0,stream>>>(target,canonical,spans);
    return cudaPeekAtLastError()==cudaSuccess?0:1;
}
}
