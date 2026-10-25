// SPDX-License-Identifier: Apache-2.0
#include "bf16_low_batch_commit_candidate.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <limits>
namespace Garnet { namespace {
constexpr int64_t linearWords=827392,kvWords=2097152;
constexpr int64_t canonicalGroup=3*linearWords+2*kvWords;
__global__ void copyLinearBF16KV(const float* source,int64_t sourceWidth,
    float* canonical,int chunk,int64_t targetWidth,Qwen38BF16KVSpans spans){
    const int b=blockIdx.y/48,layer=blockIdx.y%48;
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=linearWords||!spans.count[b])return;
    const int group=layer/3,sub=layer%3;
    const int64_t destination=group*canonicalGroup+sub*linearWords;
    const int64_t origin=sourceWidth==targetWidth?
        chunk*5120+group*(3*linearWords+2*chunk*1024)+sub*linearWords:
        layer*linearWords;
    canonical[int64_t(b)*Qwen38BF16KVCanonicalWidth+destination+i]=
        source[int64_t(b)*sourceWidth+origin+i];
}
__global__ void scatterBF16KV(const float* target,float* canonical,int batch,int chunk,
    int64_t targetWidth,Qwen38BF16KVSpans spans){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=batch*16*2*chunk*4*256)return;
    const int d=i%256;i/=256;const int head=i%4;i/=4;
    const int t=i%chunk;i/=chunk;const int which=i%2;i/=2;
    const int layer=i%16,b=i/16;
    if(t>=spans.count[b])return;
    const int64_t cache=layer*canonicalGroup+3*linearWords+which*kvWords;
    const int64_t destination=2*(int64_t(b)*Qwen38BF16KVCanonicalWidth+cache)+
        (int64_t(head)*4096+spans.start[b]+t)*256+d;
    const int64_t source=int64_t(b)*targetWidth+chunk*5120+
        layer*(3*linearWords+2*chunk*1024)+3*linearWords+
        which*chunk*1024+(t*4+head)*256+d;
    reinterpret_cast<__nv_bfloat16*>(canonical)[destination]=__float2bfloat16_rn(target[source]);
}
bool overlap(const void* a,size_t n,const void* b,size_t m){
    const auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);
    const auto maximum=(std::numeric_limits<uintptr_t>::max)();
    return !a||!b||x>maximum-n||y>maximum-m||(x<y+m&&y<x+n);
}
}
int RunQwen38BF16LowBatchCommitCandidate(const float* target,const float* linear,int64_t linearWidth,
    float* canonical,int batch,int chunk,Qwen38BF16KVSpans spans,cudaStream_t stream) noexcept {
    if(!BF16LowBatchCommitCandidateProfile(batch,chunk)||!target||!linear||!canonical)return 1;
    const int64_t targetWidth=chunk==4?43030528:40927232;
    if(linearWidth!=targetWidth&&(chunk!=4||linearWidth!=Qwen38BF16KVReplayWidth))return 1;
    if(linearWidth==targetWidth&&linear!=target)return 1;
    bool any=false;
    for(int b=0;b<8;++b){
        if(b>=batch){if(spans.start[b]||spans.count[b])return 1;continue;}
        if(spans.start[b]<0||spans.start[b]>4096||spans.count[b]<0||
            spans.count[b]>chunk||spans.count[b]>4096-spans.start[b])return 1;
        any=any||spans.count[b]!=0;
    }
    const size_t stateBytes=size_t(batch)*Qwen38BF16KVCanonicalWidth*sizeof(float);
    if(overlap(canonical,stateBytes,target,size_t(batch)*targetWidth*sizeof(float))||
        overlap(canonical,stateBytes,linear,size_t(batch)*linearWidth*sizeof(float)))return 1;
    if(!any)return 0;
    copyLinearBF16KV<<<dim3((linearWords+255)/256,batch*48),256,0,stream>>>(
        linear,linearWidth,canonical,chunk,targetWidth,spans);
    if(cudaPeekAtLastError()!=cudaSuccess)return 1;
    scatterBF16KV<<<(batch*16*2*chunk*4*256+255)/256,256,0,stream>>>(
        target,canonical,batch,chunk,targetWidth,spans);
    return cudaPeekAtLastError()==cudaSuccess?0:1;
}
}
