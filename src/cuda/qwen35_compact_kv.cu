// SPDX-License-Identifier: Apache-2.0
#include "qwen35_compact_kv.cuh"
#include <cuda_runtime.h>
#include <cstdint>
namespace Garnet { namespace {
bool valid(Qwen35CompactKVConfig c){return (c.batch==1||c.batch==2||c.batch==4||c.batch==8)&&
    c.kv>=1&&c.kv<=32&&c.capacity>=32&&c.capacity<=8192&&c.tokens>=1&&c.tokens<=128&&c.tokens<=c.capacity&&c.dim>=1&&c.dim<=256;}
__global__ void pack(Qwen35CompactKVConfig c,const float* key,const float* value,const float* slots,const float* active,float* dk,float* dv,int* error){
    size_t words=size_t(c.batch)*c.kv*c.tokens*c.dim;
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<words;i+=size_t(gridDim.x)*blockDim.x){
        int d=int(i%c.dim),t=int(i/c.dim%c.tokens),h=int(i/(size_t(c.dim)*c.tokens)%c.kv),b=int(i/(size_t(c.dim)*c.tokens*c.kv));
        float a=active[b*c.tokens+t],s=slots[b*c.tokens+t];float k=0,v=0;
        if(a==1){if(isfinite(s)&&s>=0&&s<c.capacity&&s==floorf(s)){size_t j=((size_t(b)*c.kv+h)*c.capacity+int(s))*c.dim+d;k=key[j];v=value[j];}else atomicExch(error,1);}
        else if(a!=0)atomicExch(error,1);
        dk[i]=k;dv[i]=v;
    }
}
__global__ void commit(Qwen35CompactKVConfig c,Qwen35CompactKVSpans spans,const float* dk,const float* dv,float* key,float* value){
    size_t words=size_t(c.batch)*c.kv*c.tokens*c.dim;
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<words;i+=size_t(gridDim.x)*blockDim.x){
        int d=int(i%c.dim),t=int(i/c.dim%c.tokens),h=int(i/(size_t(c.dim)*c.tokens)%c.kv),b=int(i/(size_t(c.dim)*c.tokens*c.kv));
        if(t<spans.count[b]){size_t j=((size_t(b)*c.kv+h)*c.capacity+spans.start[b]+t)*c.dim+d;key[j]=dk[i];value[j]=dv[i];}
    }
}
int blocks(Qwen35CompactKVConfig c){size_t n=(size_t(c.batch)*c.kv*c.tokens*c.dim+255)/256;return int(n<4096?n:4096);}
__global__ void report(const int* error,float* output,int batch){int i=threadIdx.x;if(i<batch)output[i]=*error?1.0f:0.0f;}
}
cudaError_t PackQwen35CompactKV(Qwen35CompactKVConfig c,const float* k,const float* v,const float* s,const float* a,float* dk,float* dv,int* error,cudaStream_t stream){
    if(!valid(c)||!k||!v||!s||!a||!dk||!dv||!error)return cudaErrorInvalidValue;
    pack<<<blocks(c),256,0,stream>>>(c,k,v,s,a,dk,dv,error);return cudaGetLastError();
}
cudaError_t CommitQwen35CompactKV(Qwen35CompactKVConfig c,Qwen35CompactKVSpans spans,const float* dk,const float* dv,float* k,float* v,cudaStream_t stream){
    if(!valid(c)||!dk||!dv||!k||!v)return cudaErrorInvalidValue;
    for(int b=0;b<c.batch;++b)if(spans.start[b]<0||spans.start[b]>c.capacity||spans.count[b]<0||spans.count[b]>c.tokens||spans.count[b]>c.capacity-spans.start[b])return cudaErrorInvalidValue;
    commit<<<blocks(c),256,0,stream>>>(c,spans,dk,dv,k,v);return cudaGetLastError();
}
cudaError_t ReportQwen35CompactKVControl(const int* error,float* output,int batch,cudaStream_t stream){
    if(!error||!output||(batch!=1&&batch!=2&&batch!=4&&batch!=8))return cudaErrorInvalidValue;
    report<<<1,32,0,stream>>>(error,output,batch);return cudaGetLastError();
}
}
