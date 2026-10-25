// SPDX-License-Identifier: Apache-2.0
#include "qwen38_bf16_kv_overlay.cuh"
#include "qwen38_bf16_kv_commit.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <limits>
namespace Garnet { namespace {
bool valid(Qwen35AttentionCoreConfig c){
    return (c.batch==4||c.batch==8)&&(c.tokens==4||c.tokens==32)&&
        c.heads==24&&c.kv==4&&c.capacity==4096&&c.dim==256;
}
__device__ float warpSum(float v){
    for(int d=16;d;d>>=1)v+=__shfl_down_sync(0xffffffff,v,d);return v;
}
__device__ float reduce(float v,bool maximum,float* temporary,float* result){
    for(int d=16;d;d>>=1){float other=__shfl_down_sync(0xffffffff,v,d);v=maximum?fmaxf(v,other):v+other;}
    const int lane=threadIdx.x%32,warp=threadIdx.x/32;
    if(!lane)temporary[warp]=v;
    __syncthreads();
    if(!warp){
        v=lane<8?temporary[lane]:(maximum?-CUDART_INF_F:0.f);
        for(int d=16;d;d>>=1){float other=__shfl_down_sync(0xffffffff,v,d);v=maximum?fmaxf(v,other):v+other;}
        if(!lane)*result=v;
    }
    __syncthreads();return *result;
}
__device__ int replacement(const float* slots,const float* active,int batch,int tokens,int position){
    for(int t=0;t<tokens;++t)if(active[batch*tokens+t]!=0&&int(slots[batch*tokens+t])==position)return t;
    return -1;
}
__device__ float readKV(const __nv_bfloat16* cache,const float* chunk,
    int b,int t,int h,int p,int d,int tokens){
    if(t>=0)return __bfloat162float(__float2bfloat16_rn(chunk[((size_t(b)*tokens+t)*4+h)*256+d]));
    return __bfloat162float(cache[size_t(b)*Qwen38BF16KVCanonicalWidth*2+(size_t(h)*4096+p)*256+d]);
}
__global__ void scores(Qwen35AttentionCoreConfig c,const float* query,const __nv_bfloat16* key,
    const float* chunkKey,const float* slots,const float* active,float* scratch){
    const int row=blockIdx.y,b=row/(c.tokens*c.heads),t=(row/c.heads)%c.tokens;
    if(active[b*c.tokens+t]==0)return;
    const int lane=threadIdx.x%32,warp=threadIdx.x/32,p=blockIdx.x*8+warp;
    const int slot=int(slots[b*c.tokens+t]),kh=(row%c.heads)/(c.heads/c.kv);
    if(p>=c.capacity)return;
    float sum=0;
    if(p<=slot){
        const int current=replacement(slots,active,b,c.tokens,p);
        for(int d=lane;d<c.dim;d+=32)sum=fmaf(query[size_t(row)*c.dim+d],readKV(key,chunkKey,b,current,kh,p,d,c.tokens),sum);
        sum=warpSum(sum);
    }
    if(!lane)scratch[size_t(row)*c.capacity+p]=p<=slot?sum*rsqrtf(float(c.dim)):-1e30f;
}
__global__ void normalize(Qwen35AttentionCoreConfig c,const float* active,float* scratch){
    const int row=blockIdx.x,b=row/(c.tokens*c.heads),t=(row/c.heads)%c.tokens;
    if(active[b*c.tokens+t]==0)return;
    __shared__ float temporary[8],result;float maximum=-CUDART_INF_F;
    float* values=scratch+size_t(row)*c.capacity;
    for(int k=threadIdx.x;k<c.capacity;k+=256)maximum=fmaxf(maximum,values[k]);
    maximum=reduce(maximum,true,temporary,&result);__syncthreads();float sum=0;
    for(int k=threadIdx.x;k<c.capacity;k+=256){float v=expf(values[k]-maximum);values[k]=v;sum+=v;}
    sum=reduce(sum,false,temporary,&result);
    for(int k=threadIdx.x;k<c.capacity;k+=256)values[k]/=sum;
}
__global__ void context(Qwen35AttentionCoreConfig c,const __nv_bfloat16* value,
    const float* chunkValue,const float* gate,const float* slots,const float* active,
    const float* scratch,float* output){
    const int row=blockIdx.y,b=row/(c.tokens*c.heads),t=(row/c.heads)%c.tokens;
    const int lane=threadIdx.x%32,warp=threadIdx.x/32,d=blockIdx.x*32+lane;
    if(active[b*c.tokens+t]==0){if(!warp)output[size_t(row)*c.dim+d]=0;return;}
    const int kh=(row%c.heads)/(c.heads/c.kv),end=min(c.capacity,int(slots[b*c.tokens+t])+1);
    float sum=0;
    for(int k=warp;k<end;k+=8){
        const int current=replacement(slots,active,b,c.tokens,k);
        sum=fmaf(scratch[size_t(row)*c.capacity+k],readKV(value,chunkValue,b,current,kh,k,d,c.tokens),sum);
    }
    __shared__ float partial[8][32];partial[warp][lane]=sum;__syncthreads();
    if(!warp){sum=0;for(int w=0;w<8;++w)sum+=partial[w][lane];output[size_t(row)*c.dim+d]=sum/(1.f+expf(-gate[size_t(row)*c.dim+d]));}
}
bool overlap(const void* a,size_t n,const void* b,size_t m){
    const auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);
    const auto maximum=(std::numeric_limits<uintptr_t>::max)();
    return !a||!b||x>maximum-n||y>maximum-m||(x<y+m&&y<x+n);
}
}
size_t Qwen38BF16KVOverlayWorkspace(const Qwen35AttentionCoreConfig& c) noexcept {
    return valid(c)?size_t(c.batch)*c.tokens*c.heads*c.capacity*sizeof(float):0;
}
int RunQwen38BF16KVOverlay(const Qwen35AttentionCoreConfig& c,const void* const* in,
    void* output,void* workspace,int64_t keyOffset,int64_t valueOffset,cudaStream_t stream) noexcept {
    if(!valid(c)||!in||!output||!workspace||keyOffset<2482176||
        (keyOffset-2482176)%6676480!=0||(keyOffset-2482176)/6676480>=16||
        valueOffset!=keyOffset+2097152)return 1;
    const size_t queryBytes=size_t(c.batch)*c.tokens*c.heads*c.dim*4;
    const size_t chunkBytes=size_t(c.batch)*c.tokens*c.kv*c.dim*4,controlBytes=size_t(c.batch)*c.tokens*4;
    const size_t stateBytes=size_t(c.batch)*Qwen38BF16KVCanonicalWidth*4;
    const size_t sizes[7]={queryBytes,chunkBytes,chunkBytes,queryBytes,stateBytes,controlBytes,controlBytes};
    const size_t scratchBytes=Qwen38BF16KVOverlayWorkspace(c);
    for(int i=0;i<7;++i)if(overlap(output,queryBytes,in[i],sizes[i])||overlap(workspace,scratchBytes,in[i],sizes[i]))return 1;
    if(overlap(output,queryBytes,workspace,scratchBytes))return 1;
    const auto* canonical=static_cast<const float*>(in[4]);
    const auto* key=reinterpret_cast<const __nv_bfloat16*>(canonical+keyOffset);
    const auto* value=reinterpret_cast<const __nv_bfloat16*>(canonical+valueOffset);
    const int rows=c.batch*c.tokens*c.heads;
    scores<<<dim3((c.capacity+7)/8,rows),256,0,stream>>>(c,(const float*)in[0],key,(const float*)in[1],(const float*)in[5],(const float*)in[6],(float*)workspace);
    if(cudaPeekAtLastError()!=cudaSuccess)return 1;
    normalize<<<rows,256,0,stream>>>(c,(const float*)in[6],(float*)workspace);
    if(cudaPeekAtLastError()!=cudaSuccess)return 1;
    context<<<dim3(c.dim/32,rows),256,0,stream>>>(c,value,(const float*)in[2],(const float*)in[3],(const float*)in[5],(const float*)in[6],(const float*)workspace,(float*)output);
    return cudaPeekAtLastError()==cudaSuccess?0:1;
}
}
