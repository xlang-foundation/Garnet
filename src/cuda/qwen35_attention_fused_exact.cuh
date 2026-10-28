// SPDX-License-Identifier: Apache-2.0
// Default-off exact FP32 attention. See SupportsQwen35AttentionCoreFusedExact.
// Original dot, reduction, and context FMA order; no fast-math or precision change.
#pragma once
#include "qwen35_attention_core.cuh"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
#include <algorithm>
#include <cstdint>
namespace Garnet { namespace AttentionFusedExact {
__device__ float warpSum(float v){
    for(int d=16;d;d>>=1)v+=__shfl_down_sync(0xffffffff,v,d);return v;
}
// scores() writes exactly -1e30f at k>slot. With finite maximum>-1e20f,
// its FP32 difference is far below expf's subnormal-underflow range, so the
// original expf result is exactly +0. Keep the original operation otherwise.
__device__ __forceinline__ float maskedExp(float value,float maximum,int k,int slot){
    if(k>slot&&isfinite(maximum)&&maximum>-1e20f)return 0.f;
    return expf(value-maximum);
}
__device__ float blockReduce(float v,bool maximum,float* temporary,float* result){
    for(int d=16;d;d>>=1){float other=__shfl_down_sync(0xffffffff,v,d);v=maximum?fmaxf(v,other):v+other;}
    int lane=threadIdx.x%32,warp=threadIdx.x/32;
    if(!lane)temporary[warp]=v;
    __syncthreads();
    if(!warp){
        v=lane<8?temporary[lane]:(maximum?-CUDART_INF_F:0.f);
        for(int d=16;d;d>>=1){float other=__shfl_down_sync(0xffffffff,v,d);v=maximum?fmaxf(v,other):v+other;}
        if(!lane)*result=v;
    }
    __syncthreads();return *result;
}
__global__ void writeKeys(Qwen35AttentionCoreConfig c,const float* key,const float* value,
    const float* slots,const float* active,float* nextKey,float* nextValue){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=c.batch*c.tokens*c.kv*c.dim)return;
    int d=i%c.dim,head=(i/c.dim)%c.kv,bt=i/(c.kv*c.dim),b=bt/c.tokens;
    if(active[bt]==0)return;
    int slot=int(slots[bt]);if(slot<0||slot>=c.capacity)return;
    size_t target=((size_t(b)*c.kv+head)*c.capacity+slot)*c.dim+d;
    nextKey[target]=key[i];nextValue[target]=value[i];
}
__global__ void scores(Qwen35AttentionCoreConfig c,const float* query,const float* key,
    const float* slots,const float* active,float* scratch){
    int queryRow=blockIdx.y,b=queryRow/(c.tokens*c.heads),t=(queryRow/c.heads)%c.tokens;
    if(active[b*c.tokens+t]==0)return;
    int head=queryRow%c.heads,kh=head/(c.heads/c.kv),lane=threadIdx.x%32,warp=threadIdx.x/32;
    int position=blockIdx.x*8+warp,slot=int(slots[b*c.tokens+t]);
    if(position>=c.capacity)return; // warp-uniform branch
    float sum=0;
    if(position<=slot){
        size_t kbase=((size_t(b)*c.kv+kh)*c.capacity+position)*c.dim;
        for(int d=lane;d<c.dim;d+=32)sum=fmaf(query[size_t(queryRow)*c.dim+d],key[kbase+d],sum);
        sum=warpSum(sum);
    }
    if(!lane)scratch[size_t(queryRow)*c.capacity+position]=position<=slot?sum*rsqrtf(float(c.dim)):-1e30f;
}
// Separate score grid retains key parallelism for small batch decode.
// A context D tile independently applies the identical softmax to its scores.
__global__ void normalizeContext(Qwen35AttentionCoreConfig c,const float* value,const float* gate,
    const float* slots,const float* active,const float* scratch,float* output){
    int row=blockIdx.y,b=row/(c.tokens*c.heads),t=(row/c.heads)%c.tokens;
    int lane=threadIdx.x%32,warp=threadIdx.x/32,d=blockIdx.x*32+lane;
    if(active[b*c.tokens+t]==0){if(!warp)output[size_t(row)*c.dim+d]=0;return;}
    extern __shared__ float storage[];
    float* values=storage;float* partial=values+c.capacity;
    float* temporary=partial+8*32;float* result=temporary+8;
    float maximum=-CUDART_INF_F;
    for(int k=threadIdx.x;k<c.capacity;k+=256){float v=scratch[size_t(row)*c.capacity+k];values[k]=v;maximum=fmaxf(maximum,v);}
    maximum=blockReduce(maximum,true,temporary,result);
    __syncthreads();float sum=0;
    for(int k=threadIdx.x;k<c.capacity;k+=256){float v=maskedExp(values[k],maximum,k,int(slots[b*c.tokens+t]));values[k]=v;sum+=v;}
    sum=blockReduce(sum,false,temporary,result);
    for(int k=threadIdx.x;k<c.capacity;k+=256)values[k]/=sum;
    __syncthreads();
    int head=row%c.heads,kh=head/(c.heads/c.kv);sum=0;
    int end=min(c.capacity,int(slots[b*c.tokens+t])+1);
    for(int k=warp;k<end;k+=8)sum=fmaf(values[k],value[((size_t(b)*c.kv+kh)*c.capacity+k)*c.dim+d],sum);
    partial[warp*32+lane]=sum;__syncthreads();
    if(!warp){sum=0;for(int w=0;w<8;++w)sum+=partial[w*32+lane];output[size_t(row)*c.dim+d]=sum/(1.f+expf(-gate[size_t(row)*c.dim+d]));}
}
// Warp-aligned vectors cover each output KV word exactly once. This runs
// before any inactive-query early return. Attention never reads these outputs.
template<int KV> __device__ void copyKVOverlay4(Qwen35AttentionCoreConfig c,
    const float* newKey,const float* newValue,const float* oldKey,const float* oldValue,
    const float* slots,const float* active,float* nextKey,float* nextValue){
    size_t block=size_t(blockIdx.y)*gridDim.x+blockIdx.x;
    size_t stride=size_t(gridDim.x)*gridDim.y*blockDim.x;
    size_t count=size_t(c.batch)*KV*512*64;
    int lane=threadIdx.x%32;
    for(size_t i=block*blockDim.x+threadIdx.x;i<count;i+=stride){
        int d4=int(i&63),position=int((i>>6)&511),group=int(i>>15);
        int b=group/KV,kh=group%KV,token=-1;
        if(!lane){
            #pragma unroll
            for(int u=0;u<4;++u)if(active[b*4+u]!=0&&int(slots[b*4+u])==position)token=u;
        }
        token=__shfl_sync(0xffffffff,token,0);
        float4 key,value;
        if(token>=0){
            size_t src=((size_t(b)*4+token)*KV+kh)*64+d4;
            key=reinterpret_cast<const float4*>(newKey)[src];value=reinterpret_cast<const float4*>(newValue)[src];
        }else{key=reinterpret_cast<const float4*>(oldKey)[i];value=reinterpret_cast<const float4*>(oldValue)[i];}
        reinterpret_cast<float4*>(nextKey)[i]=key;reinterpret_cast<float4*>(nextValue)[i]=value;
    }
}
// The small-query score grid keeps eight independent key dots per CTA.
// Each cache vector is copied exactly once, even for inactive query rows.
// Score reads use immutable old/new inputs; the next kernel reads completed KV.
template<int KV> __global__ void scoresOverlayCopy(Qwen35AttentionCoreConfig c,
    const float* query,const float* newKey,const float* newValue,const float* oldKey,
    const float* oldValue,const float* slots,const float* active,
    float* nextKey,float* nextValue,float* scratch){
    copyKVOverlay4<KV>(c,newKey,newValue,oldKey,oldValue,slots,active,nextKey,nextValue);
    int row=blockIdx.y,b=row/(c.tokens*c.heads),t=(row/c.heads)%c.tokens;
    if(active[b*c.tokens+t]==0)return;
    int head=row%c.heads,kh=head/(c.heads/KV);
    int lane=threadIdx.x%32,warp=threadIdx.x/32,slot=int(slots[b*c.tokens+t]);
    int position=blockIdx.x*8+warp;
    float sum=0;
    if(position<=slot){
        int token=-1;
        if(!lane){
            #pragma unroll
            for(int u=0;u<4;++u)if(active[b*4+u]!=0&&int(slots[b*4+u])==position)token=u;
        }
        token=__shfl_sync(0xffffffff,token,0);
        const float* selected=token>=0?newKey+((size_t(b)*4+token)*KV+kh)*256:
            oldKey+((size_t(b)*KV+kh)*512+position)*256;
        #pragma unroll
        for(int d=lane;d<256;d+=32)sum=fmaf(query[size_t(row)*256+d],selected[d],sum);
        sum=warpSum(sum);
    }
    if(!lane)scratch[size_t(row)*512+position]=position<=slot?sum*rsqrtf(256.f):-1e30f;
}
template<int KV,int Tiles> __global__ void fusedOverlay(Qwen35AttentionCoreConfig c,const float* query,const float* newKey,
    const float* newValue,const float* oldKey,const float* oldValue,const float* gate,
    const float* slots,const float* active,float* nextKey,float* nextValue,float* output){
    copyKVOverlay4<KV>(c,newKey,newValue,oldKey,oldValue,slots,active,nextKey,nextValue);
    int row=blockIdx.y,b=row/(c.tokens*c.heads),t=(row/c.heads)%c.tokens;
    int firstD=blockIdx.x*Tiles*32,localWidth=min(Tiles*32,c.dim-firstD);
    int lane=threadIdx.x%32,warp=threadIdx.x/32;
    if(active[b*c.tokens+t]==0){
        for(int localD=threadIdx.x;localD<localWidth;localD+=256)output[size_t(row)*c.dim+firstD+localD]=0;
        return;
    }
    int head=row%c.heads,kh=head/(c.heads/c.kv),slot=int(slots[b*c.tokens+t]);
    extern __shared__ float storage[];
    int* lookup=reinterpret_cast<int*>(storage);
    float* values=storage+c.capacity;
    float* partial=values+c.capacity;
    float* temporary=partial+8*localWidth;
    float* result=temporary+8;
    for(int p=threadIdx.x;p<c.capacity;p+=256)lookup[p]=-1;
    __syncthreads();
    if(!threadIdx.x){
        for(int u=0;u<4;++u){int s=int(slots[b*4+u]);if(active[b*4+u]!=0&&s>=0&&s<c.capacity)lookup[s]=u;}
    }
    __syncthreads();
    // A warp owns each key dot product in exactly the old lane/FMA order.
    for(int position=warp;position<c.capacity;position+=8){
        float sum=0;
        if(position<=slot){
            int token=lookup[position];
            const float* selected=token>=0?newKey+((size_t(b)*4+token)*KV+kh)*c.dim:
                oldKey+((size_t(b)*KV+kh)*c.capacity+position)*c.dim;
            for(int d=lane;d<c.dim;d+=32)sum=fmaf(query[size_t(row)*c.dim+d],selected[d],sum);
            sum=warpSum(sum);
        }
        if(!lane)values[position]=position<=slot?sum*rsqrtf(float(c.dim)):-1e30f;
    }
    __syncthreads();
    // Same 256-thread striding and both original reduction trees.
    float maximum=-CUDART_INF_F;
    for(int k=threadIdx.x;k<c.capacity;k+=256)maximum=fmaxf(maximum,values[k]);
    maximum=blockReduce(maximum,true,temporary,result);
    __syncthreads();float sum=0;
    for(int k=threadIdx.x;k<c.capacity;k+=256){float v=maskedExp(values[k],maximum,k,slot);values[k]=v;sum+=v;}
    sum=blockReduce(sum,false,temporary,result);
    for(int k=threadIdx.x;k<c.capacity;k+=256)values[k]/=sum;
    __syncthreads();
    // Keep the old per-dimension key order and eight-warp final sum order.
    // Each warp now handles all D tiles, reusing each probability load.
    float accum[Tiles]={};
    int end=min(c.capacity,slot+1);
    for(int k=warp;k<end;k+=8){
        float weight=values[k];
        int token=lookup[k];
        const float* selectedV=token>=0?newValue+((size_t(b)*4+token)*KV+kh)*c.dim:
            oldValue+((size_t(b)*KV+kh)*c.capacity+k)*c.dim;
        #pragma unroll
        for(int tile=0;tile<Tiles;++tile){
            if(tile<localWidth/32){
                int d=firstD+tile*32+lane;
                accum[tile]=fmaf(weight,selectedV[d],accum[tile]);
            }
        }
    }
    #pragma unroll
    for(int tile=0;tile<Tiles;++tile)if(tile<localWidth/32)partial[warp*localWidth+tile*32+lane]=accum[tile];
    __syncthreads();
    if(!warp){
        #pragma unroll
        for(int tile=0;tile<Tiles;++tile){
            if(tile<localWidth/32){
                int d=firstD+tile*32+lane;float total=0;
                for(int w=0;w<8;++w)total+=partial[w*localWidth+tile*32+lane];
                output[size_t(row)*c.dim+d]=total/(1.f+expf(-gate[size_t(row)*c.dim+d]));
            }
        }
    }
}
__global__ void fused(Qwen35AttentionCoreConfig c,const float* query,const float* key,
    const float* value,const float* gate,const float* slots,const float* active,float* output){
    int row=blockIdx.x,b=row/(c.tokens*c.heads),t=(row/c.heads)%c.tokens;
    int lane=threadIdx.x%32,warp=threadIdx.x/32;
    if(active[b*c.tokens+t]==0){
        for(int d=threadIdx.x;d<c.dim;d+=256)output[size_t(row)*c.dim+d]=0;
        return;
    }
    int head=row%c.heads,kh=head/(c.heads/c.kv),slot=int(slots[b*c.tokens+t]);
    extern __shared__ float storage[];
    float* values=storage;
    float* partial=values+c.capacity;
    float* temporary=partial+8*c.dim;
    float* result=temporary+8;
    // A warp owns each key dot product in exactly the old lane/FMA order.
    for(int position=warp;position<c.capacity;position+=8){
        float sum=0;
        if(position<=slot){
            size_t kbase=((size_t(b)*c.kv+kh)*c.capacity+position)*c.dim;
            for(int d=lane;d<c.dim;d+=32)sum=fmaf(query[size_t(row)*c.dim+d],key[kbase+d],sum);
            sum=warpSum(sum);
        }
        if(!lane)values[position]=position<=slot?sum*rsqrtf(float(c.dim)):-1e30f;
    }
    __syncthreads();
    // Same 256-thread striding and both original reduction trees.
    float maximum=-CUDART_INF_F;
    for(int k=threadIdx.x;k<c.capacity;k+=256)maximum=fmaxf(maximum,values[k]);
    maximum=blockReduce(maximum,true,temporary,result);
    __syncthreads();float sum=0;
    for(int k=threadIdx.x;k<c.capacity;k+=256){float v=maskedExp(values[k],maximum,k,int(slots[b*c.tokens+t]));values[k]=v;sum+=v;}
    sum=blockReduce(sum,false,temporary,result);
    for(int k=threadIdx.x;k<c.capacity;k+=256)values[k]/=sum;
    __syncthreads();
    // Keep the old per-dimension key order and eight-warp final sum order.
    // Each warp now handles all D tiles, reusing each probability load.
    float accum[8]={0,0,0,0,0,0,0,0};
    int end=min(c.capacity,slot+1);
    for(int k=warp;k<end;k+=8){
        float weight=values[k];
        #pragma unroll
        for(int tile=0;tile<8;++tile){
            if(tile<c.dim/32){
                int d=tile*32+lane;
                accum[tile]=fmaf(weight,value[((size_t(b)*c.kv+kh)*c.capacity+k)*c.dim+d],accum[tile]);
            }
        }
    }
    #pragma unroll
    for(int tile=0;tile<8;++tile)if(tile<c.dim/32)partial[warp*c.dim+tile*32+lane]=accum[tile];
    __syncthreads();
    if(!warp){
        #pragma unroll
        for(int tile=0;tile<8;++tile){
            if(tile<c.dim/32){
                int d=tile*32+lane;float total=0;
                for(int w=0;w<8;++w)total+=partial[w*c.dim+d];
                output[size_t(row)*c.dim+d]=total/(1.f+expf(-gate[size_t(row)*c.dim+d]));
            }
        }
    }
}
inline int run(const Qwen35AttentionCoreConfig& c,const void* const* in,
    void* const* out,void* workspace,cudaStream_t stream) noexcept {
    if(!Qwen35AttentionCoreWorkspace(c)||!in||!out||!workspace)return 1;
    for(int i=0;i<8;++i)if(!in[i])return 1;
    for(int i=0;i<3;++i)if(!out[i])return 1;
    int updates=c.batch*c.tokens*c.kv*c.dim,rows=c.batch*c.tokens*c.heads;
    std::uintptr_t alignment=reinterpret_cast<std::uintptr_t>(in[1])|reinterpret_cast<std::uintptr_t>(in[2])|
        reinterpret_cast<std::uintptr_t>(in[4])|reinterpret_cast<std::uintptr_t>(in[5])|
        reinterpret_cast<std::uintptr_t>(out[1])|reinterpret_cast<std::uintptr_t>(out[2]);
    bool small=c.tokens==4&&c.dim==256&&c.capacity==512&&rows<128&&(c.kv==2||c.kv==4)&&(alignment&15)==0;
    if(small&&rows<=32){
        if(c.kv==2)scoresOverlayCopy<2><<<dim3(64,rows),256,0,stream>>>(c,(const float*)in[0],(const float*)in[1],(const float*)in[2],(const float*)in[4],(const float*)in[5],(const float*)in[6],(const float*)in[7],(float*)out[1],(float*)out[2],(float*)workspace);
        else scoresOverlayCopy<4><<<dim3(64,rows),256,0,stream>>>(c,(const float*)in[0],(const float*)in[1],(const float*)in[2],(const float*)in[4],(const float*)in[5],(const float*)in[6],(const float*)in[7],(float*)out[1],(float*)out[2],(float*)workspace);
        size_t shared=(size_t(c.capacity)+8*32+9)*sizeof(float);
        normalizeContext<<<dim3(8,rows),256,shared,stream>>>(c,(const float*)out[2],(const float*)in[3],(const float*)in[6],(const float*)in[7],(const float*)workspace,(float*)out[0]);
    }else if(small){
        int tiles=4;int parts=8/tiles;
        size_t shared=(2*size_t(c.capacity)+8*tiles*32+9)*sizeof(float);
        if(c.kv==2&&tiles==2)fusedOverlay<2,2><<<dim3(parts,rows),256,shared,stream>>>(c,(const float*)in[0],(const float*)in[1],(const float*)in[2],(const float*)in[4],(const float*)in[5],(const float*)in[3],(const float*)in[6],(const float*)in[7],(float*)out[1],(float*)out[2],(float*)out[0]);
        else if(c.kv==2)fusedOverlay<2,4><<<dim3(parts,rows),256,shared,stream>>>(c,(const float*)in[0],(const float*)in[1],(const float*)in[2],(const float*)in[4],(const float*)in[5],(const float*)in[3],(const float*)in[6],(const float*)in[7],(float*)out[1],(float*)out[2],(float*)out[0]);
        else if(tiles==2)fusedOverlay<4,2><<<dim3(parts,rows),256,shared,stream>>>(c,(const float*)in[0],(const float*)in[1],(const float*)in[2],(const float*)in[4],(const float*)in[5],(const float*)in[3],(const float*)in[6],(const float*)in[7],(float*)out[1],(float*)out[2],(float*)out[0]);
        else fusedOverlay<4,4><<<dim3(parts,rows),256,shared,stream>>>(c,(const float*)in[0],(const float*)in[1],(const float*)in[2],(const float*)in[4],(const float*)in[5],(const float*)in[3],(const float*)in[6],(const float*)in[7],(float*)out[1],(float*)out[2],(float*)out[0]);
    }else{
        size_t bytes=size_t(c.batch)*c.kv*c.capacity*c.dim*sizeof(float);
        if(cudaMemcpyAsync(out[1],in[4],bytes,cudaMemcpyDeviceToDevice,stream)!=cudaSuccess||cudaMemcpyAsync(out[2],in[5],bytes,cudaMemcpyDeviceToDevice,stream)!=cudaSuccess)return 1;
        writeKeys<<<(updates+255)/256,256,0,stream>>>(c,(const float*)in[1],(const float*)in[2],(const float*)in[6],(const float*)in[7],(float*)out[1],(float*)out[2]);
        size_t shared=(size_t(c.capacity)+8*c.dim+9)*sizeof(float);
        fused<<<rows,256,shared,stream>>>(c,(const float*)in[0],(const float*)out[1],(const float*)out[2],(const float*)in[3],(const float*)in[6],(const float*)in[7],(float*)out[0]);
    }
    return cudaPeekAtLastError()==cudaSuccess?0:1;
}
}}
