// SPDX-License-Identifier: Apache-2.0
// Standalone transactional KV-overlay research kernel, never linked into Garnet.
#include "attention_kv_delta_core.cuh"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
namespace Garnet { namespace {
bool validDelta(Qwen35AttentionCoreConfig c){return c.batch==4&&c.tokens==4&&c.heads==24&&c.kv==4&&c.capacity==4096&&c.dim==256;}
__device__ int overlayToken(Qwen35AttentionCoreConfig c,const float* slots,const float* active,int b,int position){
 for(int t=0;t<c.tokens;++t)if(active[b*c.tokens+t]!=0&&int(slots[b*c.tokens+t])==position)return t;
 return -1;
}
__device__ float warpSum(float v){
    for(int d=16;d;d>>=1)v+=__shfl_down_sync(0xffffffff,v,d);return v;
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
__global__ void overlay(Qwen35AttentionCoreConfig c,const float* query,const float* key,
 const float* value,const float* gate,const float* slots,const float* active,const float* chunkKey,const float* chunkValue,float* output){
 constexpr bool tiled=false;
 const int row=tiled?blockIdx.y:blockIdx.x,b=row/(c.tokens*c.heads),t=(row/c.heads)%c.tokens;
 if(active[b*c.tokens+t]==0){if(tiled){if(threadIdx.x<32)output[size_t(row)*c.dim+blockIdx.x*32+threadIdx.x]=0;}else if(threadIdx.x<c.dim)output[size_t(row)*c.dim+threadIdx.x]=0;return;}
 const int end=min(c.capacity,int(slots[b*c.tokens+t])+1);
 if(end<1){if(tiled){if(threadIdx.x<32)output[size_t(row)*c.dim+blockIdx.x*32+threadIdx.x]=0;}else if(threadIdx.x<c.dim)output[size_t(row)*c.dim+threadIdx.x]=0;return;}
 const int head=row%c.heads,kh=head/(c.heads/c.kv),lane=threadIdx.x%32,warp=threadIdx.x/32;
 __shared__ float values[8192],temporary[8],result,partial[8][32];
 for(int position=warp;position<end;position+=8){
  float sum=0;const size_t base=((size_t(b)*c.kv+kh)*c.capacity+position)*c.dim;
  const int replacement=overlayToken(c,slots,active,b,position);
  const float* rowKey=replacement<0?key+base:chunkKey+((size_t(b)*c.tokens+replacement)*c.kv+kh)*c.dim;
  for(int d=lane;d<c.dim;d+=32)sum=fmaf(query[size_t(row)*c.dim+d],rowKey[d],sum);
  sum=warpSum(sum);if(!lane)values[position]=sum*rsqrtf(float(c.dim));
 }
 __syncthreads();float maximum=-CUDART_INF_F;
 for(int k=threadIdx.x;k<end;k+=256)maximum=fmaxf(maximum,values[k]);
 maximum=blockReduce(maximum,true,temporary,&result);__syncthreads();float total=0;
 for(int k=threadIdx.x;k<end;k+=256){float v=expf(values[k]-maximum);values[k]=v;total+=v;}
 total=blockReduce(total,false,temporary,&result);
 for(int k=threadIdx.x;k<end;k+=256)values[k]/=total;
 __syncthreads();
 for(int tile=tiled?int(blockIdx.x):0;tile<(tiled?int(blockIdx.x)+1:c.dim/32);++tile){
  const int d=tile*32+lane;float sum=0;
  for(int k=warp;k<end;k+=8){
   const int replacement=overlayToken(c,slots,active,b,k);
   const float* rowValue=replacement<0?value+((size_t(b)*c.kv+kh)*c.capacity+k)*c.dim:chunkValue+((size_t(b)*c.tokens+replacement)*c.kv+kh)*c.dim;
   sum=fmaf(values[k],rowValue[d],sum);
  }
  partial[warp][lane]=sum;__syncthreads();
  if(!warp){sum=0;for(int w=0;w<8;++w)sum+=partial[w][lane];output[size_t(row)*c.dim+d]=sum/(1.f+expf(-gate[size_t(row)*c.dim+d]));}
  __syncthreads();
 }
}

__global__ void commitPrefix(Qwen35AttentionCoreConfig c,const float* key,const float* value,const float* slots,const float* active,const float* accepted,float* canonicalKey,float* canonicalValue){
 const int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=c.batch*c.tokens*c.kv*c.dim)return;
 const int d=i%c.dim,head=(i/c.dim)%c.kv,bt=i/(c.kv*c.dim),b=bt/c.tokens,t=bt%c.tokens;
 if(active[bt]==0)return;
 int rank=0;for(int p=0;p<t;++p)rank+=active[b*c.tokens+p]!=0;
 if(rank>=int(accepted[b]))return;
 const int slot=int(slots[bt]);if(slot<0||slot>=c.capacity)return;
 const size_t target=((size_t(b)*c.kv+head)*c.capacity+slot)*c.dim+d;
 canonicalKey[target]=key[i];canonicalValue[target]=value[i];
}
}
int RunQwen35AttentionKVOverlay(const Qwen35AttentionCoreConfig& c,const void* const* in,void* output,cudaStream_t stream) noexcept {
 if(!validDelta(c)||!in||!output)return 1;for(int i=0;i<8;++i)if(!in[i])return 1;
 overlay<<<c.batch*c.tokens*c.heads,256,0,stream>>>(c,(const float*)in[0],(const float*)in[4],(const float*)in[5],(const float*)in[3],(const float*)in[6],(const float*)in[7],(const float*)in[1],(const float*)in[2],(float*)output);
 return cudaGetLastError()==cudaSuccess?0:1;
}
int CommitQwen35AttentionKVPrefix(const Qwen35AttentionCoreConfig& c,const void* const* in,const float* accepted,void* key,void* value,cudaStream_t stream) noexcept {
 if(!validDelta(c)||!in||!accepted||!key||!value)return 1;
 if(!in[1]||!in[2]||!in[6]||!in[7])return 1;
 const int count=c.batch*c.tokens*c.kv*c.dim;
 commitPrefix<<<(count+255)/256,256,0,stream>>>(c,(const float*)in[1],(const float*)in[2],(const float*)in[6],(const float*)in[7],accepted,(float*)key,(float*)value);
 return cudaGetLastError()==cudaSuccess?0:1;
}
}
