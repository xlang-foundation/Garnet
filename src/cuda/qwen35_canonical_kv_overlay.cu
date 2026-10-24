// SPDX-License-Identifier: Apache-2.0
// Qualified B4/T4 canonical-state kernels; explicit opt-in lowering only.
#include "qwen35_canonical_kv_overlay.cuh"
#include <climits>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
namespace Garnet { namespace {
bool validCached(Qwen35AttentionCoreConfig c){return c.batch==4&&c.tokens==4&&c.heads==24&&c.kv==4&&c.capacity==4096&&c.dim==256;}
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
struct CachedSlots { int s0,s1,s2,s3,lo,hi;
 __device__ __forceinline__ int find(int p) const {
  if(p<lo||p>hi)return -1;
  return p==s0?0:p==s1?1:p==s2?2:p==s3?3:-1;
 }
};
__device__ __forceinline__ CachedSlots loadSlots(const float* slots,const float* active,int b){
 const int offset=b*4;
 const int s0=active[offset]!=0?int(slots[offset]):-1,s1=active[offset+1]!=0?int(slots[offset+1]):-1;
 const int s2=active[offset+2]!=0?int(slots[offset+2]):-1,s3=active[offset+3]!=0?int(slots[offset+3]):-1;
 int lo=4096,hi=-1;
 if(s0>=0){lo=min(lo,s0);hi=max(hi,s0);}if(s1>=0){lo=min(lo,s1);hi=max(hi,s1);}
 if(s2>=0){lo=min(lo,s2);hi=max(hi,s2);}if(s3>=0){lo=min(lo,s3);hi=max(hi,s3);}
 return {s0,s1,s2,s3,lo,hi};
}
__global__ void scoresCached(Qwen35AttentionCoreConfig c,const float* query,const float* key,const float* chunkKey,
    const float* slots,const float* active,float* scratch,bool hybrid,size_t cacheBatchStride){
    int queryRow=blockIdx.y,b=queryRow/(c.tokens*c.heads),t=(queryRow/c.heads)%c.tokens;
    if(active[b*c.tokens+t]==0||(hybrid&&min(c.capacity,int(slots[b*c.tokens+t])+1)<=512))return;
    const CachedSlots controls=loadSlots(slots,active,b);
    int head=queryRow%c.heads,kh=head/(c.heads/c.kv),lane=threadIdx.x%32,warp=threadIdx.x/32;
    int position=blockIdx.x*8+warp,slot=int(slots[b*c.tokens+t]);
    if(position>=c.capacity)return; // warp-uniform branch
    float sum=0;
    if(position<=slot){
        size_t kbase=size_t(b)*cacheBatchStride+(size_t(kh)*c.capacity+position)*c.dim;
        const int replacement=controls.find(position);
        const float* rowKey=replacement<0?key+kbase:chunkKey+((size_t(b)*c.tokens+replacement)*c.kv+kh)*c.dim;
        for(int d=lane;d<c.dim;d+=32)sum=fmaf(query[size_t(queryRow)*c.dim+d],rowKey[d],sum);
        sum=warpSum(sum);
    }
    if(!lane)scratch[size_t(queryRow)*c.capacity+position]=position<=slot?sum*rsqrtf(float(c.dim)):-1e30f;
}
__global__ void normalizeCached(Qwen35AttentionCoreConfig c,const float* slots,const float* active,float* scratch,bool hybrid,size_t cacheBatchStride){
    int row=blockIdx.x,b=row/(c.tokens*c.heads),t=(row/c.heads)%c.tokens;
    if(active[b*c.tokens+t]==0||(hybrid&&min(c.capacity,int(slots[b*c.tokens+t])+1)<=512))return;
    __shared__ float temporary[8],result;float maximum=-CUDART_INF_F;
    float* values=scratch+size_t(row)*c.capacity;
    for(int k=threadIdx.x;k<c.capacity;k+=256)maximum=fmaxf(maximum,values[k]);
    maximum=blockReduce(maximum,true,temporary,&result);
    // All threads finish reading the max before reusing shared reduction data.
    __syncthreads();float sum=0;
    for(int k=threadIdx.x;k<c.capacity;k+=256){float v=expf(values[k]-maximum);values[k]=v;sum+=v;}
    sum=blockReduce(sum,false,temporary,&result);
    for(int k=threadIdx.x;k<c.capacity;k+=256)values[k]/=sum;
}
// A block produces32 context dimensions. Warps split keys and lanes split D,
// keeping V loads coalesced; shared partial sums avoid atomic accumulation.
__global__ void contextCached(Qwen35AttentionCoreConfig c,const float* value,const float* chunkValue,const float* gate,
    const float* slots,const float* active,const float* scratch,float* output,bool hybrid,size_t cacheBatchStride){
    int row=blockIdx.y,b=row/(c.tokens*c.heads),t=(row/c.heads)%c.tokens;
    int lane=threadIdx.x%32,warp=threadIdx.x/32,d=blockIdx.x*32+lane;
    if(active[b*c.tokens+t]==0){if(!warp)output[size_t(row)*c.dim+d]=0;return;}
    if(hybrid&&min(c.capacity,int(slots[b*c.tokens+t])+1)<=512)return;
    const CachedSlots controls=loadSlots(slots,active,b);
    int head=row%c.heads,kh=head/(c.heads/c.kv);float sum=0;
    int end=min(c.capacity,int(slots[b*c.tokens+t])+1);
    for(int k=warp;k<end;k+=8){
        float weight=scratch[size_t(row)*c.capacity+k];
        const int replacement=controls.find(k);
        const float* rowValue=replacement<0?value+size_t(b)*cacheBatchStride+(size_t(kh)*c.capacity+k)*c.dim:chunkValue+((size_t(b)*c.tokens+replacement)*c.kv+kh)*c.dim;
        sum=fmaf(weight,rowValue[d],sum);
    }
    __shared__ float partial[8][32];partial[warp][lane]=sum;__syncthreads();
    if(!warp){
        sum=0;for(int w=0;w<8;++w)sum+=partial[w][lane];
        output[size_t(row)*c.dim+d]=sum/(1.f+expf(-gate[size_t(row)*c.dim+d]));
    }
}
template<bool hybrid> __global__ void compactCached(Qwen35AttentionCoreConfig c,const float* query,const float* key,
 const float* value,const float* gate,const float* slots,const float* active,const float* chunkKey,const float* chunkValue,float* output,size_t cacheBatchStride){
 constexpr bool tiled=false;
 const int row=tiled?blockIdx.y:blockIdx.x,b=row/(c.tokens*c.heads),t=(row/c.heads)%c.tokens;
 if(active[b*c.tokens+t]==0){if(tiled){if(threadIdx.x<32)output[size_t(row)*c.dim+blockIdx.x*32+threadIdx.x]=0;}else if(threadIdx.x<c.dim)output[size_t(row)*c.dim+threadIdx.x]=0;return;}
 const int end=min(c.capacity,int(slots[b*c.tokens+t])+1);
 if(hybrid&&end>512)return;
 const CachedSlots controls=loadSlots(slots,active,b);
 if(end<1){if(tiled){if(threadIdx.x<32)output[size_t(row)*c.dim+blockIdx.x*32+threadIdx.x]=0;}else if(threadIdx.x<c.dim)output[size_t(row)*c.dim+threadIdx.x]=0;return;}
 const int head=row%c.heads,kh=head/(c.heads/c.kv),lane=threadIdx.x%32,warp=threadIdx.x/32;
 __shared__ float values[8192],temporary[8],result,partial[8][32];
 for(int position=warp;position<end;position+=8){
  float sum=0;const size_t base=size_t(b)*cacheBatchStride+(size_t(kh)*c.capacity+position)*c.dim;
  const int replacement=controls.find(position);
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
   const int replacement=controls.find(k);
   const float* rowValue=replacement<0?value+size_t(b)*cacheBatchStride+(size_t(kh)*c.capacity+k)*c.dim:chunkValue+((size_t(b)*c.tokens+replacement)*c.kv+kh)*c.dim;
   sum=fmaf(values[k],rowValue[d],sum);
  }
  partial[warp][lane]=sum;__syncthreads();
  if(!warp){sum=0;for(int w=0;w<8;++w)sum+=partial[w][lane];output[size_t(row)*c.dim+d]=sum/(1.f+expf(-gate[size_t(row)*c.dim+d]));}
  __syncthreads();
 }
}

__global__ void commitPrefixStrided(Qwen35AttentionCoreConfig c,const float* key,const float* value,const float* slots,const float* active,const float* accepted,float* canonicalKey,float* canonicalValue,size_t cacheBatchStride){
 const int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=c.batch*c.tokens*c.kv*c.dim)return;
 const int d=i%c.dim,head=(i/c.dim)%c.kv,bt=i/(c.kv*c.dim),b=bt/c.tokens,t=bt%c.tokens;
 if(active[bt]==0)return;
 int rank=0;for(int p=0;p<t;++p)rank+=active[b*c.tokens+p]!=0;
 if(rank>=int(accepted[b]))return;
 const int slot=int(slots[bt]);if(slot<0||slot>=c.capacity)return;
 const size_t target=size_t(b)*cacheBatchStride+(size_t(head)*c.capacity+slot)*c.dim+d;
 canonicalKey[target]=key[i];canonicalValue[target]=value[i];
}
}
int RunQwen35AttentionKVOverlayStrided(const Qwen35AttentionCoreConfig& c,const void* const* in,void* output,void* workspace,int strategy,size_t cacheBatchStride,cudaStream_t stream) noexcept {
 if(!validCached(c)||!in||!output||strategy<0||strategy>2||(strategy!=0&&!workspace)||cacheBatchStride<size_t(c.kv)*c.capacity*c.dim||cacheBatchStride>INT_MAX)return 1;
 for(int i=0;i<8;++i)if(!in[i]||in[i]==output)return 1;
 const int rows=c.batch*c.tokens*c.heads;const bool hybrid=strategy==2;
 if(strategy!=0){
  scoresCached<<<dim3((c.capacity+7)/8,rows),256,0,stream>>>(c,(const float*)in[0],(const float*)in[4],(const float*)in[1],(const float*)in[6],(const float*)in[7],(float*)workspace,hybrid,cacheBatchStride);
  normalizeCached<<<rows,256,0,stream>>>(c,(const float*)in[6],(const float*)in[7],(float*)workspace,hybrid,cacheBatchStride);
  contextCached<<<dim3(c.dim/32,rows),256,0,stream>>>(c,(const float*)in[5],(const float*)in[2],(const float*)in[3],(const float*)in[6],(const float*)in[7],(const float*)workspace,(float*)output,hybrid,cacheBatchStride);
 }
 if(strategy!=1){
  if(hybrid)compactCached<true><<<rows,256,0,stream>>>(c,(const float*)in[0],(const float*)in[4],(const float*)in[5],(const float*)in[3],(const float*)in[6],(const float*)in[7],(const float*)in[1],(const float*)in[2],(float*)output,cacheBatchStride);
  else compactCached<false><<<rows,256,0,stream>>>(c,(const float*)in[0],(const float*)in[4],(const float*)in[5],(const float*)in[3],(const float*)in[6],(const float*)in[7],(const float*)in[1],(const float*)in[2],(float*)output,cacheBatchStride);
 }
 return cudaGetLastError()==cudaSuccess?0:1;
}
int CommitQwen35AttentionKVPrefixStrided(const Qwen35AttentionCoreConfig& c,const void* const* in,const float* accepted,void* key,void* value,size_t cacheBatchStride,cudaStream_t stream) noexcept {
 if(!validCached(c)||!in||!accepted||!key||!value||cacheBatchStride<size_t(c.kv)*c.capacity*c.dim||cacheBatchStride>INT_MAX)return 1;
 if(!in[1]||!in[2]||!in[6]||!in[7])return 1;
 const int count=c.batch*c.tokens*c.kv*c.dim;
 commitPrefixStrided<<<(count+255)/256,256,0,stream>>>(c,(const float*)in[1],(const float*)in[2],(const float*)in[6],(const float*)in[7],accepted,(float*)key,(float*)value,cacheBatchStride);
 return cudaGetLastError()==cudaSuccess?0:1;
}
}
