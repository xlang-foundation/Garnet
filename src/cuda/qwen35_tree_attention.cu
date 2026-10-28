// SPDX-License-Identifier: Apache-2.0
#include "qwen35_tree_kernels.cuh"
#include <cuda_runtime.h>
#include <math_constants.h>
namespace Garnet { namespace {
bool overlap(const void* a,size_t n,const void* b,size_t m){auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);return x>=y?x-y<m:y-x<n;}
__device__ bool good(const Qwen35TreeMetadata& m,Qwen35AttentionCoreConfig c){return m.valid==1&&m.error==0&&m.leaf_count>=1&&m.leaf_count<=c.tokens&&m.prefix>=0&&m.prefix<=c.capacity;}
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

__global__ void scoresTree(Qwen35AttentionCoreConfig c,const Qwen35TreeMetadata* metadata,const float* query,const float* key,
    const float* oldKey,const float* active,float* scratch,float* status){
    int row=blockIdx.y,b=row/(c.tokens*c.heads),node=(row/c.heads)%c.tokens;const auto& m=metadata[b];bool okay=good(m,c);
    if(blockIdx.x==0&&threadIdx.x==0&&node==0&&row%c.heads==0)status[b]=okay?0:float(m.error?m.error:5);
    if(!okay||active[b*c.tokens+node]==0)return;
    int position=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x%32,head=row%c.heads,kh=head/(c.heads/c.kv);
    if(position>=c.capacity)return;int slot=m.prefix+m.active_length[node]-1;float sum=0;
    if(position<=slot){const float* input=position<m.prefix?oldKey+((size_t(b)*c.kv+kh)*c.capacity+position)*c.dim:
        key+((size_t(b)*c.tokens+m.active_ancestor[node][position-m.prefix])*c.kv+kh)*c.dim;
        for(int d=lane;d<c.dim;d+=32)sum=fmaf(query[size_t(row)*c.dim+d],input[d],sum);sum=warpSum(sum);}
    if(!lane)scratch[size_t(row)*c.capacity+position]=position<=slot?sum*rsqrtf(float(c.dim)):-1e30f;
}
__global__ void normalizeTree(Qwen35AttentionCoreConfig c,const Qwen35TreeMetadata* metadata,const float* active,float* scratch){
    int row=blockIdx.x,b=row/(c.tokens*c.heads),t=(row/c.heads)%c.tokens;
    if(!good(metadata[b],c)||active[b*c.tokens+t]==0)return;
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

__global__ void contextTree(Qwen35AttentionCoreConfig c,const Qwen35TreeMetadata* metadata,const float* value,const float* oldValue,
    const float* gate,const float* active,const float* scratch,float* output){
    int row=blockIdx.y,b=row/(c.tokens*c.heads),node=(row/c.heads)%c.tokens,lane=threadIdx.x%32,warp=threadIdx.x/32,dim=blockIdx.x*32+lane;
    const auto& m=metadata[b];if(!good(m,c)||active[b*c.tokens+node]==0){if(!warp)output[size_t(row)*c.dim+dim]=0;return;}
    int end=m.prefix+m.active_length[node],head=row%c.heads,kh=head/(c.heads/c.kv);float sum=0;
    for(int position=warp;position<end;position+=8){const float* input=position<m.prefix?oldValue+((size_t(b)*c.kv+kh)*c.capacity+position)*c.dim:
        value+((size_t(b)*c.tokens+m.active_ancestor[node][position-m.prefix])*c.kv+kh)*c.dim;
        sum=fmaf(scratch[size_t(row)*c.capacity+position],input[dim],sum);}
    __shared__ float partial[8][32];partial[warp][lane]=sum;__syncthreads();
    if(!warp){sum=0;for(int w=0;w<8;++w)sum+=partial[w][lane];output[size_t(row)*c.dim+dim]=sum/(1.f+expf(-gate[size_t(row)*c.dim+dim]));}
}
}
size_t Qwen35TreeAttentionWorkspace(const Qwen35AttentionCoreConfig& c) noexcept{
    if(!ValidQwen35TreeMetadataConfig({c.batch,c.tokens,c.capacity})||c.heads<1||c.heads>32||c.kv<1||c.kv>32||c.heads%c.kv||c.dim<32||c.dim>256||c.dim%32)return 0;
    return size_t(c.batch)*c.tokens*c.heads*c.capacity*4;
}
int RunQwen35TreeAttention(const Qwen35AttentionCoreConfig& c,const void* const* in,const Qwen35TreeMetadata* metadata,
    float* output,float* status,void* workspace,cudaStream_t stream) noexcept{
    size_t work=Qwen35TreeAttentionWorkspace(c);if(!work||!in||!metadata||!output||!status||!workspace)return 1;
    size_t bt=size_t(c.batch)*c.tokens,full=size_t(c.batch)*c.kv*c.capacity*c.dim;
    size_t words[]={bt*c.heads*c.dim,bt*c.kv*c.dim,bt*c.kv*c.dim,bt*c.heads*c.dim,full,full,bt};
    const void* outputs[]={output,status,workspace};size_t bytes[]={words[0]*4,size_t(c.batch)*4,work};
    for(int o=0;o<3;++o){if(reinterpret_cast<uintptr_t>(outputs[o])%4||overlap(outputs[o],bytes[o],metadata,size_t(c.batch)*sizeof(*metadata)))return 1;
        for(int j=0;j<3;++j)if(o!=j&&overlap(outputs[o],bytes[o],outputs[j],bytes[j]))return 1;
        for(int i=0;i<7;++i)if(!in[i]||reinterpret_cast<uintptr_t>(in[i])%4||overlap(outputs[o],bytes[o],in[i],words[i]*4))return 1;}
    if(reinterpret_cast<uintptr_t>(metadata)%4)return 1;
    int rows=c.batch*c.tokens*c.heads;
    scoresTree<<<dim3((c.capacity+7)/8,rows),256,0,stream>>>(c,metadata,(const float*)in[0],(const float*)in[1],(const float*)in[4],(const float*)in[6],(float*)workspace,status);
    if(cudaPeekAtLastError()!=cudaSuccess)return 1;
    normalizeTree<<<rows,256,0,stream>>>(c,metadata,(const float*)in[6],(float*)workspace);
    if(cudaPeekAtLastError()!=cudaSuccess)return 1;
    contextTree<<<dim3(c.dim/32,rows),256,0,stream>>>(c,metadata,(const float*)in[2],(const float*)in[5],(const float*)in[3],(const float*)in[6],(float*)workspace,output);
    return cudaPeekAtLastError()==cudaSuccess?0:1;
}
}
