// SPDX-License-Identifier: Apache-2.0
#include "qwen35_tree_metadata.cuh"
#include <cuda_runtime.h>
#include <cmath>
namespace Garnet { namespace {
__global__ void ClearTreeMetadata(size_t words,int32_t* output){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<words)output[i]=0;
}
__global__ void BuildTreeMetadata(Qwen35TreeMetadataConfig c,const float* parents,const float* active,
    const float* prefix,Qwen35TreeMetadata* output){
    int b=blockIdx.x*blockDim.x+threadIdx.x;if(b>=c.batch)return;
    auto& m=output[b];
    for(int n=0;n<c.nodes;++n)if(active[b*c.nodes+n]!=0&&active[b*c.nodes+n]!=1){m.error=2;return;}
    float start=prefix[b];if(!isfinite(start)||start<0||start>c.capacity||start!=floorf(start)){m.error=3;return;}
    m.prefix=int(start);bool child[Qwen35TreeMaxNodes]{};
    for(int n=0;n<c.nodes;++n){
        float value=parents[b*c.nodes+n];
        if(!isfinite(value)||value!=floorf(value)||(n==0?value!=-1:value<0||value>=n)){m.error=1;return;}
        int p=int(value);m.parent[n]=p;m.depth[n]=n?m.depth[p]+1:0;
        if(m.depth[n]>=Qwen35TreeMaxPath){m.error=4;return;}
        if(n)child[p]=true;m.owner[n]=-1;
    }
    int maximum=0;
    for(int n=0;n<c.nodes;++n){
        int path[Qwen35TreeMaxPath],length=0;for(int p=n;p>=0;p=m.parent[p])path[length++]=p;
        int count=0;for(int j=length-1;j>=0;--j)if(active[b*c.nodes+path[j]]==1)m.active_ancestor[n][count++]=path[j];
        m.active_length[n]=count;if(count>maximum)maximum=count;
        if(!child[n]){int id=m.leaf_count++;m.leaf[id]=n;for(int p=n;p>=0;p=m.parent[p])if(m.owner[p]<0)m.owner[p]=id;}
    }
    if(m.prefix>c.capacity-maximum){m.error=3;return;}
    // Stream dependencies publish the full descriptor before any consumer runs.
    m.valid=1;
}
bool overlaps(const void* a,size_t n,const void* b,size_t m){
    auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);return x>=y?x-y<m:y-x<n;
}
}
bool ValidQwen35TreeMetadataConfig(Qwen35TreeMetadataConfig c) noexcept{
    return (c.batch==1||c.batch==2||c.batch==4||c.batch==8)&&c.nodes>=1&&c.nodes<=Qwen35TreeMaxNodes&&c.capacity>=32&&c.capacity<=8192;
}
int RunQwen35TreeMetadata(Qwen35TreeMetadataConfig c,const float* parents,const float* active,
    const float* prefix,Qwen35TreeMetadata* output,cudaStream_t stream) noexcept{
    if(!ValidQwen35TreeMetadataConfig(c)||!parents||!active||!prefix||!output)return 1;
    size_t out=size_t(c.batch)*sizeof(Qwen35TreeMetadata),nodes=size_t(c.batch)*c.nodes*4,rows=size_t(c.batch)*4;
    if(reinterpret_cast<uintptr_t>(output)%alignof(Qwen35TreeMetadata))return 1;
    const float* input[]={parents,active,prefix};size_t bytes[]={nodes,nodes,rows};
    for(int i=0;i<3;++i)if(reinterpret_cast<uintptr_t>(input[i])%4||overlaps(output,out,input[i],bytes[i]))return 1;
    size_t words=out/4;ClearTreeMetadata<<<(words+127)/128,128,0,stream>>>(words,reinterpret_cast<int32_t*>(output));
    if(cudaPeekAtLastError()!=cudaSuccess)return 1;
    BuildTreeMetadata<<<1,32,0,stream>>>(c,parents,active,prefix,output);
    return cudaPeekAtLastError()==cudaSuccess?0:1;
}
}
