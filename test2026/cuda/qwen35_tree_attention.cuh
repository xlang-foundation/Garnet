// SPDX-License-Identifier: Apache-2.0
// UNWIRED. Include after production delta/attention and qwen35_tree_state.cuh.
// Same attention arithmetic; K/V comes from a readonly prefix plus ancestors.
namespace Qwen35TreeAttention {
using Descriptor=Qwen35Tree::Descriptor;
struct Prefix {int start[8]{};};
struct Selection {Prefix prefix;int length[8]{},node[8][Qwen35Tree::MaxPath]{};};
inline bool ValidConfig(const Garnet::Qwen35AttentionCoreConfig& c,const Descriptor& d){
    return Qwen35Tree::Valid(d)&&(c.batch==1||c.batch==2||c.batch==4||c.batch==8)&&c.tokens==d.count&&
        c.heads>=1&&c.heads<=32&&c.kv>=1&&c.kv<=32&&c.heads%c.kv==0&&c.dim>=32&&c.dim<=256&&c.dim%32==0&&c.capacity>=32&&c.capacity<=8192;
}
inline bool ValidPrefix(const Garnet::Qwen35AttentionCoreConfig& c,const Descriptor& d,Prefix p){
    int length=0;for(int n=0;n<d.count;++n)length=std::max(length,d.depth[n]+1);
    for(int b=0;b<c.batch;++b)if(p.start[b]<0||p.start[b]>c.capacity-length)return false;
    return true;
}
inline bool BuildSelection(const Garnet::Qwen35AttentionCoreConfig& c,const Descriptor& d,Prefix prefix,const int* last,Selection& result){
    if(!ValidConfig(c,d)||!last)return false;
    Selection s;s.prefix=prefix;
    for(int b=0;b<c.batch;++b){
        if(last[b]<-1||last[b]>=d.count)return false;
        int path[Qwen35Tree::MaxPath],count=0;
        for(int n=last[b];n>=0;n=d.parent[n])path[count++]=n;
        if(prefix.start[b]<0||prefix.start[b]>c.capacity-count)return false;
        s.length[b]=count;for(int j=0;j<count;++j)s.node[b][j]=path[count-1-j];
    }
    result=s;return true;
}
inline bool ValidSelection(const Garnet::Qwen35AttentionCoreConfig& c,const Descriptor& d,const Selection& s){
    if(!ValidConfig(c,d))return false;
    for(int b=0;b<c.batch;++b){
        int length=s.length[b];if(length<0||length>Qwen35Tree::MaxPath||s.prefix.start[b]<0||s.prefix.start[b]>c.capacity-length)return false;
        for(int j=0;j<length;++j){int n=s.node[b][j];if(n<0||n>=d.count||d.parent[n]!=(j?s.node[b][j-1]:-1))return false;}
    }
    return true;
}
__device__ int Path(const Descriptor& d,int node,int* path){
    int length=0;for(int n=node;n>=0;n=d.parent[n])path[length++]=n;return length;
}
__device__ int ActivePath(const Descriptor& d,const float* active,int b,int node,int* path){
    int reverse[Qwen35Tree::MaxPath],length=Path(d,node,reverse),count=0;
    for(int j=length-1;j>=0;--j)if(active[b*d.count+reverse[j]]!=0)path[count++]=reverse[j];return count;
}
__global__ void Scores(Garnet::Qwen35AttentionCoreConfig c,Descriptor d,Prefix prefix,const float* query,
    const float* key,const float* oldKey,const float* active,float* scratch){
    int row=blockIdx.y,b=row/(c.tokens*c.heads),node=(row/c.heads)%c.tokens;
    if(active[b*c.tokens+node]==0)return;
    int path[Qwen35Tree::MaxPath],length=ActivePath(d,active,b,node,path);
    int position=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x%32,head=row%c.heads,kh=head/(c.heads/c.kv);
    if(position>=c.capacity)return;
    int slot=prefix.start[b]+length-1;float sum=0;
    if(position<=slot){
        const float* source=position<prefix.start[b]?oldKey+((size_t(b)*c.kv+kh)*c.capacity+position)*c.dim:
            key+((size_t(b)*c.tokens+path[position-prefix.start[b]])*c.kv+kh)*c.dim;
        for(int dim=lane;dim<c.dim;dim+=32)sum=fmaf(query[size_t(row)*c.dim+dim],source[dim],sum);
        sum=Garnet::warpSum(sum);
    }
    if(!lane)scratch[size_t(row)*c.capacity+position]=position<=slot?sum*rsqrtf(float(c.dim)):-1e30f;
}
__global__ void Context(Garnet::Qwen35AttentionCoreConfig c,Descriptor d,Prefix prefix,const float* value,
    const float* oldValue,const float* gate,const float* active,const float* scratch,float* output){
    int row=blockIdx.y,b=row/(c.tokens*c.heads),node=(row/c.heads)%c.tokens;
    int lane=threadIdx.x%32,warp=threadIdx.x/32,dim=blockIdx.x*32+lane;
    if(active[b*c.tokens+node]==0){if(!warp)output[size_t(row)*c.dim+dim]=0;return;}
    int path[Qwen35Tree::MaxPath],length=ActivePath(d,active,b,node,path),end=prefix.start[b]+length;
    int head=row%c.heads,kh=head/(c.heads/c.kv);float sum=0;
    for(int position=warp;position<end;position+=8){
        const float* source=position<prefix.start[b]?oldValue+((size_t(b)*c.kv+kh)*c.capacity+position)*c.dim:
            value+((size_t(b)*c.tokens+path[position-prefix.start[b]])*c.kv+kh)*c.dim;
        sum=fmaf(scratch[size_t(row)*c.capacity+position],source[dim],sum);
    }
    __shared__ float partial[8][32];partial[warp][lane]=sum;__syncthreads();
    if(!warp){sum=0;for(int w=0;w<8;++w)sum+=partial[w][lane];output[size_t(row)*c.dim+dim]=sum/(1.f+expf(-gate[size_t(row)*c.dim+dim]));}
}
inline int Run(const Garnet::Qwen35AttentionCoreConfig& c,const Descriptor& d,Prefix prefix,
    const void* const* in,float* output,float* workspace,cudaStream_t stream){
    // Inputs Q,K,V,gate,oldK,oldV,active. Prefix/descriptor validated on host.
    if(!ValidConfig(c,d)||!ValidPrefix(c,d,prefix)||!in||!output||!workspace)return 1;
    size_t bt=size_t(c.batch)*c.tokens,full=size_t(c.batch)*c.kv*c.capacity*c.dim;
    size_t words[]={bt*c.heads*c.dim,bt*c.kv*c.dim,bt*c.kv*c.dim,bt*c.heads*c.dim,full,full,bt};
    size_t out=words[0]*4,work=bt*c.heads*c.capacity*4;
    if(reinterpret_cast<uintptr_t>(output)%4||reinterpret_cast<uintptr_t>(workspace)%4||Qwen35Tree::Overlap(output,out,workspace,work))return 1;
    for(int i=0;i<7;++i)if(!in[i]||reinterpret_cast<uintptr_t>(in[i])%4||Qwen35Tree::Overlap(output,out,in[i],words[i]*4)||Qwen35Tree::Overlap(workspace,work,in[i],words[i]*4))return 1;
    int rows=c.batch*c.tokens*c.heads;
    Scores<<<dim3((c.capacity+7)/8,rows),256,0,stream>>>(c,d,prefix,(const float*)in[0],(const float*)in[1],(const float*)in[4],(const float*)in[6],workspace);
    if(cudaPeekAtLastError()!=cudaSuccess)return 1;
    Garnet::normalize<<<rows,256,0,stream>>>(c,(const float*)in[6],workspace);
    if(cudaPeekAtLastError()!=cudaSuccess)return 1;
    Context<<<dim3(c.dim/32,rows),256,0,stream>>>(c,d,prefix,(const float*)in[2],(const float*)in[5],(const float*)in[3],(const float*)in[6],workspace,output);
    return cudaPeekAtLastError()==cudaSuccess?0:1;
}
__global__ void CheckActive(int count,const float* active,int* status){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count&&active[i]!=0&&active[i]!=1)atomicExch(status,1);
}
__global__ void Commit(Garnet::Qwen35AttentionCoreConfig c,Selection s,const float* key,const float* value,
    const float* active,float* canonicalKey,float* canonicalValue,const int* status){
    if(*status)return;
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x,total=size_t(c.batch)*Qwen35Tree::MaxPath*c.kv*c.dim;
    if(i>=total)return;
    int dim=i%c.dim,head=i/c.dim%c.kv,step=i/(c.dim*c.kv)%Qwen35Tree::MaxPath,b=i/(size_t(c.dim)*c.kv*Qwen35Tree::MaxPath);
    if(step>=s.length[b])return;int node=s.node[b][step];if(active[b*c.tokens+node]!=1)return;
    int slot=s.prefix.start[b];for(int j=0;j<step;++j)slot+=active[b*c.tokens+s.node[b][j]]==1;
    size_t target=((size_t(b)*c.kv+head)*c.capacity+slot)*c.dim+dim;
    size_t source=((size_t(b)*c.tokens+node)*c.kv+head)*c.dim+dim;
    canonicalKey[target]=key[source];canonicalValue[target]=value[source];
}
inline int CommitSelected(const Garnet::Qwen35AttentionCoreConfig& c,const Descriptor& d,const Selection& s,
    const float* key,const float* value,const float* active,float* canonicalKey,float* canonicalValue,int* status,cudaStream_t stream){
    if(!ValidSelection(c,d,s)||!key||!value||!active||!canonicalKey||!canonicalValue||!status)return 1;
    const void* ptr[]={key,value,active,canonicalKey,canonicalValue,status};
    size_t words[]={size_t(c.batch)*c.tokens*c.kv*c.dim,size_t(c.batch)*c.tokens*c.kv*c.dim,size_t(c.batch)*c.tokens,
        size_t(c.batch)*c.kv*c.capacity*c.dim,size_t(c.batch)*c.kv*c.capacity*c.dim,1};
    for(int i=0;i<6;++i)if(reinterpret_cast<uintptr_t>(ptr[i])%4)return 1;
    for(int o=3;o<6;++o)for(int i=0;i<6;++i)if(i!=o&&Qwen35Tree::Overlap(ptr[o],words[o]*4,ptr[i],words[i]*4))return 1;
    if(cudaMemsetAsync(status,0,4,stream)!=cudaSuccess)return 1;
    CheckActive<<<(c.batch*c.tokens+127)/128,128,0,stream>>>(c.batch*c.tokens,active,status);
    if(cudaPeekAtLastError()!=cudaSuccess)return 1;
    size_t total=size_t(c.batch)*Qwen35Tree::MaxPath*c.kv*c.dim;
    Commit<<<(total+255)/256,256,0,stream>>>(c,s,key,value,active,canonicalKey,canonicalValue,status);
    return cudaPeekAtLastError()==cudaSuccess?0:1;
}
}
