// SPDX-License-Identifier: Apache-2.0
// PREPARED ONLY: not installed, compiled, tested or admitted by any serving path.
#include "qwen35_tree_commit.cuh"
#include <cub/block/block_reduce.cuh>
#include <cmath>
#include <initializer_list>
namespace Garnet { namespace {
bool overlap(const void* a,size_t n,const void* b,size_t m){auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);return x>=y?x-y<m:y-x<n;}
__host__ __device__ int channels(Qwen35TreeCommitConfig c){return (2*c.key_heads+c.value_heads)*128;}
__host__ __device__ int layers(Qwen35TreeCommitConfig c){return c.linear_layers+c.full_layers;}
__host__ __device__ int64_t linearState(Qwen35TreeCommitConfig c){return int64_t(channels(c))*c.kernel+int64_t(c.value_heads)*128*128;}
__host__ __device__ int64_t fullState(Qwen35TreeCommitConfig c){return int64_t(2)*c.kv_heads*c.capacity*c.head_dim;}
__host__ __device__ int64_t linearPayload(Qwen35TreeCommitConfig c){return int64_t(c.nodes)*(channels(c)+2*c.value_heads);}
__host__ __device__ int64_t fullPayload(Qwen35TreeCommitConfig c){return int64_t(2)*c.nodes*c.kv_heads*c.head_dim;}
__device__ Qwen35TreeCommitLayer item(int i,Qwen35TreeCommitConfig c,const Qwen35TreeCommitLayer* linear,const Qwen35TreeCommitLayer* full){return i<c.linear_layers?linear[i]:full[i-c.linear_layers];}
__device__ bool validRange(int64_t start,int64_t width,int64_t low,int64_t high){return start>=low&&start<=high&&width>=0&&width<=high-start;}
__global__ void CheckLayouts(Qwen35TreeCommitConfig c,const Qwen35TreeCommitLayer* linear,const Qwen35TreeCommitLayer* full,int* status){
    int i=threadIdx.x;if(i>=layers(c))return;auto a=item(i,c,linear,full);
    int64_t sw=i<c.linear_layers?linearState(c):fullState(c),pw=i<c.linear_layers?linearPayload(c):fullPayload(c);
    if(a.index<0||a.index>=layers(c)||!validRange(a.state,sw,0,c.state_width)||!validRange(a.payload,pw,c.hidden_width,c.status_offset)){atomicExch(status,1);return;}
    for(int j=0;j<layers(c);++j)if(i!=j){auto b=item(j,c,linear,full);int64_t bs=j<c.linear_layers?linearState(c):fullState(c),bp=j<c.linear_layers?linearPayload(c):fullPayload(c);
        if(b.index<0||b.index>=layers(c)||!validRange(b.state,bs,0,c.state_width)||!validRange(b.payload,bp,c.hidden_width,c.status_offset)){atomicExch(status,1);continue;}
        if(a.index==b.index||(a.state<b.state+bs&&b.state<a.state+sw)||(a.payload<b.payload+bp&&b.payload<a.payload+pw))atomicExch(status,1);}
}
__global__ void CheckPacket(Qwen35TreeCommitConfig c,const Qwen35TreeCommitSelection* selection,const Qwen35TreeMetadata* metadata,const float* packed,int* status){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(c.batch)*c.packed_width)return;
    int b=i/c.packed_width;int64_t word=i%c.packed_width;float value=packed[i];const auto& m=metadata[b];
    if(!isfinite(value)){atomicExch(status,1);return;}
    if(word==0&&(m.valid!=1||m.error!=0||selection->prefix[b]<0||selection->prefix[b]>c.capacity||selection->last[b]<-1||selection->last[b]>=c.nodes||m.prefix!=selection->prefix[b]))atomicExch(status,1);
    if(word>=c.status_offset&&word<c.metadata_offset&&value!=0)atomicExch(status,1);
    if(word>=c.metadata_offset){int k=int(word-c.metadata_offset);const auto* original=reinterpret_cast<const int32_t*>(&m);if(value!=float(original[k]))atomicExch(status,1);}
}
__global__ void CommitRecurrent(Qwen35TreeCommitConfig c,const Qwen35TreeCommitSelection* selection,const Qwen35TreeMetadata* metadata,
    const float* packed,float* canonical,const Qwen35TreeCommitLayer* layouts,const float* parameters,const int* status){
    if(*status)return;
    using Reduce=cub::BlockReduce<float,128>;__shared__ typename Reduce::TempStorage temp;
    __shared__ volatile float key[128];__shared__ float factor;
    int lane=threadIdx.x,head=blockIdx.x%c.value_heads,b=blockIdx.x/c.value_heads,l=blockIdx.y,last=selection->last[b];
    if(last<0)return;const auto& m=metadata[b];int count=m.active_length[last];if(!count)return;
    auto layout=layouts[l];int ch=channels(c),kh=head/(c.value_heads/c.key_heads);
    const float* packet=packed+size_t(b)*c.packed_width+layout.payload;
    float* old=canonical+size_t(b)*c.state_width+layout.state;
    int parameterStride=ch*c.kernel+2*c.value_heads;const float* p=parameters+size_t(l)*parameterStride;
    int kc=c.key_heads*128+kh*128+lane,vc=2*c.key_heads*128+head*128+lane;
    float hk[4],hv[4];for(int j=0;j<c.kernel;++j){hk[j]=old[kc*c.kernel+j];hv[j]=old[vc*c.kernel+j];}
    float state[128];
    #pragma unroll
    for(int j=0;j<128;++j)state[j]=old[ch*c.kernel+(head*128+j)*128+lane];
    for(int step=0;step<count;++step){int node=m.active_ancestor[last][step];
        for(int j=0;j<c.kernel-1;++j){hk[j]=hk[j+1];hv[j]=hv[j+1];}
        hk[c.kernel-1]=packet[node*ch+kc];hv[c.kernel-1]=packet[node*ch+vc];float kval=0,vval=0;
        for(int j=0;j<c.kernel;++j){kval=__fadd_rn(kval,__fmul_rn(hk[j],p[kc*c.kernel+j]));vval=__fadd_rn(vval,__fmul_rn(hv[j],p[vc*c.kernel+j]));}
        kval=kval/(1.0f+expf(-kval));vval=vval/(1.0f+expf(-vval));
        float sum=Reduce(temp).Sum(kval*kval);if(lane==0)factor=1.0f/sqrtf(sum+1e-6f);__syncthreads();
        key[lane]=kval*factor;__syncthreads();
        float av=packet[c.nodes*(ch+c.value_heads)+node*c.value_heads+head]+p[ch*c.kernel+head];
        float softplus=fmaxf(av,0.0f)+logf(1.0f+expf(-fabsf(av)));
        float decay=expf(-expf(p[ch*c.kernel+c.value_heads+head])*softplus);
        float gate=1.0f/(1.0f+expf(-packet[c.nodes*ch+node*c.value_heads+head]));float memory=0;
        #pragma unroll
        for(int j=0;j<128;++j){state[j]=__fmul_rn(state[j],decay);memory=__fadd_rn(memory,__fmul_rn(state[j],key[j]));}
        float delta=(vval-memory)*gate;
        #pragma unroll
        for(int j=0;j<128;++j)state[j]=__fadd_rn(state[j],__fmul_rn(key[j],delta));
        __syncthreads();
    }
    #pragma unroll
    for(int j=0;j<128;++j)old[ch*c.kernel+(head*128+j)*128+lane]=state[j];
}
// Separate launch AFTER CommitRecurrent: no block can read already-updated old
// convolution history while replaying recurrence.
__global__ void CommitConvolution(Qwen35TreeCommitConfig c,const Qwen35TreeCommitSelection* selection,const Qwen35TreeMetadata* metadata,
    const float* packed,float* canonical,const Qwen35TreeCommitLayer* layouts,const int* status){
    if(*status)return;size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;int ch=channels(c);if(i>=size_t(c.batch)*c.linear_layers*ch)return;
    int channel=i%ch,l=i/ch%c.linear_layers,b=i/(size_t(ch)*c.linear_layers),last=selection->last[b];if(last<0)return;
    const auto& m=metadata[b];int count=m.active_length[last];if(!count)return;auto layout=layouts[l];
    const float* packet=packed+size_t(b)*c.packed_width+layout.payload;float* old=canonical+size_t(b)*c.state_width+layout.state+channel*c.kernel;
    float h[4];for(int j=0;j<c.kernel;++j)h[j]=old[j];
    for(int step=0;step<count;++step){for(int j=0;j<c.kernel-1;++j)h[j]=h[j+1];h[c.kernel-1]=packet[m.active_ancestor[last][step]*ch+channel];}
    for(int j=0;j<c.kernel;++j)old[j]=h[j];
}
__global__ void CommitKV(Qwen35TreeCommitConfig c,const Qwen35TreeCommitSelection* selection,const Qwen35TreeMetadata* metadata,
    const float* packed,float* canonical,const Qwen35TreeCommitLayer* layouts,const int* status){
    if(*status)return;size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    size_t per=size_t(8)*c.kv_heads*c.head_dim;if(i>=size_t(c.batch)*c.full_layers*per)return;
    int dim=i%c.head_dim,head=i/c.head_dim%c.kv_heads,step=i/(c.head_dim*c.kv_heads)%8,l=i/per%c.full_layers,b=i/(per*c.full_layers),last=selection->last[b];
    if(last<0)return;const auto& m=metadata[b];if(step>=m.active_length[last])return;auto layout=layouts[l];int node=m.active_ancestor[last][step];
    const float* packet=packed+size_t(b)*c.packed_width+layout.payload;
    float* state=canonical+size_t(b)*c.state_width+layout.state;
    size_t source=(size_t(node)*c.kv_heads+head)*c.head_dim+dim,target=(size_t(head)*c.capacity+m.prefix+step)*c.head_dim+dim;
    state[target]=packet[source];state[size_t(c.kv_heads)*c.capacity*c.head_dim+target]=packet[size_t(c.nodes)*c.kv_heads*c.head_dim+source];
}
}
bool ValidQwen35TreeCommitConfig(const Qwen35TreeCommitConfig& c) noexcept{
    if(!ValidQwen35TreeMetadataConfig({c.batch,c.nodes,c.capacity})||c.key_heads<1||c.key_heads>64||c.value_heads<1||c.value_heads>64||c.value_heads%c.key_heads||c.kernel<1||c.kernel>4||c.kv_heads<1||c.kv_heads>32||c.head_dim<32||c.head_dim>256||c.head_dim%32||c.linear_layers<1||c.linear_layers>128||c.full_layers<0||c.full_layers>128||layers(c)>128)return false;
    return c.hidden_width>=c.nodes&&c.hidden_width<=int64_t(c.nodes)*16384&&c.hidden_width%c.nodes==0&&c.state_width>0&&c.state_width<=INT32_MAX&&c.packed_width>0&&c.packed_width<=INT32_MAX&&
        c.state_width==c.linear_layers*linearState(c)+c.full_layers*fullState(c)&&c.status_offset==c.hidden_width+c.linear_layers*linearPayload(c)+c.full_layers*fullPayload(c)&&
        c.metadata_offset==c.status_offset+layers(c)&&c.packed_width==c.metadata_offset+836;
}
size_t Qwen35TreeCommitWorkspace(const Qwen35TreeCommitConfig& c) noexcept{return ValidQwen35TreeCommitConfig(c)?size_t(c.batch)*sizeof(Qwen35TreeMetadata):0;}
int RunQwen35TreeCommit(const Qwen35TreeCommitConfig& c,const Qwen35TreeCommitSelection* s,const float* packed,float* canonical,
    const float* parents,const float* active,const float* prefix,const Qwen35TreeCommitLayer* linear,const Qwen35TreeCommitLayer* full,
    const float* parameters,void* workspace,int* status,cudaStream_t stream) noexcept{
    size_t work=Qwen35TreeCommitWorkspace(c);if(!work)return 1;
    const void* ptr[]={packed,canonical,parents,active,prefix,linear,full,parameters,s,workspace,status};
    size_t bytes[]={size_t(c.batch)*c.packed_width*4,size_t(c.batch)*c.state_width*4,size_t(c.batch)*c.nodes*4,size_t(c.batch)*c.nodes*4,size_t(c.batch)*4,
        size_t(c.linear_layers)*sizeof(*linear),size_t(c.full_layers)*sizeof(*full),size_t(c.linear_layers)*(channels(c)*c.kernel+2*c.value_heads)*4,sizeof(*s),work,4};
    for(int i=0;i<11;++i)if(bytes[i]&&(!ptr[i]||reinterpret_cast<uintptr_t>(ptr[i])%((i==5||i==6)?8:4)))return 1;
    for(int o:{1,9,10})for(int i=0;i<11;++i)if(i!=o&&bytes[i]&&overlap(ptr[o],bytes[o],ptr[i],bytes[i]))return 1;
    auto* metadata=static_cast<Qwen35TreeMetadata*>(workspace);
    if(cudaMemsetAsync(status,0,4,stream)!=cudaSuccess)return 1;
    if(RunQwen35TreeMetadata({c.batch,c.nodes,c.capacity},parents,active,prefix,metadata,stream))return 1;
    CheckLayouts<<<1,128,0,stream>>>(c,linear,full,status);if(cudaPeekAtLastError()!=cudaSuccess)return 1;
    CheckPacket<<<(size_t(c.batch)*c.packed_width+255)/256,256,0,stream>>>(c,s,metadata,packed,status);if(cudaPeekAtLastError()!=cudaSuccess)return 1;
    CommitRecurrent<<<dim3(c.batch*c.value_heads,c.linear_layers),128,0,stream>>>(c,s,metadata,packed,canonical,linear,parameters,status);if(cudaPeekAtLastError()!=cudaSuccess)return 1;
    size_t count=size_t(c.batch)*c.linear_layers*channels(c);
    CommitConvolution<<<(count+127)/128,128,0,stream>>>(c,s,metadata,packed,canonical,linear,status);if(cudaPeekAtLastError()!=cudaSuccess)return 1;
    if(c.full_layers){count=size_t(c.batch)*c.full_layers*8*c.kv_heads*c.head_dim;CommitKV<<<(count+255)/256,256,0,stream>>>(c,s,metadata,packed,canonical,full,status);}
    return cudaPeekAtLastError()==cudaSuccess?0:1;
}
}
