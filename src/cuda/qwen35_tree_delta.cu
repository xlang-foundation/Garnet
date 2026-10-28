// SPDX-License-Identifier: Apache-2.0
#include "qwen35_tree_kernels.cuh"
#include "../backends/tensorrt/qwen35_delta_plugin.h"
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <cmath>
namespace Garnet { namespace {
bool overlap(const void* a,size_t n,const void* b,size_t m){auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);return x>=y?x-y<m:y-x<n;}
__device__ bool good(const Qwen35TreeMetadata& m,Qwen35TreeMetadataConfig c){return m.valid==1&&m.error==0&&m.leaf_count>=1&&m.leaf_count<=c.nodes&&m.prefix>=0&&m.prefix<=c.capacity;}
__global__ void convolutionTree(Qwen35DeltaConfig c,Qwen35TreeMetadataConfig profile,const Qwen35TreeMetadata* metadata,
    const float* input,const float* old,const float* active,const float* weights,float* conv,float* output,float* status){
    int channels=2*c.kh*128+c.vh*128;size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=size_t(profile.batch)*profile.nodes*channels)return;
    if(i<size_t(profile.batch)*profile.nodes*c.vh*128)output[i]=0;
    int ch=i%channels,node=i/channels%profile.nodes,b=i/(size_t(channels)*profile.nodes);const auto& m=metadata[b];
    bool okay=good(m,profile);if(ch==0&&node==0)status[b]=okay?0:float(m.error?m.error:5);
    if(!okay||active[b*profile.nodes+node]==0){conv[i]=0;return;}
    float h[4];for(int k=0;k<c.kernel;++k)h[k]=old[(b*channels+ch)*c.kernel+k];
    for(int j=0;j<m.active_length[node];++j){for(int k=0;k<c.kernel-1;++k)h[k]=h[k+1];
        h[c.kernel-1]=input[(size_t(b)*profile.nodes+m.active_ancestor[node][j])*channels+ch];}
    float value=0;for(int k=0;k<c.kernel;++k)value=__fadd_rn(value,__fmul_rn(h[k],weights[ch*c.kernel+k]));
    conv[i]=value/(1.0f+expf(-value));
}
__global__ void Recurrent(Qwen35DeltaConfig c,Qwen35TreeMetadataConfig profile,const Qwen35TreeMetadata* metadata,const float* conv,const float* z,
    const float* beta,const float* a,const float* old,const float* active,const float* dt,const float* alog,
    const float* norm,float* output){
    using Reduce=cub::BlockReduce<float,128>;
    __shared__ typename Reduce::TempStorage temp;
    __shared__ volatile float q[128],k[128];__shared__ float factor;
    const int lane=threadIdx.x,head=blockIdx.x%c.vh,b=blockIdx.x/c.vh,leaf=blockIdx.y;
    const int kh=head/(c.vh/c.kh),channels=2*c.kh*128+c.vh*128;
    const auto& d=metadata[b];if(!good(d,profile)||leaf>=d.leaf_count)return;
    float state[128];
    #pragma unroll
    for(int j=0;j<128;++j)state[j]=old[((b*c.vh+head)*128+j)*128+lane];
    int path[Qwen35TreeMaxPath],length=0;for(int n=d.leaf[leaf];n>=0;n=d.parent[n])path[length++]=n;
    for(int p=length-1;p>=0;--p){
        int n=path[p],bt=b*profile.nodes+n,offset=bt*channels;bool owns=d.owner[n]==leaf;
        if(active[bt]==0){if(owns)output[(bt*c.vh+head)*128+lane]=0;continue;}
        float qv=conv[offset+kh*128+lane],kv=conv[offset+c.kh*128+kh*128+lane];
        float sum=Reduce(temp).Sum(qv*qv);
        if(lane==0)factor=1.0f/sqrtf(sum+1e-6f);__syncthreads();
        q[lane]=__fmul_rn(__fmul_rn(qv,factor),1.0f/sqrtf(128.0f));__syncthreads();
        sum=Reduce(temp).Sum(kv*kv);
        if(lane==0)factor=1.0f/sqrtf(sum+1e-6f);__syncthreads();
        k[lane]=kv*factor;__syncthreads();
        const float av=a[bt*c.vh+head]+dt[head];
        const float softplus=fmaxf(av,0.0f)+logf(1.0f+expf(-fabsf(av)));
        const float decay=expf(-expf(alog[head])*softplus),gate=1.0f/(1.0f+expf(-beta[bt*c.vh+head]));
        float memory=0;
        #pragma unroll
        for(int j=0;j<128;++j){state[j]=__fmul_rn(state[j],decay);memory=__fadd_rn(memory,__fmul_rn(state[j],k[j]));}
        const float value=conv[offset+2*c.kh*128+head*128+lane],delta=(value-memory)*gate;float core=0;
        #pragma unroll
        for(int j=0;j<128;++j){state[j]=__fadd_rn(state[j],__fmul_rn(k[j],delta));core=__fadd_rn(core,__fmul_rn(state[j],q[j]));}
        // RMS does not feed recurrence. Only the unique owner publishes a node.
        if(owns){
            sum=Reduce(temp).Sum(core*core);
            if(lane==0)factor=1.0f/sqrtf(sum/128.0f+c.epsilon);__syncthreads();
            const float zz=z[(bt*c.vh+head)*128+lane];
            output[(bt*c.vh+head)*128+lane]=__fmul_rn(__fmul_rn(__fmul_rn(core,factor),norm[lane]),zz/(1.0f+expf(-zz)));__syncthreads();
        }
    }
}

}
size_t Qwen35TreeDeltaWorkspace(const Qwen35DeltaConfig& c,Qwen35TreeMetadataConfig p) noexcept{
    if(!ValidQwen35TreeMetadataConfig(p)||c.kd!=128||c.vd!=128||c.kh<1||c.kh>64||c.vh<1||c.vh>64||c.vh%c.kh||c.kernel<1||c.kernel>4||!std::isfinite(c.epsilon)||c.epsilon<=0)return 0;
    return size_t(p.batch)*p.nodes*(2*c.kh*128+c.vh*128)*4;
}
int RunQwen35TreeDelta(const Qwen35DeltaConfig& c,Qwen35TreeMetadataConfig p,const void* const* in,const Qwen35TreeMetadata* metadata,
    float* output,float* status,void* workspace,cudaStream_t stream) noexcept{
    size_t work=Qwen35TreeDeltaWorkspace(c,p);if(!work||!in||!metadata||!output||!status||!workspace)return 1;
    size_t ch=2*c.kh*128+c.vh*128,bt=size_t(p.batch)*p.nodes;
    size_t words[]={bt*ch,bt*c.vh*128,bt*c.vh,bt*c.vh,size_t(p.batch)*ch*c.kernel,size_t(p.batch)*c.vh*128*128,bt,ch*c.kernel,size_t(c.vh),size_t(c.vh),128};
    const void* outputs[]={output,status,workspace};size_t bytes[]={bt*c.vh*128*4,size_t(p.batch)*4,work};
    for(int o=0;o<3;++o){if(reinterpret_cast<uintptr_t>(outputs[o])%4||overlap(outputs[o],bytes[o],metadata,size_t(p.batch)*sizeof(*metadata)))return 1;
        for(int j=0;j<3;++j)if(o!=j&&overlap(outputs[o],bytes[o],outputs[j],bytes[j]))return 1;
        for(int i=0;i<11;++i)if(!in[i]||reinterpret_cast<uintptr_t>(in[i])%4||overlap(outputs[o],bytes[o],in[i],words[i]*4))return 1;}
    if(reinterpret_cast<uintptr_t>(metadata)%4)return 1;
    convolutionTree<<<(bt*ch+127)/128,128,0,stream>>>(c,p,metadata,(const float*)in[0],(const float*)in[4],(const float*)in[6],(const float*)in[7],(float*)workspace,output,status);
    if(cudaPeekAtLastError()!=cudaSuccess)return 1;
    Recurrent<<<dim3(p.batch*c.vh,p.nodes),128,0,stream>>>(c,p,metadata,(float*)workspace,(const float*)in[1],(const float*)in[2],(const float*)in[3],(const float*)in[5],(const float*)in[6],(const float*)in[8],(const float*)in[9],(const float*)in[10],output);
    return cudaPeekAtLastError()==cudaSuccess?0:1;
}
}
