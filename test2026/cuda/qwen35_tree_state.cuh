// SPDX-License-Identifier: Apache-2.0
// UNWIRED experiment. Include after the unchanged production qwen35_delta.cu.
// Each leaf owns a private register state. No recurrent state per tree node.
namespace Qwen35Tree {
constexpr int MaxNodes=64,MaxPath=8;
struct Descriptor {int count=0,leaf_count=0,parent[MaxNodes]{},depth[MaxNodes]{},leaf[MaxNodes]{},owner[MaxNodes]{};};
inline bool Build(const int* parent,int count,Descriptor& result){
    if(!parent||count<1||count>MaxNodes||parent[0]!=-1)return false;
    Descriptor d;d.count=count;bool children[MaxNodes]{};
    for(int n=0;n<count;++n){
        if(n&&(parent[n]<0||parent[n]>=n))return false;
        d.parent[n]=parent[n];d.depth[n]=n?d.depth[parent[n]]+1:0;
        if(d.depth[n]>=MaxPath)return false;
        if(n)children[parent[n]]=true;d.owner[n]=-1;
    }
    for(int n=0;n<count;++n)if(!children[n]){
        int id=d.leaf_count++;d.leaf[id]=n;
        for(int p=n;p>=0;p=d.parent[p])if(d.owner[p]<0)d.owner[p]=id;
    }
    result=d;return true;
}
inline bool Valid(const Descriptor& d){
    Descriptor expected;if(!Build(d.parent,d.count,expected))return false;
    if(d.leaf_count!=expected.leaf_count)return false;
    for(int i=0;i<MaxNodes;++i)if(d.parent[i]!=expected.parent[i]||d.depth[i]!=expected.depth[i]||d.leaf[i]!=expected.leaf[i]||d.owner[i]!=expected.owner[i])return false;
    return true;
}
__global__ void Convolution(Garnet::Qwen35DeltaConfig c,int batch,Descriptor d,const float* input,
    const float* old,const float* active,const float* weight,float* conv){
    int channels=2*c.kh*128+c.vh*128,i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=batch*d.count*channels)return;
    int ch=i%channels,node=(i/channels)%d.count,b=i/(channels*d.count);
    if(active[b*d.count+node]==0){conv[i]=0;return;}
    int path[MaxPath],length=0;
    for(int p=node;p>=0;p=d.parent[p])path[length++]=p;
    float h[4];for(int k=0;k<c.kernel;++k)h[k]=old[(b*channels+ch)*c.kernel+k];
    for(int j=length-1;j>=0;--j){int n=path[j];if(active[b*d.count+n]==0)continue;
        for(int k=0;k<c.kernel-1;++k)h[k]=h[k+1];
        h[c.kernel-1]=input[(b*d.count+n)*channels+ch];
    }
    float value=0;for(int k=0;k<c.kernel;++k)value=__fadd_rn(value,__fmul_rn(h[k],weight[ch*c.kernel+k]));
    conv[i]=value/(1.0f+expf(-value));
}
__global__ void Recurrent(Garnet::Qwen35DeltaConfig c,Descriptor d,const float* conv,const float* z,
    const float* beta,const float* a,const float* old,const float* active,const float* dt,const float* alog,
    const float* norm,float* output){
    using Reduce=cub::BlockReduce<float,128>;
    __shared__ typename Reduce::TempStorage temp;
    __shared__ volatile float q[128],k[128];__shared__ float factor;
    const int lane=threadIdx.x,head=blockIdx.x%c.vh,b=blockIdx.x/c.vh,leaf=blockIdx.y;
    const int kh=head/(c.vh/c.kh),channels=2*c.kh*128+c.vh*128;
    float state[128];
    #pragma unroll
    for(int j=0;j<128;++j)state[j]=old[((b*c.vh+head)*128+j)*128+lane];
    int path[MaxPath],length=0;for(int n=d.leaf[leaf];n>=0;n=d.parent[n])path[length++]=n;
    for(int p=length-1;p>=0;--p){
        int n=path[p],bt=b*d.count+n,offset=bt*channels;bool owns=d.owner[n]==leaf;
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
inline bool Overlap(const void* a,size_t n,const void* b,size_t m){
    auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);return x>=y?x-y<m:y-x<n;
}
inline int Run(const Garnet::Qwen35DeltaConfig& c,int batch,const Descriptor& d,const void* const* in,
    float* output,float* workspace,cudaStream_t stream){
    if(!Valid(d)||(batch!=1&&batch!=2&&batch!=4&&batch!=8)||c.kd!=128||c.vd!=128||c.kh<1||c.kh>64||c.vh<1||c.vh>64||c.vh%c.kh||c.kernel<1||c.kernel>4||!std::isfinite(c.epsilon)||c.epsilon<=0||!in||!output||!workspace)return 1;
    size_t ch=2*c.kh*128+c.vh*128,bt=size_t(batch)*d.count;
    size_t words[]={bt*ch,bt*c.vh*128,bt*c.vh,bt*c.vh,size_t(batch)*ch*c.kernel,
        size_t(batch)*c.vh*128*128,bt,ch*c.kernel,size_t(c.vh),size_t(c.vh),128};
    size_t outBytes=bt*c.vh*128*4,workBytes=bt*ch*4;
    if(reinterpret_cast<uintptr_t>(output)%4||reinterpret_cast<uintptr_t>(workspace)%4||Overlap(output,outBytes,workspace,workBytes))return 1;
    for(int i=0;i<11;++i)if(!in[i]||reinterpret_cast<uintptr_t>(in[i])%4||Overlap(output,outBytes,in[i],words[i]*4)||Overlap(workspace,workBytes,in[i],words[i]*4))return 1;
    Convolution<<<(bt*ch+127)/128,128,0,stream>>>(c,batch,d,(const float*)in[0],(const float*)in[4],(const float*)in[6],(const float*)in[7],workspace);
    if(cudaPeekAtLastError()!=cudaSuccess)return 1;
    Recurrent<<<dim3(batch*c.vh,d.leaf_count),128,0,stream>>>(c,d,workspace,(const float*)in[1],(const float*)in[2],(const float*)in[3],(const float*)in[5],(const float*)in[6],(const float*)in[8],(const float*)in[9],(const float*)in[10],output);
    return cudaPeekAtLastError()==cudaSuccess?0:1;
}
}
