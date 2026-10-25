// SPDX-License-Identifier: Apache-2.0
// Collect raw witnesses; independent Python FP64 replay must qualify numerics.
#include "bf16_kv_low_batch_candidate.h"
#include "bf16_low_batch_overlay_plugin.h"
#include "qwen38_bf16_kv_commit.cuh"
#include "bf16_kv_fixture_layout.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace Garnet;
namespace {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void require(bool okay,const char* s){if(!okay)throw std::runtime_error(s);}
struct Device {
    void* p=nullptr;size_t bytes;
    explicit Device(size_t n):bytes(n){check(cudaMalloc(&p,n));}
    ~Device(){if(p)cudaFree(p);}
    Device(const Device&)=delete;Device& operator=(const Device&)=delete;
    float* f(){return static_cast<float*>(p);}
};
struct Stream {cudaStream_t p=nullptr;Stream(){check(cudaStreamCreateWithFlags(&p,cudaStreamNonBlocking));}~Stream(){if(p)cudaStreamDestroy(p);}};
struct Spans {int start[8],count[8];};
__host__ __device__ uint32_t bits(uint32_t i,uint32_t salt,uint32_t exponent){
    uint32_t u=i*1664525u+salt;u^=u>>13;u*=0x85ebca6bu;u^=u>>16;
    return exponent|(u&0x807fffffu);
}
__host__ __device__ uint16_t bf16(uint32_t u){return uint16_t((u+0x7fffu+((u>>16)&1))>>16);}
__host__ __device__ uint32_t currentBits(uint32_t i,bool value){
    const uint32_t ties[4]={0x3f808000u,0x3f818000u,0xbf808000u,0xbf818000u};
    if(i%256<4)return ties[i%256];return bits(i,value?6197:5171,0x3f000000u);
}
float asFloat(uint32_t u){float v;std::memcpy(&v,&u,4);return v;}
__host__ __device__ uint16_t cacheBits(int b,int which,int h,int p,int d,int layer,int chunk,Spans spans,bool seeded){
    if(seeded&&p>=spans.start[b]&&p<spans.start[b]+spans.count[b]-1){
        const int t=p-spans.start[b];return bf16(currentBits(((b*chunk+t)*4+h)*256+d,which!=0));
    }
    const uint32_t i=((b*4+h)*4096+p)*256+d;
    return bf16(bits(i,(which?2017:1009)+layer*131,0x3f000000u));
}
__global__ void fillCache(float* state,int batch,int chunk,int layer,int64_t offset,Spans spans,bool seeded){
    const int64_t n=int64_t(batch)*2*4*4096*256;
    auto* raw=reinterpret_cast<uint16_t*>(state);
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=int64_t(gridDim.x)*blockDim.x){
        int64_t u=i;const int d=u%256;u/=256;const int p=u%4096;u/=4096;
        const int h=u%4;u/=4;const int which=u%2,b=u/2;
        raw[2*(int64_t(b)*Qwen38BF16KVCanonicalWidth+offset)+i%(2*4*4096*256)]=cacheBits(b,which,h,p,d,layer,chunk,spans,seeded);
    }
}
__global__ void checkCanonical(const float* state,int batch,int chunk,int layer,int64_t offset,Spans spans,bool seeded,unsigned long long* errors){
    const int64_t n=int64_t(batch)*Qwen38BF16KVCanonicalWidth;
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=int64_t(gridDim.x)*blockDim.x){
        const int b=i/Qwen38BF16KVCanonicalWidth;const int64_t col=i%Qwen38BF16KVCanonicalWidth;
        uint32_t expected=0xd3d3d3d3u;
        if(col>=offset&&col<offset+4194304){
            uint32_t packed=0;
            for(int half=0;half<2;++half){
                int64_t u=2*(col-offset)+half;const int d=u%256;u/=256;const int p=u%4096;u/=4096;
                const int h=u%4,which=u/4;
                packed|=uint32_t(cacheBits(b,which,h,p,d,layer,chunk,spans,seeded))<<(half*16);
            }
            expected=packed;
        }
        if(__float_as_uint(state[i])!=expected)atomicAdd(errors,1ull);
    }
}
void copied(const std::vector<float>& host,Device& gpu,cudaStream_t s){require(host.size()*4==gpu.bytes,"Input size mismatch");check(cudaMemcpyAsync(gpu.p,host.data(),gpu.bytes,cudaMemcpyHostToDevice,s));}
void readonly(const std::vector<float>& host,Device& gpu,cudaStream_t s){
    std::vector<float> back(host.size());check(cudaMemcpyAsync(back.data(),gpu.p,gpu.bytes,cudaMemcpyDeviceToHost,s));check(cudaStreamSynchronize(s));
    require(std::memcmp(back.data(),host.data(),gpu.bytes)==0,"Readonly small input changed");
}
void guards(Device& data,size_t body,cudaStream_t s){
    unsigned char back[32];check(cudaMemcpyAsync(back,data.p,32,cudaMemcpyDeviceToHost,s));check(cudaStreamSynchronize(s));
    for(auto v:back)require(v==0xa5,"Leading canary changed");
    check(cudaMemcpyAsync(back,static_cast<char*>(data.p)+32+body,32,cudaMemcpyDeviceToHost,s));check(cudaStreamSynchronize(s));
    for(auto v:back)require(v==0xa5,"Trailing canary changed");
}
void checkState(Device& state,Device& errors,int b,int t,int layer,int64_t offset,Spans spans,bool seeded,cudaStream_t s){
    check(cudaMemsetAsync(errors.p,0,8,s));
    checkCanonical<<<4096,256,0,s>>>(state.f(),b,t,layer,offset,spans,seeded,(unsigned long long*)errors.p);check(cudaGetLastError());
    unsigned long long count=0;check(cudaMemcpyAsync(&count,errors.p,8,cudaMemcpyDeviceToHost,s));check(cudaStreamSynchronize(s));require(count==0,"Readonly canonical carrier changed");
}
int pluginEnqueues=0;
int RunViaB1Plugin(const Qwen35AttentionCoreConfig& c,const void* const* inputs,void* result,void* workspace,int64_t ko,int64_t vo,cudaStream_t stream){
    BF16KVLowBatchOverlayConfig config{c,Qwen38BF16KVCanonicalWidth,ko,vo};
    require(ValidBF16KVLowBatchOverlayConfig(config),"Plugin config rejected");
    BF16KVLowBatchOverlayPlugin plugin(config);
    nvinfer1::PluginTensorDesc descriptors[8]{};
    const int64_t shapes[8][4]={{c.batch,c.tokens,24,256},{c.batch,c.tokens,4,256},{c.batch,c.tokens,4,256},
        {c.batch,c.tokens,24,256},{c.batch,Qwen38BF16KVCanonicalWidth,0,0},{c.batch,c.tokens,0,0},
        {c.batch,c.tokens,0,0},{c.batch,c.tokens,24,256}};
    for(int i=0;i<8;++i){auto& d=descriptors[i];d.type=nvinfer1::DataType::kFLOAT;d.format=nvinfer1::TensorFormat::kLINEAR;
        d.dims.nbDims=(i<4||i==7)?4:2;for(int j=0;j<d.dims.nbDims;++j)d.dims.d[j]=shapes[i][j];
        require(plugin.supportsFormatCombination(i,descriptors,7,1),"Plugin positive format rejected");}
    require(plugin.getWorkspaceSize(descriptors,7,descriptors+7,1)==Qwen38BF16KVLowBatchCandidateWorkspace(c),"Plugin workspace differs");
    void* outputs[1]={result};++pluginEnqueues;
    return plugin.enqueue(descriptors,descriptors+7,inputs,outputs,workspace,stream);
}
void execute(int id,int batch,int chunk,int layer,int end,std::ofstream& witnesses,uint64_t& witnessOffset){
    std::printf("{\"event\":\"case_started\",\"id\":%d,\"batch\":%d,\"chunk\":%d,\"layer\":%d,\"end\":%d}\n",id,batch,chunk,layer,end);std::fflush(stdout);
    Stream stream;const auto s=stream.p;Qwen35AttentionCoreConfig c{batch,chunk,24,4,4096,256};
    const auto item=(chunk==4?bf16Layers4:bf16Layers32)[layer];require(item.full,"Fixture layer is not full attention");
    const int64_t ko=item.canonical,vo=ko+2097152;
    const size_t nq=size_t(batch)*chunk*24*256,nkv=size_t(batch)*chunk*4*256,nctrl=batch*chunk;
    const size_t scratch=Qwen38BF16KVLowBatchCandidateWorkspace(c);require(scratch==size_t(batch)*chunk*24*4096*4,"Workspace geometry mismatch");
    Device q(nq*4),k(nkv*4),v(nkv*4),gate(nq*4),slots(nctrl*4),active(nctrl*4);
    Device state(size_t(batch)*Qwen38BF16KVCanonicalWidth*4),output(nq*4+64),work(scratch+64),errors(8);
    float* result=output.f()+8;void* workspace=work.f()+8;
    std::vector<float> hq(nq),hk(nkv),hv(nkv),hg(nq),hs(nctrl,4096),ha(nctrl,0);Spans spans{};
    for(size_t i=0;i<nq;++i){hq[i]=asFloat(bits(uint32_t(i),3119,0x3e800000u));hg[i]=asFloat(bits(uint32_t(i),4127,0x3f800000u));}
    for(size_t i=0;i<nkv;++i){hk[i]=asFloat(currentBits(uint32_t(i),false));hv[i]=asFloat(currentBits(uint32_t(i),true));}
    for(int b=0;b<batch;++b){
        spans.count[b]=end?std::min(std::min(chunk,end),1+(b*7+id)%chunk):0;spans.start[b]=end-spans.count[b];
        for(int t=0;t<spans.count[b];++t){hs[b*chunk+t]=float(spans.start[b]+t);ha[b*chunk+t]=1;}
    }
    copied(hq,q,s);copied(hk,k,s);copied(hv,v,s);copied(hg,gate,s);copied(hs,slots,s);copied(ha,active,s);
    check(cudaMemsetAsync(state.p,0xd3,state.bytes,s));fillCache<<<4096,256,0,s>>>(state.f(),batch,chunk,layer,ko,spans,false);
    check(cudaMemsetAsync(output.p,0xa5,output.bytes,s));check(cudaMemsetAsync(work.p,0xa5,work.bytes,s));check(cudaStreamSynchronize(s));
    const void* inputs[7]={q.p,k.p,v.p,gate.p,state.p,slots.p,active.p};int rejected=0;
    auto reject=[&](Qwen35AttentionCoreConfig config,const void* const* in,void* dst,void* tmp,int64_t key,int64_t value){
        require(RunQwen38BF16KVLowBatchCandidate(config,in,dst,tmp,key,value,s)!=0,"Malformed/aliased operator call accepted");++rejected;
    };
    if(id==0||id==14){
        reject(c,nullptr,result,workspace,ko,vo);reject(c,inputs,nullptr,workspace,ko,vo);reject(c,inputs,result,nullptr,ko,vo);
        for(int i=0;i<7;++i){
            const void* bad[7];std::copy(inputs,inputs+7,bad);bad[i]=nullptr;reject(c,bad,result,workspace,ko,vo);
            reject(c,inputs,const_cast<void*>(inputs[i]),workspace,ko,vo);
            reject(c,inputs,result,const_cast<void*>(inputs[i]),ko,vo);
            reject(c,inputs,static_cast<char*>(const_cast<void*>(inputs[i]))+4,workspace,ko,vo);
            reject(c,inputs,result,static_cast<char*>(const_cast<void*>(inputs[i]))+4,ko,vo);
        }
        reject(c,inputs,result,result,ko,vo);
        for(int mode=0;mode<6;++mode){auto bad=c;if(mode==0)bad.batch=2;if(mode==1)bad.tokens=8;if(mode==2)bad.heads=23;if(mode==3)bad.kv=3;if(mode==4)bad.capacity=2048;if(mode==5)bad.dim=128;require(Qwen38BF16KVLowBatchCandidateWorkspace(bad)==0,"Malformed workspace admitted");reject(bad,inputs,result,workspace,ko,vo);}
        for(int badBatch:{4,8}){auto bad=c;bad.batch=badBatch;require(Qwen38BF16KVLowBatchCandidateWorkspace(bad)==0,"Legacy batch admitted into low-batch profile");reject(bad,inputs,result,workspace,ko,vo);}
        for(auto key:{int64_t(-1),ko+1,int64_t(2482176)+16*6676480,int64_t(2482176)+10870784})reject(c,inputs,result,workspace,key,key+2097152);
        reject(c,inputs,result,workspace,ko,vo+1);
        checkState(state,errors,batch,chunk,layer,ko,spans,false,s);
    }
    require(RunViaB1Plugin(c,inputs,result,workspace,ko,vo,s)==0,"Valid overlay failed");
    std::vector<float> original(nq);check(cudaMemcpyAsync(original.data(),result,nq*4,cudaMemcpyDeviceToHost,s));check(cudaStreamSynchronize(s));
    for(size_t i=0;i<nq;++i){require(std::isfinite(original[i]),"Nonfinite context");const int bt=i/(24*256);if(!ha[bt])require(original[i]==0,"Inactive context is nonzero");}
    const bool graph=(id==1||id==15||id==63);
    if(graph){
        cudaGraph_t captured=nullptr;cudaGraphExec_t execution=nullptr;
        check(cudaStreamBeginCapture(s,cudaStreamCaptureModeGlobal));
        require(RunViaB1Plugin(c,inputs,result,workspace,ko,vo,s)==0,"Captured overlay failed");
        check(cudaStreamEndCapture(s,&captured));check(cudaGraphInstantiate(&execution,captured,0));
        check(cudaMemsetAsync(result,0xa5,nq*4,s));check(cudaGraphLaunch(execution,s));
        std::vector<float> back(nq);check(cudaMemcpyAsync(back.data(),result,nq*4,cudaMemcpyDeviceToHost,s));check(cudaStreamSynchronize(s));
        check(cudaGraphExecDestroy(execution));check(cudaGraphDestroy(captured));require(std::memcmp(back.data(),original.data(),nq*4)==0,"Graph replay changed context");
    }
    checkState(state,errors,batch,chunk,layer,ko,spans,false,s);
    readonly(hq,q,s);readonly(hk,k,s);readonly(hv,v,s);readonly(hg,gate,s);readonly(hs,slots,s);readonly(ha,active,s);
    guards(output,nq*4,s);guards(work,scratch,s);
    const uint64_t startOffset=witnessOffset;std::vector<int> sampleTriples;
    for(int b=0;b<batch;++b)if(spans.count[b]){
        std::vector<int> ts{0,spans.count[b]/2,spans.count[b]-1};std::sort(ts.begin(),ts.end());ts.erase(std::unique(ts.begin(),ts.end()),ts.end());
        for(int t:ts)for(int h:{0,5,6,12,23}){
            const float* row=original.data()+((b*chunk+t)*24+h)*256;
            witnesses.write(reinterpret_cast<const char*>(row),256*4);witnessOffset+=256*4;sampleTriples.insert(sampleTriples.end(),{b,t,h});
        }
    }
    if(end){
        fillCache<<<4096,256,0,s>>>(state.f(),batch,chunk,layer,ko,spans,true);
        std::vector<float> nextQ=hq,nextK=hk,nextV=hv,nextG=hg,nextS(nctrl,4096),nextA(nctrl,0);
        for(int b=0;b<batch;++b){
            const int last=spans.count[b]-1;
            std::copy_n(hq.data()+(b*chunk+last)*24*256,24*256,nextQ.data()+b*chunk*24*256);
            std::copy_n(hg.data()+(b*chunk+last)*24*256,24*256,nextG.data()+b*chunk*24*256);
            std::copy_n(hk.data()+(b*chunk+last)*4*256,4*256,nextK.data()+b*chunk*4*256);
            std::copy_n(hv.data()+(b*chunk+last)*4*256,4*256,nextV.data()+b*chunk*4*256);
            nextS[b*chunk]=float(end-1);nextA[b*chunk]=1;
        }
        copied(nextQ,q,s);copied(nextK,k,s);copied(nextV,v,s);copied(nextG,gate,s);copied(nextS,slots,s);copied(nextA,active,s);
        check(cudaMemsetAsync(result,0xa5,nq*4,s));require(RunViaB1Plugin(c,inputs,result,workspace,ko,vo,s)==0,"Cached-prefix overlay failed");
        std::vector<float> back(nq);check(cudaMemcpyAsync(back.data(),result,nq*4,cudaMemcpyDeviceToHost,s));check(cudaStreamSynchronize(s));
        for(int b=0;b<batch;++b){
            require(std::memcmp(back.data()+b*chunk*24*256,original.data()+(b*chunk+spans.count[b]-1)*24*256,24*256*4)==0,"Chunk/cache BF16 precision differs");
            for(int t=1;t<chunk;++t)for(int i=0;i<24*256;++i)require(back[(b*chunk+t)*24*256+i]==0,"Cached-prefix inactive context nonzero");
        }
        checkState(state,errors,batch,chunk,layer,ko,spans,true,s);
        readonly(nextQ,q,s);readonly(nextK,k,s);readonly(nextV,v,s);readonly(nextG,gate,s);readonly(nextS,slots,s);readonly(nextA,active,s);
        guards(output,nq*4,s);guards(work,scratch,s);
    }
    std::printf("{\"event\":\"case_collected\",\"id\":%d,\"batch\":%d,\"chunk\":%d,\"layer\":%d,\"end\":%d,\"witness_offset\":%llu,\"witness_bytes\":%llu,\"rejected_calls\":%d,\"graph_replay_exact\":%s,\"prefix_cache_exact\":%s,\"readonly_exact\":true,\"canaries_exact\":true,\"samples\":[",id,batch,chunk,layer,end,(unsigned long long)startOffset,(unsigned long long)(witnessOffset-startOffset),rejected,graph?"true":"false",end?"true":"false");
    for(size_t i=0;i<sampleTriples.size();i+=3)std::printf("%s[%d,%d,%d]",i?",":"",sampleTriples[i],sampleTriples[i+1],sampleTriples[i+2]);
    std::puts("]}");std::fflush(stdout);require(bool(witnesses),"Witness output failed");
}
}
int main(int argc,char** argv){
    try{
        require(argc==2,"Pass a new witness binary path");std::ifstream existing(argv[1],std::ios::binary);require(!existing.good(),"Preserve old witness file");
        std::ofstream witnesses(argv[1],std::ios::binary|std::ios::out);require(bool(witnesses),"Cannot create witness file");
        require(EnsureBF16KVLowBatchOverlayPluginRegistered(),"Plugin registry failed");
        int count=0;check(cudaGetDeviceCount(&count));require(count==1,"Single visible GPU required");cudaDeviceProp p{};check(cudaGetDeviceProperties(&p,0));
        require(p.major==12&&p.minor==0&&std::string(p.name).find("5090")!=std::string::npos,"Dedicated RTX5090 required");
        std::printf("{\"event\":\"gpu_identity\",\"name\":\"%s\",\"sm\":120,\"uuid\":\"GPU-",p.name);
        for(int i=0;i<16;++i){if(i==4||i==6||i==8||i==10)std::printf("-");std::printf("%02x",static_cast<unsigned char>(p.uuid.bytes[i]));}std::puts("\"}");std::fflush(stdout);
        int id=0;uint64_t offset=0;
        for(int batch:{1})for(int chunk:{4,32})for(int layer:{3,63})for(int end:{0,1,511,512,513,4095,4096})execute(id++,batch,chunk,layer,end,witnesses,offset);
        for(int layer=7;layer<63;layer+=4)execute(id++,1,4,layer,513,witnesses,offset);
        execute(63,1,32,3,32,witnesses,offset);execute(95,1,32,63,4096,witnesses,offset);id+=2;
        witnesses.close();require(bool(witnesses)&&id==44&&pluginEnqueues==87,"Witness/plugin-call completion failed");
        std::printf("{\"status\":\"bf16_low_batch_plugin_collected_needs_fp64_replay\",\"cases\":%d,\"witness_bytes\":%llu,\"host_plugin_enqueue_calls\":87,\"negative_calls_use_direct_kernel\":true,\"performance_admitted\":false,\"model_qualified\":false}\n",id,(unsigned long long)offset);return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
