// SPDX-License-Identifier: Apache-2.0
// Standalone study probe, no Garnet DLL and no timing/performance admission.
#include "bf16_low_batch_commit_candidate.h"
#include "bf16_kv_fixture_layout.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <vector>
using namespace Garnet;
namespace {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void require(bool okay,const char* message){if(!okay)throw std::runtime_error(message);}
struct Device {
    void* allocation=nullptr;void* data=nullptr;size_t bytes;
    explicit Device(size_t n):bytes(n){check(cudaMalloc(&allocation,n+64));data=static_cast<unsigned char*>(allocation)+32;check(cudaMemset(allocation,0x5a,n+64));}
    ~Device(){if(allocation)cudaFree(allocation);}
    Device(const Device&)=delete;Device& operator=(const Device&)=delete;
    float* floats(){return static_cast<float*>(data);}
    void guards(){std::array<unsigned char,32> a{},b{};check(cudaMemcpy(a.data(),allocation,32,cudaMemcpyDeviceToHost));check(cudaMemcpy(b.data(),static_cast<unsigned char*>(data)+bytes,32,cudaMemcpyDeviceToHost));for(auto x:a)require(x==0x5a,"Leading device canary changed");for(auto x:b)require(x==0x5a,"Trailing device canary changed");}
};
__host__ __device__ uint32_t pattern(uint64_t i,uint32_t salt){
    return 0x3f000000u|((uint32_t(i)*97u+salt)&0x7fffffu)|((i&1)?0x80000000u:0);
}
__global__ void fill(float* data,uint64_t n,uint32_t salt){
    for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=uint64_t(gridDim.x)*blockDim.x)
        data[i]=__uint_as_float(pattern(i,salt));
}
__global__ void verifyReadOnly(const float* data,uint64_t n,uint32_t salt,unsigned long long* errors){
    for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=uint64_t(gridDim.x)*blockDim.x)
        if(__float_as_uint(data[i])!=pattern(i,salt))atomicAdd(errors,1ull);
}
__global__ void verifyLayer(const float* state,int64_t targetWidth,int64_t sourceWidth,
    int chunk,BF16LayerFixture layer,Qwen38BF16KVSpans spans,unsigned long long* errors){
    const int b=blockIdx.y;
    const int64_t width=layer.full?layer.width*2:layer.width;
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<width;i+=int64_t(gridDim.x)*blockDim.x){
        bool same;
        if(!layer.full){
            uint32_t expected=0xd3d3d3d3u;
            if(spans.count[b]){
                const int64_t offset=sourceWidth==targetWidth?layer.target:layer.replay;
                expected=pattern(int64_t(b)*sourceWidth+offset+i,sourceWidth==targetWidth?19:811);
            }
            same=__float_as_uint(state[int64_t(b)*Qwen38BF16KVCanonicalWidth+layer.canonical+i])==expected;
        }else{
            int64_t index=i;const int d=index%256;index/=256;const int slot=index%4096;index/=4096;
            const int head=index%4,which=index/4;uint16_t expected=0xd3d3u;
            if(slot>=spans.start[b]&&slot<spans.start[b]+spans.count[b]){
                const int t=slot-spans.start[b];
                const uint32_t bits=pattern(int64_t(b)*targetWidth+layer.target+
                    which*chunk*1024+(t*4+head)*256+d,19);
                expected=uint16_t((bits+0x7fffu+((bits>>16)&1))>>16);
            }
            same=reinterpret_cast<const uint16_t*>(state)[2*(int64_t(b)*Qwen38BF16KVCanonicalWidth+layer.canonical)+i]==expected;
        }
        if(!same)atomicAdd(errors,1ull);
    }
}
void verify(Device& state,Device& errors,int batch,int chunk,int64_t sourceWidth,Qwen38BF16KVSpans spans){
    check(cudaMemset(errors.data,0,errors.bytes));
    const auto* layers=chunk==4?bf16Layers4:bf16Layers32;
    const int64_t targetWidth=chunk==4?43030528:40927232;
    for(int i=0;i<64;++i)verifyLayer<<<dim3(1024,batch),256>>>(state.floats(),targetWidth,sourceWidth,chunk,layers[i],spans,(unsigned long long*)errors.data);
    check(cudaGetLastError());unsigned long long mismatches=0;
    check(cudaMemcpy(&mismatches,errors.data,sizeof(mismatches),cudaMemcpyDeviceToHost));
    require(mismatches==0,"Canonical state byte mismatch");
}
void observe(Device& state,Device& target,Device& replay,Device& errors,int chunk,
    int source,int trial,const char* phase,Qwen38BF16KVSpans spans,
    std::ofstream& witnesses,uint64_t& offset,int& id){
    const auto* layers=chunk==4?bf16Layers4:bf16Layers32;
    const int64_t targetWidth=chunk==4?43030528:40927232;
    const int64_t sourceWidth=source?Qwen38BF16KVReplayWidth:targetWidth;
    // Independent host construction and comparison of EVERY canonical byte.
    // Integer BF16 RNE avoids relying on host floating point modes or CUDA.
    std::vector<uint32_t> expected(size_t(Qwen38BF16KVCanonicalWidth),0xd3d3d3d3u);
    const int start=spans.start[0],count=spans.count[0];
    for(int l=0;l<64;++l){
        const auto layer=layers[l];
        if(!layer.full){
            if(count)for(int64_t i=0;i<layer.width;++i)
                expected[size_t(layer.canonical+i)]=pattern((source?layer.replay:layer.target)+i,source?811:19);
        }else for(int which=0;which<2;++which)for(int h=0;h<4;++h)
            for(int t=0;t<count;++t)for(int d=0;d<256;++d){
                uint32_t bits=pattern(layer.target+which*chunk*1024+(t*4+h)*256+d,19);
                uint16_t rounded=uint16_t((bits+0x7fffu+((bits>>16)&1u))>>16);
                const size_t at=4*size_t(layer.canonical)+2*size_t(((which*4+h)*4096+start+t)*256+d);
                std::memcpy(reinterpret_cast<unsigned char*>(expected.data())+at,&rounded,2);
            }
    }
    std::vector<uint32_t> actual(size_t(Qwen38BF16KVCanonicalWidth),0u);
    check(cudaMemcpy(actual.data(),state.data,state.bytes,cudaMemcpyDeviceToHost));
    require(std::memcmp(actual.data(),expected.data(),state.bytes)==0,"Independent full host canonical bytes differ");
    std::vector<int> slots{0,4095};
    if(start)slots.push_back(start-1);if(start<4096)slots.push_back(start);
    if(count)slots.push_back(start+count-1);if(start+count<4096)slots.push_back(start+count);
    std::sort(slots.begin(),slots.end());slots.erase(std::unique(slots.begin(),slots.end()),slots.end());
    const auto* bytes=reinterpret_cast<const unsigned char*>(actual.data());std::vector<unsigned char> raw;
    auto append=[&](size_t at,int width){raw.insert(raw.end(),bytes+at,bytes+at+width);};
    for(int l=0;l<64;++l){const auto layer=layers[l];
        if(!layer.full){for(int64_t i:{int64_t(0),layer.width/2,layer.width-1})append(size_t(layer.canonical+i)*4,4);}
        else for(int which=0;which<2;++which)for(int h=0;h<4;++h)for(int slot:slots)for(int d:{0,1,128,255})
            append(4*size_t(layer.canonical)+2*size_t(((which*4+h)*4096+slot)*256+d),2);
    }
    state.guards();target.guards();replay.guards();errors.guards();
    witnesses.write(reinterpret_cast<const char*>(raw.data()),std::streamsize(raw.size()));require(bool(witnesses),"Witness write failed");
    std::printf("{\"event\":\"case_collected\",\"id\":%d,\"batch\":1,\"chunk\":%d,\"source\":%d,\"trial\":%d,\"phase\":\"%s\",\"start\":%d,\"count\":%d,\"witness_offset\":%llu,\"witness_bytes\":%llu,\"gpu_full_state_exact\":true,\"host_full_state_exact\":true,\"canaries_exact\":true}\n",id++,chunk,source,trial,phase,start,count,(unsigned long long)offset,(unsigned long long)raw.size());
    offset+=raw.size();std::fflush(stdout);
}
void exercise(int chunk,std::ofstream& witnesses,uint64_t& offset,int& id){
    const int batch=1;const int64_t targetWidth=chunk==4?43030528:40927232;
    Device target(size_t(batch)*targetWidth*4),state(size_t(batch)*Qwen38BF16KVCanonicalWidth*4);
    Device replay(size_t(batch)*Qwen38BF16KVReplayWidth*4),errors(8);
    fill<<<4096,256>>>(target.floats(),size_t(batch)*targetWidth,19);
    fill<<<4096,256>>>(replay.floats(),size_t(batch)*Qwen38BF16KVReplayWidth,811);check(cudaGetLastError());
    check(cudaMemset(state.data,0xd3,state.bytes));Qwen38BF16KVSpans zero{};int rejected=0,graphs=0;
    auto call=[&](const float* t,const float* l,int64_t w,float* s,int b,int c,Qwen38BF16KVSpans p){
        const int status=RunQwen38BF16LowBatchCommitCandidate(t,l,w,s,b,c,p,nullptr);if(status!=0)++rejected;return status;
    };
    require(call(target.floats(),target.floats(),targetWidth,state.floats(),batch,chunk,zero)==0,"Zero-prefix reject");
    require(call(nullptr,target.floats(),targetWidth,state.floats(),batch,chunk,zero)!=0,"Null target accepted");
    require(call(target.floats(),nullptr,targetWidth,state.floats(),batch,chunk,zero)!=0,"Null linear accepted");
    require(call(target.floats(),target.floats(),targetWidth,nullptr,batch,chunk,zero)!=0,"Null canonical accepted");
    require(call(target.floats(),target.floats(),targetWidth,target.floats(),batch,chunk,zero)!=0,"Canonical alias accepted");
    require(call(target.floats(),target.floats(),targetWidth,target.floats()+1,batch,chunk,zero)!=0,"Partial canonical alias accepted");
    require(call(target.floats(),replay.floats(),targetWidth,state.floats(),batch,chunk,zero)!=0,"Wrong full linear source accepted");
    require(call(target.floats(),target.floats(),targetWidth-1,state.floats(),batch,chunk,zero)!=0,"Wrong width accepted");
    for(int badBatch:{-1,0,2,4,8})require(call(target.floats(),target.floats(),targetWidth,state.floats(),badBatch,chunk,zero)!=0,"Unqualified batch accepted");
    require(call(target.floats(),target.floats(),targetWidth,state.floats(),batch,8,zero)!=0,"Wrong chunk accepted");
    if(chunk==32)require(call(target.floats(),replay.floats(),Qwen38BF16KVReplayWidth,state.floats(),batch,chunk,zero)!=0,"T32 replay accepted");
    if(chunk==4)require(call(target.floats(),state.floats()+1,Qwen38BF16KVReplayWidth,state.floats(),batch,chunk,zero)!=0,"Canonical/replay partial alias accepted");
    for(int row=1;row<8;++row)for(int field=0;field<2;++field){auto bad=zero;bad.start[row]=field?0:1;bad.count[row]=field?1:0;
        require(call(target.floats(),target.floats(),targetWidth,state.floats(),batch,chunk,bad)!=0,"Unused row accepted");}
    for(int mode=0;mode<5;++mode){auto bad=zero;
        if(mode==0)bad.start[0]=-1;if(mode==1)bad.start[0]=4097;if(mode==2)bad.count[0]=-1;
        if(mode==3)bad.count[0]=chunk+1;if(mode==4){bad.start[0]=4096;bad.count[0]=1;}
        require(call(target.floats(),target.floats(),targetWidth,state.floats(),batch,chunk,bad)!=0,"Invalid span accepted");
    }
    verify(state,errors,batch,chunk,targetWidth,zero);observe(state,target,replay,errors,chunk,0,-1,"zero",zero,witnesses,offset,id);
    for(int trial=0;trial<5;++trial){Qwen38BF16KVSpans spans{};
        const int counts4[5]={0,1,2,3,4},counts32[5]={0,1,16,31,32},starts[5]={0,511,512,513,4096-chunk};
        spans.count[0]=(chunk==4?counts4:counts32)[trial];spans.start[0]=spans.count[0]?starts[(2*trial)%5]:4096;
        for(int source=0;source<(chunk==4?2:1);++source){check(cudaMemset(state.data,0xd3,state.bytes));
            const int64_t width=source?Qwen38BF16KVReplayWidth:targetWidth;const float* linear=source?replay.floats():target.floats();
            require(call(target.floats(),linear,width,state.floats(),batch,chunk,spans)==0,"Valid commit failed");
            verify(state,errors,batch,chunk,width,spans);observe(state,target,replay,errors,chunk,source,trial,"trial",spans,witnesses,offset,id);
        }
    }
    for(int source=0;source<(chunk==4?2:1);++source){Qwen38BF16KVSpans spans{};spans.start[0]=4096-chunk;spans.count[0]=chunk;
        check(cudaMemset(state.data,0xd3,state.bytes));check(cudaDeviceSynchronize());
        cudaStream_t stream=nullptr;cudaGraph_t captured=nullptr;cudaGraphExec_t execution=nullptr;check(cudaStreamCreate(&stream));
        const int64_t width=source?Qwen38BF16KVReplayWidth:targetWidth;const float* linear=source?replay.floats():target.floats();
        check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
        require(RunQwen38BF16LowBatchCommitCandidate(target.floats(),linear,width,state.floats(),batch,chunk,spans,stream)==0,"Graph capture call failed");
        check(cudaStreamEndCapture(stream,&captured));check(cudaGraphInstantiate(&execution,captured,0));
        for(int repeat=0;repeat<3;++repeat){check(cudaMemsetAsync(state.data,0xd3,state.bytes,stream));check(cudaGraphLaunch(execution,stream));check(cudaStreamSynchronize(stream));verify(state,errors,batch,chunk,width,spans);}
        check(cudaGraphExecDestroy(execution));check(cudaGraphDestroy(captured));check(cudaStreamDestroy(stream));++graphs;
        observe(state,target,replay,errors,chunk,source,-1,"graph",spans,witnesses,offset,id);
    }
    check(cudaMemset(errors.data,0,errors.bytes));
    verifyReadOnly<<<4096,256>>>(target.floats(),size_t(batch)*targetWidth,19,(unsigned long long*)errors.data);
    verifyReadOnly<<<4096,256>>>(replay.floats(),size_t(batch)*Qwen38BF16KVReplayWidth,811,(unsigned long long*)errors.data);
    check(cudaGetLastError());unsigned long long changed=0;check(cudaMemcpy(&changed,errors.data,sizeof(changed),cudaMemcpyDeviceToHost));require(changed==0,"Readonly input modified");
    state.guards();target.guards();replay.guards();errors.guards();require(rejected==33,"Negative accounting changed");
    std::printf("{\"event\":\"profile_passed\",\"batch\":1,\"chunk\":%d,\"full_state_byte_cases\":%d,\"graph_profiles\":%d,\"graph_replays_each\":3,\"rejected_calls\":%d,\"readonly_inputs_exact\":true,\"canaries_exact\":true}\n",chunk,chunk==4?13:7,graphs,rejected);std::fflush(stdout);
}
}
int main(int argc,char** argv){
    try{
        require(argc==2,"Pass a new witness binary path");std::ifstream existing(argv[1],std::ios::binary);require(!existing.good(),"Preserve old witness file");
        std::ofstream witnesses(argv[1],std::ios::binary|std::ios::out);require(bool(witnesses),"Cannot create witness file");
        int count=0;check(cudaGetDeviceCount(&count));require(count==1,"Single visible GPU required");cudaDeviceProp p{};check(cudaGetDeviceProperties(&p,0));require(p.major==12&&p.minor==0&&std::string(p.name).find("5090")!=std::string::npos,"Dedicated RTX5090 required");
        std::printf("{\"event\":\"gpu_identity\",\"name\":\"%s\",\"sm\":120,\"uuid\":\"GPU-",p.name);
        for(int i=0;i<16;++i){if(i==4||i==6||i==8||i==10)std::printf("-");std::printf("%02x",static_cast<unsigned char>(p.uuid.bytes[i]));}std::puts("\"}");std::fflush(stdout);
        int id=0;uint64_t offset=0;for(int chunk:{4,32})exercise(chunk,witnesses,offset,id);
        witnesses.close();require(bool(witnesses)&&id==20,"Witness completion failed");
        std::printf("{\"status\":\"bf16_low_batch_commit_collected_needs_byte_replay\",\"cases\":20,\"witness_bytes\":%llu,\"performance_admitted\":false,\"model_qualified\":false}\n",(unsigned long long)offset);return 0;
    }catch(const std::exception& error){std::fprintf(stderr,"%s\n",error.what());return 1;}
}
