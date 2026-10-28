// SPDX-License-Identifier: Apache-2.0
// Standalone study probe, no Garnet DLL and no timing/performance admission.
#include "qwen38_bf16_kv_commit.cuh"
#include "bf16_kv_fixture_layout.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
using namespace Garnet;
namespace {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void require(bool okay,const char* message){if(!okay)throw std::runtime_error(message);}
struct Device {
    void* data=nullptr;size_t bytes;
    explicit Device(size_t n):bytes(n){check(cudaMalloc(&data,n));}
    ~Device(){if(data)cudaFree(data);}
    Device(const Device&)=delete;Device& operator=(const Device&)=delete;
    float* floats(){return static_cast<float*>(data);}
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
void exercise(int batch,int chunk){
    const int64_t targetWidth=chunk==4?43030528:40927232;
    Device target(size_t(batch)*targetWidth*4),state(size_t(batch)*Qwen38BF16KVCanonicalWidth*4);
    Device replay(size_t(batch)*Qwen38BF16KVReplayWidth*4),errors(8);
    fill<<<4096,256>>>(target.floats(),size_t(batch)*targetWidth,19);
    fill<<<4096,256>>>(replay.floats(),size_t(batch)*Qwen38BF16KVReplayWidth,811);check(cudaGetLastError());
    check(cudaMemset(state.data,0xd3,state.bytes));Qwen38BF16KVSpans zero{};
    int rejected=0;
    auto call=[&](const float* t,const float* l,int64_t w,float* s,int b,int c,Qwen38BF16KVSpans p){
        const int status=RunQwen38BF16KVCommit(t,l,w,s,b,c,p,nullptr);
        if(status!=0)++rejected;return status;
    };
    require(call(target.floats(),target.floats(),targetWidth,state.floats(),batch,chunk,zero)==0,"Zero-prefix reject");
    require(call(nullptr,target.floats(),targetWidth,state.floats(),batch,chunk,zero)!=0,"Null target accepted");
    require(call(target.floats(),nullptr,targetWidth,state.floats(),batch,chunk,zero)!=0,"Null linear accepted");
    require(call(target.floats(),target.floats(),targetWidth,nullptr,batch,chunk,zero)!=0,"Null canonical accepted");
    require(call(target.floats(),target.floats(),targetWidth,target.floats(),batch,chunk,zero)!=0,"Canonical alias accepted");
    require(call(target.floats(),target.floats(),targetWidth,target.floats()+1,batch,chunk,zero)!=0,"Partial canonical alias accepted");
    require(call(target.floats(),replay.floats(),targetWidth,state.floats(),batch,chunk,zero)!=0,"Wrong full linear source accepted");
    require(call(target.floats(),target.floats(),targetWidth-1,state.floats(),batch,chunk,zero)!=0,"Wrong width accepted");
    require(call(target.floats(),target.floats(),targetWidth,state.floats(),2,chunk,zero)!=0,"Wrong batch accepted");
    require(call(target.floats(),target.floats(),targetWidth,state.floats(),batch,8,zero)!=0,"Wrong chunk accepted");
    if(chunk==32)require(call(target.floats(),replay.floats(),Qwen38BF16KVReplayWidth,state.floats(),batch,chunk,zero)!=0,"T32 replay accepted");
    if(chunk==4)require(call(target.floats(),state.floats()+1,Qwen38BF16KVReplayWidth,state.floats(),batch,chunk,zero)!=0,"Canonical/replay partial alias accepted");
    if(batch==4){auto bad=zero;bad.count[4]=1;require(call(target.floats(),target.floats(),targetWidth,state.floats(),batch,chunk,bad)!=0,"Unused row accepted");}
    for(int mode=0;mode<5;++mode){
        auto bad=zero;
        if(mode==0)bad.start[batch-1]=-1;
        if(mode==1)bad.start[batch-1]=4097;
        if(mode==2)bad.count[batch-1]=-1;
        if(mode==3)bad.count[batch-1]=chunk+1;
        if(mode==4){bad.start[batch-1]=4096;bad.count[batch-1]=1;}
        require(call(target.floats(),target.floats(),targetWidth,state.floats(),batch,chunk,bad)!=0,"Invalid span accepted");
    }
    verify(state,errors,batch,chunk,targetWidth,zero);
    for(int trial=0;trial<5;++trial){
        Qwen38BF16KVSpans spans{};
        for(int b=0;b<batch;++b){
            const int counts4[5]={0,1,2,3,4},counts32[5]={0,1,16,31,32};
            spans.count[b]=(chunk==4?counts4:counts32)[(b+trial)%5];
            const int starts[5]={0,511,512,513,4096-chunk};spans.start[b]=starts[(b+2*trial)%5];
            if(!spans.count[b])spans.start[b]=4096;
        }
        for(int source=0;source<(chunk==4?2:1);++source){
            check(cudaMemset(state.data,0xd3,state.bytes));
            const int64_t width=source?Qwen38BF16KVReplayWidth:targetWidth;
            const float* linear=source?replay.floats():target.floats();
            require(call(target.floats(),linear,width,state.floats(),batch,chunk,spans)==0,"Valid commit failed");
            verify(state,errors,batch,chunk,width,spans);
        }
    }
    check(cudaMemset(errors.data,0,errors.bytes));
    verifyReadOnly<<<4096,256>>>(target.floats(),size_t(batch)*targetWidth,19,(unsigned long long*)errors.data);
    verifyReadOnly<<<4096,256>>>(replay.floats(),size_t(batch)*Qwen38BF16KVReplayWidth,811,(unsigned long long*)errors.data);
    check(cudaGetLastError());unsigned long long changed=0;
    check(cudaMemcpy(&changed,errors.data,sizeof(changed),cudaMemcpyDeviceToHost));require(changed==0,"Readonly input modified");
    require(rejected==14+(batch==4?1:0)+1,"Negative-call accounting changed");
    std::printf("{\"event\":\"profile_passed\",\"batch\":%d,\"chunk\":%d,\"full_state_byte_cases\":%d,\"rejected_calls\":%d,\"readonly_inputs_exact\":true}\n",batch,chunk,chunk==4?11:6,rejected);std::fflush(stdout);
}
}
int main(){
    try{
        int count=0;check(cudaGetDeviceCount(&count));require(count==1,"Dedicated single visible GPU required");
        cudaDeviceProp p{};check(cudaGetDeviceProperties(&p,0));require(p.major==12&&p.minor==0,"SM120 required");
        require(std::string(p.name).find("5090")!=std::string::npos,"RTX5090 required");
        std::printf("{\"event\":\"gpu_identity\",\"name\":\"%s\",\"sm\":120,\"uuid\":\"GPU-",p.name);
        for(int i=0;i<16;++i){if(i==4||i==6||i==8||i==10)std::printf("-");std::printf("%02x",static_cast<unsigned char>(p.uuid.bytes[i]));}
        std::puts("\"}");std::fflush(stdout);
        for(int batch:{4,8})for(int chunk:{4,32})exercise(batch,chunk);
        std::puts("{\"status\":\"bf16_kv_commit_native_operator_passed\",\"performance_admitted\":false,\"model_qualified\":false}");return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
