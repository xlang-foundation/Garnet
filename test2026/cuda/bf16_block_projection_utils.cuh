#pragma once
// SPDX-License-Identifier: Apache-2.0
// Unwired component: exact BF16-bit decoding, then the original production GEMM.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
using json=nlohmann::json;
namespace Garnet {
std::shared_ptr<void> CreateBF16ProjectionContext();
int RunBF16Projection(const std::shared_ptr<void>&,const void*,const void*,void*,int,int,int,cudaStream_t,void*,size_t) noexcept;
}
void check(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));}
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
std::vector<unsigned char> bytes(const std::string& path){
    std::ifstream stream(path,std::ios::binary|std::ios::ate);require(bool(stream),"Missing fixture");
    size_t size=size_t(stream.tellg());stream.seekg(0);std::vector<unsigned char> result(size);
    stream.read(reinterpret_cast<char*>(result.data()),size);require(bool(stream),"Fixture read failed");return result;
}
struct Buffer {
    unsigned char* owner=nullptr;unsigned char* data=nullptr;size_t size=0;
    explicit Buffer(size_t n):size(n){check(cudaMalloc(&owner,n+512));data=owner+256;check(cudaMemset(owner,0xa5,n+512));check(cudaDeviceSynchronize());}
    ~Buffer(){if(owner)cudaFree(owner);}
    Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
    bool guards()const{
        std::array<unsigned char,512> host{};check(cudaMemcpy(host.data(),owner,256,cudaMemcpyDeviceToHost));
        check(cudaMemcpy(host.data()+256,data+size,256,cudaMemcpyDeviceToHost));
        return std::all_of(host.begin(),host.end(),[](unsigned char v){return v==0xa5;});
    }
};
template<bool Streaming> __device__ unsigned char loadByte(const unsigned char* p){
    if constexpr(Streaming)return static_cast<unsigned char>(__ldcs(reinterpret_cast<const char*>(p)));
    else return *p;
}
template<bool Streaming> __device__ unsigned short loadWord(const unsigned char* p){return unsigned(loadByte<Streaming>(p))|(unsigned(loadByte<Streaming>(p+1))<<8);}
template<bool Streaming> __device__ unsigned short unpack(unsigned code,unsigned base,const unsigned char* payload){
    unsigned delta=code&15;
    if(delta==15)return loadWord<Streaming>(payload+384+2*(code>>4));
    return ((code&0x800)<<4)|((base-delta)<<7)|((code>>4)&127);
}
template<bool Streaming> __global__ void decodeBlocks(const unsigned char* blob,unsigned short* output,size_t words){
    size_t block=blockIdx.x,first=block*256,pair=threadIdx.x*2;
    size_t blocks=(words+255)/256;const auto* header=blob+block*8;
    unsigned offset=*reinterpret_cast<const unsigned*>(header),base=header[4],mode=header[5];
    unsigned count=unsigned(header[6])|(unsigned(header[7])<<8);
    if(pair>=count||first+pair>=words)return;
    const auto* payload=blob+blocks*8+offset;unsigned short low,high=0;
    if(mode==0){low=loadWord<Streaming>(payload+pair*2);if(pair+1<count)high=loadWord<Streaming>(payload+(pair+1)*2);}
    else {
        const auto* p=payload+(pair/2)*3;
        unsigned packed=unsigned(loadByte<Streaming>(p))|(unsigned(loadByte<Streaming>(p+1))<<8)|(unsigned(loadByte<Streaming>(p+2))<<16);
        low=unpack<Streaming>(packed&4095,base,payload);high=unpack<Streaming>(packed>>12,base,payload);
    }
    if(pair+1<count&&first+pair+1<words)
        reinterpret_cast<unsigned*>(output)[(first+pair)/2]=unsigned(low)|(unsigned(high)<<16);
    else output[first+pair]=low;
}
void decode(Buffer& encoded,Buffer& stage,size_t words,cudaStream_t stream,bool streaming){
    if(streaming)decodeBlocks<true><<<unsigned((words+255)/256),128,0,stream>>>(encoded.data,reinterpret_cast<unsigned short*>(stage.data),words);
    else decodeBlocks<false><<<unsigned((words+255)/256),128,0,stream>>>(encoded.data,reinterpret_cast<unsigned short*>(stage.data),words);
    check(cudaGetLastError());
}
__global__ void initialize(__nv_bfloat16* p,size_t count){
    size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(index>=count)return;
    unsigned h=unsigned(index)*747796405u+2891336453u;h=((h>>((h>>28)+4))^h)*277803737u;h=(h>>22)^h;
    p[index]=__float2bfloat16_rn((float(h%257)-128.f)/1024.f);
}
__global__ void evict(unsigned* p,size_t count,unsigned round){
    size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(index<count)p[index]=p[index]+round+1;
}
struct Captured {
    cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    ~Captured(){if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);}
};
std::unique_ptr<Captured> capture(cudaStream_t stream,const std::function<void()>& body){
    auto result=std::make_unique<Captured>();check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    try{body();}catch(...){cudaStreamEndCapture(stream,&result->graph);throw;}
    check(cudaStreamEndCapture(stream,&result->graph));check(cudaGraphInstantiateWithFlags(&result->executable,result->graph,0));return result;
}
double median(std::vector<double> values){require(!values.empty(),"Missing timing samples");std::sort(values.begin(),values.end());size_t n=values.size();return n%2?values[n/2]:(values[n/2-1]+values[n/2])/2;}
struct Timer {
    cudaEvent_t begin=nullptr,ready=nullptr,end=nullptr;
    Timer(){check(cudaEventCreate(&begin));check(cudaEventCreate(&ready));check(cudaEventCreate(&end));}
    ~Timer(){cudaEventDestroy(begin);cudaEventDestroy(ready);cudaEventDestroy(end);}
    std::array<double,3> run(Captured& graph,cudaStream_t stream,Buffer& eviction,bool cold,unsigned round){
        check(cudaEventRecord(begin,stream));
        if(cold)evict<<<unsigned((eviction.size/4+255)/256),256,0,stream>>>(reinterpret_cast<unsigned*>(eviction.data),eviction.size/4,round);
        check(cudaEventRecord(ready,stream));check(cudaGraphLaunch(graph.executable,stream));
        check(cudaEventRecord(end,stream));check(cudaEventSynchronize(end));
        float evictionMs=0,graphMs=0,totalMs=0;check(cudaEventElapsedTime(&evictionMs,begin,ready));
        check(cudaEventElapsedTime(&graphMs,ready,end));check(cudaEventElapsedTime(&totalMs,begin,end));
        return {graphMs*1000.,evictionMs*1000.,totalMs*1000.};
    }
};
std::uint64_t differentWords(const std::vector<unsigned char>& a,const std::vector<unsigned char>& b){
    require(a.size()==b.size()&&a.size()%2==0,"Word comparison size mismatch");std::uint64_t count=0;
    for(size_t i=0;i<a.size();i+=2)count+=(a[i]!=b[i]||a[i+1]!=b[i+1]);return count;
}
std::vector<unsigned char> download(const Buffer& value){std::vector<unsigned char> result(value.size);check(cudaMemcpy(result.data(),value.data,value.size,cudaMemcpyDeviceToHost));return result;}
