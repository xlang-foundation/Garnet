// SPDX-License-Identifier: Apache-2.0
// Isolated correctness only. This is neither a model nor a timing benchmark.
#include "qwen38_bf16_prefill128_commit.cuh"
#include "prefill128_commit_fixture.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
using namespace Garnet;
namespace {
constexpr uint32_t guard=0x5a65a65a,initial=0x3f003f00;
constexpr size_t cw=106823680,tw=44564480;
void require(bool v,const char* message){if(!v)throw std::runtime_error(message);}
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Device{void* p=nullptr;explicit Device(size_t n){check(cudaMalloc(&p,n));}~Device(){if(p)cudaFree(p);}Device(const Device&)=delete;};
struct Stream{cudaStream_t p=nullptr;Stream(){check(cudaStreamCreateWithFlags(&p,cudaStreamNonBlocking));}~Stream(){if(p)cudaStreamDestroy(p);}};
uint32_t word(size_t i){
    const uint32_t ties[]={0x3f808000u,0x3f818000u,0xbf808000u,0xbf818000u,0x00008000u,0x80008000u,0,0x80000000u};
    const uint32_t dimension=uint32_t(i)&31;if(dimension<8)return ties[dimension];
    uint32_t x=uint32_t(i>>8)*1664525u+dimension*1013904223u+17u;
    x^=x>>13;x*=0x85ebca6bu;x^=x>>16;return 0x3f000000u|(x&0x807fffffu);
}
uint16_t rounded(uint32_t x){return uint16_t((x+0x7fffu+((x>>16)&1))>>16);}
void encode(const std::filesystem::path& path,const std::vector<uint32_t>& data){
    require(data.size()%32==0,"Codec alignment");std::ofstream file(path,std::ios::binary);require(bool(file),"Output file unavailable");
    const uint64_t words=data.size();file.write("GNP128R1",8);file.write(reinterpret_cast<const char*>(&words),8);
    for(size_t i=0;i<data.size();){uint32_t count=1;const auto* block=data.data()+i;
        while(i+size_t(count)*32<data.size()&&count!=UINT32_MAX&&std::memcmp(block,data.data()+i+size_t(count)*32,128)==0)++count;
        file.write(reinterpret_cast<const char*>(&count),4);file.write(reinterpret_cast<const char*>(block),128);i+=size_t(count)*32;
    }require(bool(file),"Full-state codec write failed");
}
void oracle(std::vector<uint32_t>& state,const std::vector<uint32_t>& target,int start,int count){
    std::fill(state.begin(),state.end(),initial);std::fill_n(state.begin(),32,guard);std::fill_n(state.end()-32,32,guard);
    if(!count)return;
    for(const auto& row:prefill128Fixture){
        if(!row.full){std::copy_n(target.begin()+32+row.target,row.first+row.second,state.begin()+32+row.canonical);continue;}
        for(int which=0;which<2;++which)for(int t=0;t<count;++t)for(int head=0;head<4;++head)for(int d=0;d<256;++d){
            const size_t destination=2*(32+row.canonical+which*row.first)+(size_t(head)*4096+start+t)*256+d;
            const size_t source=32+row.target+which*131072+(size_t(t)*4+head)*256+d;
            const auto v=rounded(target[source]);auto& packed=state[destination/2];const int shift=int(destination%2)*16;
            packed=(packed&~(0xffffu<<shift))|(uint32_t(v)<<shift);
        }
    }
}
}
int main(int argc,char** argv){
    try{
        require(argc==2,"Pass existing owned output folder");const std::filesystem::path out=argv[1];require(std::filesystem::is_directory(out),"Output directory required");
        require(!std::filesystem::exists(out/"cases.jsonl"),"Never overwrite output evidence");
        int device=0;check(cudaGetDevice(&device));cudaDeviceProp props{};check(cudaGetDeviceProperties(&props,device));
        require(props.major==12&&props.minor==0&&std::string(props.name)=="NVIDIA GeForce RTX 5090","Explicit RTX5090 SM120 required");
        const uint32_t little=1;require(*reinterpret_cast<const unsigned char*>(&little)==1,"Little endian codec required");
        Stream stream;Device t((tw+64)*4),c((cw+64)*4);
        std::vector<uint32_t> target(tw+64,guard),expected(cw+64),actual(cw+64),readonly(tw+64);
        for(size_t i=0;i<tw;++i)target[i+32]=word(i);
        check(cudaMemcpyAsync(t.p,target.data(),target.size()*4,cudaMemcpyHostToDevice,stream.p));check(cudaStreamSynchronize(stream.p));
        encode(out/"target-input.rle",target);std::ofstream records(out/"cases.jsonl",std::ios::binary);int id=0,valid=0,invalid=0,graphs=0;
        cudaGraph_t sharedPlan=nullptr;cudaGraphExec_t sharedExec=nullptr;
        auto run=[&](int start,int count,bool graph,int batch,int chunk,int defect){
            Qwen38BF16KVSpans spans{};spans.start[0]=start;spans.count[0]=count;
            const bool good=defect==0;
            oracle(expected,target,0,0);check(cudaMemcpyAsync(c.p,expected.data(),expected.size()*4,cudaMemcpyHostToDevice,stream.p));check(cudaStreamSynchronize(stream.p));
            const float* input=static_cast<float*>(t.p)+32;float* destination=static_cast<float*>(c.p)+32;
            if(defect==1)input=nullptr;if(defect==2)destination=nullptr;
            if(defect==3)input=reinterpret_cast<const float*>(reinterpret_cast<uintptr_t>(input)+1);
            if(defect==4)destination=reinterpret_cast<float*>(reinterpret_cast<uintptr_t>(destination)+1);
            if(defect==5)destination=const_cast<float*>(input);
            if(defect==6)input=reinterpret_cast<const float*>(UINTPTR_MAX-3);
            if(defect>=10&&defect<17)spans.start[defect-9]=1;
            if(defect>=20&&defect<27)spans.count[defect-19]=1;
            int code=0;size_t nodes=0;
            if(!graph||!sharedExec){
                if(graph)check(cudaStreamBeginCapture(stream.p,cudaStreamCaptureModeThreadLocal));
                code=RunQwen38BF16Prefill128Commit(input,destination,batch,chunk,spans,stream.p);
                if(graph){check(cudaStreamEndCapture(stream.p,&sharedPlan));check(cudaGraphInstantiate(&sharedExec,sharedPlan,nullptr,nullptr,0));}
            }
            if(graph){check(cudaGraphGetNodes(sharedPlan,nullptr,&nodes));require(nodes==2,"Exactly two commit kernels required");check(cudaGraphLaunch(sharedExec,stream.p));++graphs;}
            check(cudaStreamSynchronize(stream.p));check(cudaGetLastError());
            check(cudaMemcpyAsync(actual.data(),c.p,actual.size()*4,cudaMemcpyDeviceToHost,stream.p));
            check(cudaMemcpyAsync(readonly.data(),t.p,readonly.size()*4,cudaMemcpyDeviceToHost,stream.p));check(cudaStreamSynchronize(stream.p));
            const auto file="case-"+std::to_string(id)+".rle";encode(out/file,actual);
            if(good)oracle(expected,target,start,count);
            const bool matches=actual==expected,unchanged=readonly==target;
            records<<"{\"id\":"<<id<<",\"start\":"<<start<<",\"count\":"<<count<<",\"batch\":"<<batch<<",\"chunk\":"<<chunk<<",\"defect\":"<<defect
                <<",\"graph\":"<<(graph?"true":"false")<<",\"graph_nodes\":"<<nodes<<",\"return_code\":"<<code<<",\"canonical_file\":\""<<file
                <<"\",\"full_state_words_compared\":"<<actual.size()<<",\"target_words_compared\":"<<readonly.size()<<",\"full_state_exact\":"<<(matches?"true":"false")
                <<",\"target_readonly\":"<<(unchanged?"true":"false")<<"}\n";records.flush();
            require(bool(records),"Case ledger write failed");++id;require(matches,"Full-state CPU oracle mismatch");require(unchanged,"Target input or guards changed");
            require(good?code==0:code!=0,"Wrong commit admission");if(good)++valid;else ++invalid;
        };
        const int spans[][2]={{0,0},{0,1},{0,127},{0,128},{1,128},{511,128},{512,128},{513,128},{2047,2},{3968,128},{3969,127},{4095,1},{4096,0}};
        for(auto& span:spans)run(span[0],span[1],false,1,128,0);
        for(int i=0;i<3;++i)run(511,128,true,1,128,0);
        check(cudaGraphExecDestroy(sharedExec));check(cudaGraphDestroy(sharedPlan));sharedExec=nullptr;sharedPlan=nullptr;
        for(int defect:{1,2,3,4,5,6,10,11,12,13,14,15,16,20,21,22,23,24,25,26})run(0,128,false,1,128,defect);
        for(int batch:{0,2,4,8})run(0,128,false,batch,128,7);
        for(int chunk:{0,4,32,64,256})run(0,128,false,1,chunk,8);
        for(auto& span:std::vector<std::pair<int,int>>{{-1,1},{4097,0},{0,-1},{0,129},{3969,128},{4096,1}})run(span.first,span.second,false,1,128,9);
        encode(out/"target-final.rle",readonly);
        std::cout<<"{\"status\":\"prefill128_commit_full_state_collected\",\"valid_cases\":"<<valid<<",\"invalid_cases\":"<<invalid<<",\"graph_replays\":"<<graphs
            <<",\"performance_admitted\":false,\"runtime_qualified\":false}\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
