// SPDX-License-Identifier: Apache-2.0
#include "../../src/backends/metal/metal_context.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
using namespace Garnet;
namespace {
void Need(bool value,const char* message){if(!value)throw std::runtime_error(message);}
void Array(const std::vector<double>& xs){std::cout<<'[';for(size_t i=0;i<xs.size();++i){if(i)std::cout<<',';std::cout<<xs[i];}std::cout<<']';}
void Cell(MetalContext& gpu,size_t index,size_t batch,size_t heads,size_t dim,size_t capacity,size_t chunk,
          unsigned scenario,const std::filesystem::path& output,int repetitions) {
    const size_t width=heads*capacity*dim,tokenWidth=heads*dim;
    std::vector<float> cache(batch*width),tokens(batch*chunk*tokenWidth),slots(batch*chunk),active(slots.size());
    for(size_t i=0;i<cache.size();++i)cache[i]=float(int((i*131u+71u)%65521)-32760)*0x1p-16f;
    for(size_t i=0;i<tokens.size();++i)tokens[i]=float(int((i*37u+97u)%65521)-32760)*0x1p-15f;
    const bool inplace=scenario!=7,contiguous=scenario!=6&&scenario!=8;
    for(size_t row=0;row<batch;++row)for(size_t t=0;t<chunk;++t) {
        const bool owner=scenario==1?row==0:scenario==2?row==batch-1:scenario==3?row%2==0:scenario==5?false:true;
        bool live=owner&&(scenario!=4||t+1<chunk);
        if(scenario==6)live=owner&&t%3!=1;
        active[row*chunk+t]=float(live);
        slots[row*chunk+t]=float(capacity-chunk+t);
        if(scenario==6)slots[row*chunk+t]=float(t%std::min(size_t(5),capacity)); // Overwrites plus gaps.
    }
    auto expected=cache;
    for(size_t row=0;row<batch;++row)for(size_t t=0;t<chunk;++t)if(active[row*chunk+t])
        for(size_t head=0;head<heads;++head)for(size_t d=0;d<dim;++d)
            expected[row*width+(head*capacity+size_t(slots[row*chunk+t]))*dim+d]=tokens[(row*chunk+t)*tokenWidth+head*dim+d];
    auto input=gpu.Upload(tokens,batch*chunk,tokenWidth),old=gpu.Upload(cache,batch,width);
    auto position=gpu.Upload(slots,batch,chunk),mask=gpu.Upload(active,batch,chunk);
    auto next=inplace?old:gpu.Upload(std::vector<float>(batch*width,-99.f),batch,width);
    std::vector<double> device,wall;size_t scatter=0;
    const int warmups=repetitions>1?3:0;
    for(int repeat=-warmups;repeat<repetitions;++repeat) {
        const auto start=std::chrono::steady_clock::now();gpu.Begin();
        gpu.EncodeKVChunk(input,old,position,mask,next,heads,dim,capacity,chunk,contiguous);
        gpu.SubmitAndWait();const auto finish=std::chrono::steady_clock::now();
#ifdef GARNET_METAL_CHUNK_KV_SCATTER
        scatter=gpu.LastChunkKVScatterDispatches();
        Need(scatter==size_t(inplace&&contiguous&&dim%4==0&&capacity<=kMetalAttentionChunkCapacityLimit),"Unexpected scatter/fallback route");
#endif
        if(repeat>=0){device.push_back(gpu.LastGPUMs());wall.push_back(std::chrono::duration<double,std::milli>(finish-start).count());}
    }
    auto actual=gpu.Download(next);Need(actual.size()==expected.size()&&!std::memcmp(actual.data(),expected.data(),actual.size()*4),"Complete private KV cache differs from CPU reference");
    if(!inplace){auto original=gpu.Download(old);Need(!std::memcmp(original.data(),cache.data(),cache.size()*4),"Separate input cache was modified");}
    if(!output.empty()) {
        const auto path=output/("case-"+std::to_string(index)+".f32");std::ofstream file(path,std::ios::binary);
        file.write(reinterpret_cast<const char*>(actual.data()),std::streamsize(actual.size()*4));file.close();Need(bool(file),"Cannot retain complete KV output");
    }
    std::cout<<"{\"kind\":\"cell\",\"index\":"<<index<<",\"batch\":"<<batch<<",\"heads\":"<<heads<<",\"dim\":"<<dim<<",\"capacity\":"<<capacity<<",\"chunk\":"<<chunk
        <<",\"scenario\":"<<scenario<<",\"inplace\":"<<(inplace?"true":"false")<<",\"contiguous\":"<<(contiguous?"true":"false")<<",\"scatter_dispatches\":"<<scatter
        <<",\"full_cpu_reference_words\":"<<actual.size()<<",\"warmups\":"<<warmups<<",\"gpu_ms\":";Array(device);std::cout<<",\"wall_ms\":";Array(wall);std::cout<<"}\n"<<std::flush;
}
}
int main(int argc,char** argv){try{
    const bool benchmark=argc==3&&std::string(argv[1])=="--benchmark";Need(argc==1||benchmark,"Usage: test [--benchmark NEW_DIRECTORY]");
    std::filesystem::path output;if(benchmark){output=argv[2];Need(std::filesystem::create_directory(output),"Benchmark output must be new");}
    size_t index=0;
    {
        MetalContext gpu;bool enabled=false;
#ifdef GARNET_METAL_CHUNK_KV_SCATTER
        enabled=true;
#endif
        std::cout<<std::setprecision(12)<<"{\"kind\":\"metadata\",\"device\":"<<std::quoted(gpu.DeviceName())<<",\"scatter_compiled\":"<<(enabled?"true":"false")
            <<",\"scope\":\"Complete KV cache CPU parity and sequential operator latency; not model throughput\"}\n";
        if(benchmark)for(unsigned scenario=0;scenario<9;++scenario)Cell(gpu,index++,4,8,128,4096,128,scenario,output,9);
        else {
            for(unsigned scenario=0;scenario<9;++scenario) {
                Cell(gpu,index++,1,1,4,16,4,scenario,{},1);
                Cell(gpu,index++,4,2,8,32,8,scenario,{},1);
                Cell(gpu,index++,2,2,3,16,4,scenario,{},1); // Scalar fallback.
            }
            Cell(gpu,index++,1,1,4,kMetalAttentionChunkCapacityLimit+16,4,0,{},1); // Above-profile cache fallback.
        }
    }
    auto owners=MetalContext::ProfileOwnerStats();Need(!owners.matrices&&!owners.affineWeights&&!owners.bf16Weights&&!owners.pipelineSets,"KV test retained native owners");
    std::cout<<"{\"kind\":\"closed\",\"cases\":"<<index<<",\"owners\":0}\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
