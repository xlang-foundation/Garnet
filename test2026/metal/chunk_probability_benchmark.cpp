// SPDX-License-Identifier: Apache-2.0
// Same-context A/B diagnostic. Does not measure full-model serving throughput.
#include "metal_context.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
using namespace Garnet;
namespace {
void Need(bool value,const char* reason){if(!value)throw std::runtime_error(reason);}
void Array(const std::vector<double>& values){std::cout<<"[";for(size_t i=0;i<values.size();++i){if(i)std::cout<<",";std::cout<<values[i];}std::cout<<"]";}
void Cell(MetalContext& gpu,size_t activeRows,size_t activeTokens,size_t prefix,int warmups,int repetitions) {
    constexpr size_t batch=2,chunk=128,heads=24,kvHeads=4,dim=256,capacity=4096;
    constexpr size_t rows=batch*chunk,qWidth=heads*dim,kvWidth=kvHeads*dim;
    std::vector<float> q(rows*qWidth),k(rows*kvWidth),v(k.size()),kc(batch*kvHeads*capacity*dim),vc(kc.size()),gate(q.size()),slots(rows),active(rows);
    auto fill=[](std::vector<float>& values,double phase){for(size_t i=0;i<values.size();++i)values[i]=float(std::sin(i*.131+phase)*.2);};
    fill(q,.1);fill(k,.7);fill(v,1.1);fill(kc,2.1);fill(vc,3.2);fill(gate,4.3);
    for(size_t b=0;b<batch;++b)for(size_t t=0;t<chunk;++t){slots[b*chunk+t]=float(prefix+t);active[b*chunk+t]=float(b<activeRows&&t<activeTokens);}
    std::vector<MetalMatrix> inputs={gpu.Upload(q,rows,qWidth),gpu.Upload(k,rows,kvWidth),gpu.Upload(v,rows,kvWidth),
        gpu.Upload(kc,batch,kvHeads*capacity*dim),gpu.Upload(vc,batch,kvHeads*capacity*dim),
        gpu.Upload(slots,batch,chunk),gpu.Upload(active,batch,chunk),gpu.Upload(gate,rows,qWidth)};
    std::array<MetalMatrix,2> outputs{gpu.Allocate(rows,qWidth),gpu.Allocate(rows,qWidth)};
    std::array<std::vector<double>,2> gpuMs,wallMs;
    for(int repeat=-warmups;repeat<repetitions;++repeat)for(int step=0;step<2;++step) {
        const int variant=(repeat+warmups+step)%2;
        const auto began=std::chrono::steady_clock::now();
        gpu.Begin();gpu.EncodeAttentionChunk(inputs,outputs[variant],heads,kvHeads,dim,capacity,chunk,true,variant==0);gpu.SubmitAndWait();
        const auto ended=std::chrono::steady_clock::now();
        Need(gpu.LastChunkAttentionOptimizedDispatches()==size_t(variant==1),"Probability policy route differs");
        if(repeat>=0){gpuMs[variant].push_back(gpu.LastGPUMs());wallMs[variant].push_back(std::chrono::duration<double,std::milli>(ended-began).count());}
    }
    const auto control=gpu.Download(outputs[0]),candidate=gpu.Download(outputs[1]);
    Need(control.size()==candidate.size(),"Probability policy changed output shape");
    const bool bitwise=std::memcmp(control.data(),candidate.data(),control.size()*sizeof(float))==0;
    double maximumDelta=0;
    for(size_t i=0;i<candidate.size();++i) {
        const double delta=std::abs(double(candidate[i])-control[i]);maximumDelta=std::max(maximumDelta,delta);
        Need(std::isfinite(control[i])&&std::isfinite(candidate[i])&&delta<=1e-5+2e-4*std::abs(double(control[i])),"Probability policy exceeds the original control/FP64 error bound");
    }
    for(size_t b=0;b<batch;++b)for(size_t t=0;t<chunk;++t)if(!active[b*chunk+t])
        for(size_t d=0;d<qWidth;++d)Need(candidate[(b*chunk+t)*qWidth+d]==0,"Inactive query retained stale output");
    std::cout<<"{\"kind\":\"cell\",\"active_rows\":"<<activeRows<<",\"active_tokens_per_row\":"<<activeTokens<<",\"prefix\":"<<prefix
        <<",\"warmups\":"<<warmups<<",\"repetitions\":"<<repetitions<<",\"bitwise_equal\":"<<(bitwise?"true":"false")<<",\"maximum_control_delta\":"<<maximumDelta<<",\"control_gpu_ms\":";Array(gpuMs[0]);
    std::cout<<",\"candidate_gpu_ms\":";Array(gpuMs[1]);std::cout<<",\"control_wall_ms\":";Array(wallMs[0]);std::cout<<",\"candidate_wall_ms\":";Array(wallMs[1]);std::cout<<"}\n"<<std::flush;
}
}
int main(int argc,char** argv) {
    try {
        const int repetitions=argc>1?std::stoi(argv[1]):9,warmups=argc>2?std::stoi(argv[2]):3;
        Need(repetitions>=3&&repetitions<=100&&warmups>=1&&warmups<=20,"Invalid sample policy");
        MetalContext gpu;
#ifdef GARNET_METAL_CHUNK_ATTENTION_INDEX32
        const char* policy="index32";
        const char* rounding="32-bit bounded element indexes; FP32 operands and arithmetic order retained; bit parity measured";
#else
        const char* policy="probability";
        const char* rounding="Default fast math; shared normalized probabilities may change rounding; bit parity measured";
#endif
        std::cout<<std::setprecision(12)<<"{\"kind\":\"metadata\",\"device\":"<<std::quoted(gpu.DeviceName())
            <<",\"candidate_policy\":"<<std::quoted(policy)<<",\"physical_batch\":2,\"chunk\":128,\"capacity\":4096,\"heads\":24,\"kv_heads\":4,\"dim\":256,\"index_bytes\":"<<kMetalAttentionIndexBytes
            <<",\"rounding_scope\":"<<std::quoted(rounding)<<",\"dtype\":\"float32\",\"gated\":true,\"order\":\"alternating within each cell\",\"scope\":\"Isolated read-only attention control/candidate comparison; excludes load, compile and upload; not model throughput\"}\n";
        for(size_t activeRows:{1,2})for(size_t prefix:{0,128,1024,1920,3968})Cell(gpu,activeRows,128,prefix,warmups,repetitions);
        Cell(gpu,1,4,256,warmups,repetitions);Cell(gpu,1,6,2048,warmups,repetitions);
        return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<"\n";return 1;}
}
