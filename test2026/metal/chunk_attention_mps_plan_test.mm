// SPDX-License-Identifier: Apache-2.0
#include "metal_context.h"
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <thread>
using namespace Garnet;
namespace {
void Need(bool value,const char* why){if(!value)throw std::runtime_error(why);}
struct Job {
    MetalContext gpu;
    std::vector<MetalMatrix> inputs;
    MetalMatrix output;
    std::vector<float> control;
    explicit Job(float phase) {
        constexpr size_t batch=2,chunk=8,heads=4,kv=2,dim=8,capacity=64,rows=batch*chunk;
        auto values=[&](size_t count,float offset){std::vector<float> v(count);for(size_t i=0;i<count;++i)v[i]=.2f*std::sin(i*.17f+phase+offset);return v;};
        std::vector<float> slots(rows),active(rows,1);
        for(size_t b=0;b<batch;++b)for(size_t t=0;t<chunk;++t)slots[b*chunk+t]=float(32+t-(t>3?2:0));
        active[3]=0;
        inputs={gpu.Upload(values(rows*heads*dim,0),rows,heads*dim),gpu.Upload(values(rows*kv*dim,1),rows,kv*dim),
            gpu.Upload(values(rows*kv*dim,2),rows,kv*dim),gpu.Upload(values(batch*kv*capacity*dim,3),batch,kv*capacity*dim),
            gpu.Upload(values(batch*kv*capacity*dim,4),batch,kv*capacity*dim),gpu.Upload(slots,batch,chunk),
            gpu.Upload(active,batch,chunk),gpu.Upload(values(rows*heads*dim,5),rows,heads*dim)};
        output=gpu.Allocate(rows,heads*dim);
        if(!gpu.ChunkAttentionMPSPlanCount()) {
            gpu.Begin();bool rejected=false;
            try{gpu.EncodeAttentionChunk(inputs,output,heads,kv,dim,capacity,chunk,true);}catch(const std::logic_error&){rejected=true;}
            gpu.Abort();Need(rejected,"Lazily prepared MPS scratch during execution");
        }
        gpu.PrepareAttentionChunk(inputs,heads,kv,dim,capacity,chunk);
        gpu.Begin();gpu.EncodeAttentionChunk(inputs,output,heads,kv,dim,capacity,chunk,true,true);gpu.SubmitAndWait();
        control=gpu.Download(output);
    }
    void Run() {
        for(size_t step=0;step<12;++step) {
            @autoreleasepool {
                gpu.Begin();gpu.EncodeAttentionChunk(inputs,output,4,2,8,64,8,true);
                // Shared scratch stays leased through partial submissions.
                gpu.Flush();gpu.SubmitAndWait();
                Need(gpu.LastChunkAttentionMPSDispatches()==1,"MPS plan coverage differs");
                const auto actual=gpu.Download(output);
                for(size_t i=0;i<actual.size();++i)
                    Need(std::isfinite(actual[i])&&std::abs(double(actual[i])-control[i])<=1e-5+2e-4*std::abs(double(control[i])),
                        "Concurrent shared scratch changed a job's output");
            }
        }
    }
};
}
int main(){try{
    for(size_t cycle=0;cycle<2;++cycle) {
        @autoreleasepool {
            Job first(.1f),second(2.4f);
            Need(first.gpu.ChunkAttentionMPSPlanCount()==1&&second.gpu.ChunkAttentionMPSPlanCount()==1,"Identical profiles duplicated MPS scratch");
            const auto bytes=first.gpu.ChunkAttentionMPSScratchBytes();
            Need(bytes>0&&bytes==second.gpu.ChunkAttentionMPSScratchBytes(),"Shared plan byte accounting differs");
            first.gpu.Begin();bool rejected=false;
            try{first.gpu.PrepareAttentionChunk(first.inputs,4,2,8,64,8);}catch(const std::logic_error&){rejected=true;}
            first.gpu.Abort();Need(rejected,"Prepared new scratch during command encoding");
            std::exception_ptr failures[2];
            std::thread a([&]{try{first.Run();}catch(...){first.gpu.Abort();failures[0]=std::current_exception();}});
            std::thread b([&]{try{second.Run();}catch(...){second.gpu.Abort();failures[1]=std::current_exception();}});
            a.join();b.join();for(const auto& failure:failures)if(failure)std::rethrow_exception(failure);
            Need(first.gpu.ChunkAttentionMPSPlanCount()==1&&first.gpu.ChunkAttentionMPSScratchBytes()==bytes,"Execution changed the prepared MPS profile");
        }
        const auto owners=MetalContext::ProfileOwnerStats();
        Need(!owners.matrices&&!owners.affineWeights&&!owners.bf16Weights&&!owners.pipelineSets,"Shared MPS plan retained Garnet resource owners");
        std::cout<<"MPS shared plan cycle="<<cycle<<" two contexts,24 partial-submission runs, all owners closed; device bytes="<<MetalContext::MemoryStats().allocated<<"\n";
    }
    return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<"\n";return 1;}}
