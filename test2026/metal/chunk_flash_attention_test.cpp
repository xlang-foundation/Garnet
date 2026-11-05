// SPDX-License-Identifier: Apache-2.0
// Actual native FP32 tiled attention, independently checked in FP64.
#include "metal_context.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace Garnet;
using JSON=nlohmann::json;
namespace {
void Need(bool v,const char* why){if(!v)throw std::runtime_error(why);}
struct Fixture {
    MetalContext gpu;size_t batch,chunk,heads,kv,dim,capacity;
    std::vector<float> q,k,v,kc,vc,gate,slots,active;
    std::vector<MetalMatrix> inputs;std::array<MetalMatrix,2> output;MetalMatrix pending,zeros;
    Fixture(size_t b,size_t t,size_t h,size_t kvh,size_t d,size_t c,bool prepare=true):batch(b),chunk(t),heads(h),kv(kvh),dim(d),capacity(c) {
        auto values=[](size_t n,double phase,double amplitude){std::vector<float> a(n);for(size_t i=0;i<n;++i)a[i]=float(amplitude*std::sin(i*.313+phase)+.13*std::cos(i*.077+phase));return a;};
        q=values(batch*chunk*heads*dim,.1,1.3);k=values(batch*chunk*kv*dim,.7,1.2);v=values(k.size(),1.1,.7);
        kc=values(batch*kv*capacity*dim,2.3,1.1);vc=values(kc.size(),3.1,.8);gate=values(q.size(),4.7,1.6);
        slots.resize(batch*chunk);active.resize(slots.size());
        inputs={gpu.Upload(q,batch*chunk,heads*dim),gpu.Upload(k,batch*chunk,kv*dim),gpu.Upload(v,batch*chunk,kv*dim),
            gpu.Upload(kc,batch,kv*capacity*dim),gpu.Upload(vc,batch,kv*capacity*dim),gpu.Upload(slots,batch,chunk),gpu.Upload(active,batch,chunk),gpu.Upload(gate,batch*chunk,heads*dim)};
        output={gpu.Allocate(batch*chunk,heads*dim),gpu.Allocate(batch*chunk,heads*dim)};
        pending=gpu.Allocate(batch,chunk);zeros=gpu.Upload(std::vector<float>(slots.size(),0),batch,chunk);
        if(prepare)gpu.PrepareAttentionChunk(inputs,heads,kv,dim,capacity,chunk);
    }
    JSON Cell(const char* label,std::array<size_t,4> counts,size_t prefix,bool gated,int mutation,bool benchmark) {
        size_t live=0;for(size_t b=0;b<batch;++b)for(size_t t=0;t<chunk;++t){active[b*chunk+t]=float(t<counts[b]);slots[b*chunk+t]=t<counts[b]?float(prefix+t):0;live+=t<counts[b];}
        if(mutation==1){active[3]=0;--live;} // gap
        if(mutation==2)slots[8]=slots[6]; // overwrite
        if(mutation==3)std::reverse(slots.begin(),slots.begin()+chunk); // noncontiguous
        gpu.Update(inputs[5],slots);gpu.Update(inputs[6],active);
        std::array<std::vector<double>,2> times,wall;size_t flash=0;
        const auto before=MetalContext::ProfileOwnerStats();const auto plans=gpu.ChunkAttentionMPSPlanCount(),scratch=gpu.ChunkAttentionMPSScratchBytes();
        for(int repeat=benchmark?-3:0;repeat<(benchmark?9:1);++repeat)for(int step=0;step<2;++step) {
            const int variant=(step+repeat+3)%2;auto args=inputs;
            auto began=std::chrono::steady_clock::now();gpu.Begin();
            if(mutation==4){gpu.EncodeAdd(inputs[5],zeros,pending);args[5]=pending;}
            gpu.EncodeAttentionChunk(args,output[variant],heads,kv,dim,capacity,chunk,gated,false,variant==0);
            gpu.SubmitAndWait();double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-began).count();
            const bool eligible=variant==1&&dim==128&&chunk>=8&&chunk%8==0&&live>8&&mutation==0;
            Need(gpu.LastChunkFlashAttentionDispatches()==size_t(eligible),"Unexpected flash/fallback selection");
            if(variant==1)flash=gpu.LastChunkFlashAttentionDispatches();
            if(repeat>=0){times[variant].push_back(gpu.LastGPUMs());wall[variant].push_back(elapsed);}
        }
        const auto after=MetalContext::ProfileOwnerStats();
        Need(before.matrices==after.matrices&&before.affineWeights==after.affineWeights&&before.pipelineSets==after.pipelineSets&&
            gpu.ChunkAttentionMPSPlanCount()==plans&&gpu.ChunkAttentionMPSScratchBytes()==scratch,"Execution changed prepared allocation/owners");
        auto control=gpu.Download(output[0]),actual=gpu.Download(output[1]);size_t samples=0,inactive=0;double maximumError=0,maximumDifference=0;
        for(size_t i=0;i<actual.size();++i){Need(std::isfinite(actual[i]),"Nonfinite flash output");maximumDifference=std::max(maximumDifference,std::abs(double(actual[i])-control[i]));}
        for(size_t b=0;b<batch;++b)for(size_t t=0;t<chunk;++t)for(size_t h=0;h<heads;++h) {
            size_t flat=b*chunk+t;
            if(!active[flat]){for(size_t d=0;d<dim;++d){Need(actual[(flat*heads+h)*dim+d]==0,"Inactive row retained stale output");++inactive;}continue;}
            if(benchmark&&t!=0&&t!=chunk/2&&t!=chunk-1)continue;
            if(benchmark&&h!=0&&h!=heads-1)continue;
            size_t visible=size_t(slots[flat])+1,kh=h/(heads/kv);std::vector<double> scores(visible);std::vector<int> selected(visible,-1);double maximum=-INFINITY;
            for(size_t s=0;s<visible;++s) {
                for(size_t j=0;j<=t;++j)if(active[b*chunk+j]&&slots[b*chunk+j]==float(s))selected[s]=int(j);
                double sum=0;for(size_t d=0;d<dim;++d){float key=selected[s]<0?kc[((b*kv+kh)*capacity+s)*dim+d]:k[((b*chunk+size_t(selected[s]))*kv+kh)*dim+d];sum+=double(q[(flat*heads+h)*dim+d])*key;}
                scores[s]=sum/std::sqrt(double(dim));maximum=std::max(maximum,scores[s]);
            }
            double denominator=0;for(auto& score:scores){score=std::exp(score-maximum);denominator+=score;}
            for(size_t d=0;d<dim;++d) {
                if(benchmark&&d!=0&&d!=dim-1)continue;
                double expected=0;for(size_t s=0;s<visible;++s){float value=selected[s]<0?vc[((b*kv+kh)*capacity+s)*dim+d]:v[((b*chunk+size_t(selected[s]))*kv+kh)*dim+d];expected+=scores[s]/denominator*value;}
                size_t i=(flat*heads+h)*dim+d;if(gated)expected/=1+std::exp(-double(gate[i]));
                double error=std::abs(double(actual[i])-expected),limit=1e-5+2e-4*std::abs(expected);
                Need(error<=limit,"Flash attention exceeds independent FP64 oracle");maximumError=std::max(maximumError,error/limit);++samples;
            }
        }
        Need(gpu.Download(inputs[3])==kc&&gpu.Download(inputs[4])==vc,"Flash attention modified input KV");
        return JSON{{"label",label},{"batch",batch},{"chunk",chunk},{"heads",heads},{"kv_heads",kv},{"dim",dim},{"capacity",capacity},
            {"prefix",prefix},{"counts",counts},{"gated",gated},{"mutation",mutation},{"flash_dispatches",flash},{"full_output_words",actual.size()},
            {"fp64_samples",samples},{"fp64_maximum_error_fraction",maximumError},{"maximum_control_difference",maximumDifference},{"inactive_zero_words",inactive},
            {"mps_gpu_ms",times[0]},{"flash_gpu_ms",times[1]},{"mps_wall_ms",wall[0]},{"flash_wall_ms",wall[1]},
            {"warmups",benchmark?3:0},{"repetitions",benchmark?9:1},{"prepared_plans",plans},{"prepared_scratch_bytes",scratch}};
    }
};
}
int main(int argc,char** argv){try{
    bool benchmark=argc==3&&std::string(argv[1])=="--benchmark";Need(argc==1||benchmark,"Usage: flash probe [--benchmark NEW_DIRECTORY]");
    JSON report={{"scope","FP32 tiled online-softmax operator; synthetic finite operands; not model token speed"},{"cells",JSON::array()},{"rejections",JSON::array()}};
    std::filesystem::path folder;if(benchmark){folder=argv[2];Need(std::filesystem::create_directory(folder),"New directory required");}
    {
        if(benchmark) {
            for(size_t b:{size_t(1),size_t(2),size_t(4)}){Fixture f(b,128,32,8,128,4096);for(size_t prefix:{size_t(0),size_t(1920),size_t(3968)})
                report["cells"].push_back(f.Cell("full-profile",{128,128,128,128},prefix,false,0,true));}
        }else {
            {
                Fixture guard(1,24,4,2,128,128,false);
                for(size_t t=0;t<24;++t){guard.slots[t]=float(t);guard.active[t]=1;}
                guard.gpu.Update(guard.inputs[5],guard.slots);guard.gpu.Update(guard.inputs[6],guard.active);
                auto reject=[&](const char* label,const char* reason,auto function){bool failed=false;guard.gpu.Begin();try{function();}catch(const std::exception& e){failed=std::string(e.what()).find(reason)!=std::string::npos;}guard.gpu.Abort();Need(failed,label);report["rejections"].push_back(label);};
                auto run=[&]{guard.gpu.EncodeAttentionChunk(guard.inputs,guard.output[0],4,2,128,128,24,false);};
                reject("unprepared-profile","Flash attention fallback profile",run);
                guard.gpu.PrepareAttentionChunk(guard.inputs,4,2,128,128,24);
                reject("output-alias","output alias",[&]{guard.gpu.EncodeAttentionChunk(guard.inputs,guard.inputs[0],4,2,128,128,24,false);});
                auto original=guard.inputs[0];guard.inputs[0]=guard.gpu.Allocate(24,4*128);
                reject("uninitialized-query","Uninitialized MPS attention input",run);guard.inputs[0]=original;
                guard.gpu.Begin();run();bool pending=false;
                try{guard.gpu.Download(guard.output[0]);}catch(const std::logic_error&){pending=true;}
                Need(pending,"Pending flash output was readable");guard.gpu.Abort();
                bool invalid=false;try{guard.gpu.Download(guard.output[0]);}catch(const std::logic_error&){invalid=true;}
                Need(invalid&&guard.gpu.LastChunkFlashAttentionDispatches()==0,"Abort retained valid flash output/counters");
                guard.gpu.Begin();run();guard.gpu.SubmitAndWait();
                Need(guard.gpu.LastChunkFlashAttentionDispatches()==1,"Full flash write did not recover aborted output");
                report["rejections"].push_back("pending-read");report["rejections"].push_back("aborted-read");
                report["cells"].push_back(guard.Cell("abort-recovery",{24,0,0,0},0,false,0,false));
            }
            Fixture a(1,16,4,2,128,128);report["cells"].push_back(a.Cell("initial",{16,0,0,0},0,false,0,false));
            report["cells"].push_back(a.Cell("cached-gated",{16,0,0,0},112,true,0,false));
            for(int m=1;m<=4;++m)report["cells"].push_back(a.Cell("fallback",{16,0,0,0},80,m%2!=0,m,false));
            report["cells"].push_back(a.Cell("small",{5,0,0,0},80,false,0,false));
            report["cells"].push_back(a.Cell("paused",{0,0,0,0},0,true,0,false));
            Fixture b(2,32,4,2,128,128);report["cells"].push_back(b.Cell("last-owner",{0,32,0,0},96,false,0,false));
            report["cells"].push_back(b.Cell("ragged-prefix",{11,23,0,0},64,true,0,false));
            Fixture c(4,64,8,2,128,256);report["cells"].push_back(c.Cell("mixed-owners",{0,45,64,7},128,false,0,false));
            Fixture d(2,8,4,2,128,128);report["cells"].push_back(d.Cell("short-query-tile",{8,8,0,0},120,true,0,false));
            Fixture e(1,16,4,2,64,128);report["cells"].push_back(e.Cell("dimension-fallback",{16,0,0,0},112,false,0,false));
            // The large-profile independent oracle samples queries/dimensions;
            // every output word is still checked finite/inactive-zero above.
            Fixture full(4,128,32,8,128,4096);report["cells"].push_back(full.Cell("full-profile",{128,0,127,33},3968,false,0,true));
        }
    }
    auto owners=MetalContext::ProfileOwnerStats();Need(!owners.matrices&&!owners.affineWeights&&!owners.bf16Weights&&!owners.pipelineSets,"Flash probe retained native owners");
    report["native_owners_after_close"]=0;
    if(benchmark){std::ofstream out(folder/"report.json");out<<report.dump(2)<<'\n';Need(bool(out),"Cannot save flash report");}
    std::cout<<report.dump()<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
