// SPDX-License-Identifier: Apache-2.0
// Fixed-layout request schedules. Optional serial operator timing, not token/s.
#include "metal_context.h"
#include "nlohmann/json.hpp"
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace Garnet;
using JSON=nlohmann::json;
namespace {
void Need(bool value,const char* why){if(!value)throw std::runtime_error(why);}
void Save(const std::filesystem::path& path,const std::vector<float>& v){std::ofstream s(path,std::ios::binary);s.write(reinterpret_cast<const char*>(v.data()),std::streamsize(v.size()*4));Need(bool(s),"Cannot save attention output");}
struct Fixture {
    MetalContext gpu;
    size_t batch=4,chunk,heads,kv=2,dim,capacity;
    bool ragged;
    std::vector<float> q,k,v,kc,vc,gate,slots,flags;
    std::vector<MetalMatrix> inputs;
    std::array<MetalMatrix,2> outputs;
    MetalMatrix copiedSlots,zeroSlots;
    Fixture(bool benchmark):chunk(benchmark?128:16),heads(benchmark?32:4),kv(benchmark?8:2),
        dim(benchmark?128:8),capacity(benchmark?4096:128),ragged(!benchmark) {
        size_t rows=batch*chunk;
        auto values=[](size_t n,double phase){std::vector<float> a(n);for(size_t i=0;i<n;++i)a[i]=float(.2*std::sin(i*.131+phase));return a;};
        q=values(rows*heads*dim,.1);k=values(rows*kv*dim,.7);v=values(k.size(),1.1);
        kc=values(batch*kv*capacity*dim,2.1);vc=values(kc.size(),3.2);gate=values(q.size(),4.3);
        slots.resize(rows);flags.resize(rows);
        inputs={gpu.Upload(q,rows,heads*dim),gpu.Upload(k,rows,kv*dim),gpu.Upload(v,rows,kv*dim),
            gpu.Upload(kc,batch,kv*capacity*dim),gpu.Upload(vc,batch,kv*capacity*dim),
            gpu.Upload(slots,batch,chunk),gpu.Upload(flags,batch,chunk),gpu.Upload(gate,rows,heads*dim)};
        outputs={gpu.Allocate(rows,heads*dim),gpu.Allocate(rows,heads*dim)};
        copiedSlots=gpu.Allocate(batch,chunk);zeroSlots=gpu.Upload(std::vector<float>(rows,0),batch,chunk);
        gpu.PrepareAttentionChunk(inputs,heads,kv,dim,capacity,chunk);
    }
    double Oracle(const std::vector<float>& got,size_t& samples,bool gated) {
        double maxFraction=0;
        for(size_t b=0;b<batch;++b)for(size_t t:{size_t(0),chunk/2,chunk-1})
            for(size_t h:{size_t(0),heads-1}) {
                const size_t flat=b*chunk+t;
                if(!flags[flat])continue;
                const size_t visible=size_t(slots[flat])+1,kvh=h/(heads/kv);
                std::vector<double> scores(visible);std::vector<int> selected(visible,-1);
                double maximum=-INFINITY;
                for(size_t s=0;s<visible;++s) {
                    for(size_t j=0;j<=t;++j)if(flags[b*chunk+j]&&slots[b*chunk+j]==float(s))selected[s]=int(j);
                    double sum=0;
                    for(size_t d=0;d<dim;++d) {
                        float key=selected[s]<0?kc[((b*kv+kvh)*capacity+s)*dim+d]:k[((b*chunk+size_t(selected[s]))*kv+kvh)*dim+d];
                        sum+=double(q[(flat*heads+h)*dim+d])*key;
                    }
                    scores[s]=sum/std::sqrt(double(dim));maximum=std::max(maximum,scores[s]);
                }
                double denominator=0;for(auto& score:scores){score=std::exp(score-maximum);denominator+=score;}
                for(size_t d:{size_t(0),dim-1}) {
                    double expected=0;
                    for(size_t s=0;s<visible;++s) {
                        float value=selected[s]<0?vc[((b*kv+kvh)*capacity+s)*dim+d]:v[((b*chunk+size_t(selected[s]))*kv+kvh)*dim+d];
                        expected+=scores[s]/denominator*value;
                    }
                    size_t index=(flat*heads+h)*dim+d;
                    if(gated)expected/=1+std::exp(-double(gate[index]));
                    double error=std::abs(double(got[index])-expected),limit=1e-5+2e-4*std::abs(expected);
                    Need(error<=limit,"Attention request schedule exceeds independent FP64 oracle");
                    maxFraction=std::max(maxFraction,error/limit);++samples;
                }
            }
        return maxFraction;
    }
    JSON Cell(size_t mask,size_t prefix,bool unknown,bool gated,int warmups,int repetitions,const std::filesystem::path& folder) {
        size_t owners=0;
        for(size_t b=0;b<batch;++b) {
            bool live=(mask&(size_t(1)<<b))!=0;owners+=live;
            for(size_t t=0;t<chunk;++t) {
                slots[b*chunk+t]=float(prefix+t-(ragged&&t>chunk/2?2:0));
                flags[b*chunk+t]=float(live&&(!ragged||t!=3));
            }
        }
        gpu.Update(inputs[5],slots);gpu.Update(inputs[6],flags);
        size_t plans=gpu.ChunkAttentionMPSPlanCount(),scratch=gpu.ChunkAttentionMPSScratchBytes();
        std::array<std::vector<double>,2> gpuMs,hostMs;std::array<size_t,2> matrices{},skipped{};
        const bool scalar=owners*(chunk-size_t(ragged))<=8;
        for(int repeat=-warmups;repeat<repetitions;++repeat)for(int step=0;step<2;++step) {
            const int variant=(repeat+warmups+step)%2;
            auto began=std::chrono::steady_clock::now();gpu.Begin();auto args=inputs;
            if(unknown){gpu.EncodeAdd(inputs[5],zeroSlots,copiedSlots);args[5]=copiedSlots;}
            gpu.EncodeAttentionChunk(args,outputs[variant],heads,kv,dim,capacity,chunk,gated,false,variant==0);
            gpu.SubmitAndWait();double host=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-began).count();
            const size_t expected=(unknown||variant==0)?batch:owners;
            const size_t expectedMatrices=(!unknown&&scalar)?0:expected*kv*4;
            const size_t expectedSkipped=(!unknown&&scalar)?0:(batch-expected)*kv*4;
            Need(gpu.LastChunkAttentionMPSMatrices()==expectedMatrices&&gpu.LastChunkAttentionMPSSkippedMatrices()==expectedSkipped,
                "Request schedule matrix accounting differs");
            Need(gpu.LastChunkAttentionUnknownControls()==size_t(unknown),"GPU-produced controls selected a host-only schedule");
            Need(gpu.ChunkAttentionMPSPlanCount()==plans&&gpu.ChunkAttentionMPSScratchBytes()==scratch,"Request execution changed prepared storage");
            if(repeat>=0){gpuMs[variant].push_back(gpu.LastGPUMs());hostMs[variant].push_back(host);}
            matrices[variant]=gpu.LastChunkAttentionMPSMatrices();skipped[variant]=gpu.LastChunkAttentionMPSSkippedMatrices();
        }
        auto full=gpu.Download(outputs[0]),selected=gpu.Download(outputs[1]);
        Need(full.size()==selected.size()&&std::memcmp(full.data(),selected.data(),full.size()*4)==0,
            "Prepared request products changed any public output bit");
        size_t inactive=0;
        for(size_t row=0;row<batch*chunk;++row)for(size_t d=0;d<heads*dim;++d) {
            Need(std::isfinite(selected[row*heads*dim+d]),"Nonfinite attention output");
            if(!flags[row]){Need(selected[row*heads*dim+d]==0,"Inactive owner/token read stale scratch");++inactive;}
        }
        size_t samples=0;double fraction=Oracle(selected,samples,gated);
        if(!folder.empty()) {
            auto label="mask"+std::to_string(mask)+"-p"+std::to_string(prefix)+"-u"+std::to_string(unknown);
            Save(folder/(label+"-full.f32"),full);Save(folder/(label+"-selected.f32"),selected);
        }
        return JSON{{"mask",mask},{"prefix",prefix},{"unknown_controls",unknown},{"gated",gated},
            {"ragged_controls",ragged},
            {"warmups",warmups},{"repetitions",repetitions},{"output_words",selected.size()},{"inactive_zero_words",inactive},
            {"bitwise_equal",true},{"fp64_samples",samples},{"fp64_maximum_error_fraction",fraction},
            {"full_matrices",matrices[0]},{"selected_matrices",matrices[1]},{"skipped_matrices",skipped[1]},
            {"full_gpu_ms",gpuMs[0]},{"selected_gpu_ms",gpuMs[1]},{"full_host_ms",hostMs[0]},{"selected_host_ms",hostMs[1]},
            {"plans",plans},{"scratch_bytes",scratch}};
    }
};
}
int main(int argc,char** argv){try{
    const bool benchmark=argc==3&&std::string(argv[1])=="--benchmark";
    Need(argc==1||benchmark,"Usage: probe [--benchmark NEW_OUTPUT_DIRECTORY]");
    std::filesystem::path folder;if(benchmark){folder=argv[2];Need(std::filesystem::create_directory(folder),"New output directory required");}
    JSON report={{"scope","Native fixed-layout attention schedules only; synthetic inputs, not model token speed or serving"},
        {"physical_batch",4},{"chunk",benchmark?128:16},{"heads",benchmark?32:4},{"kv_heads",benchmark?8:2},
        {"dimension",benchmark?128:8},{"capacity",benchmark?4096:128},{"cells",JSON::array()}};
    {
        Fixture fixture(benchmark);report["device"]=fixture.gpu.DeviceName();
        if(benchmark) {
            for(size_t mask:{size_t(1),size_t(8),size_t(5),size_t(3),size_t(15)})
                for(size_t prefix:{size_t(0),size_t(3968)})report["cells"].push_back(fixture.Cell(mask,prefix,false,false,3,9,folder));
        }else {
            // Traverse every owner subset twice, through stale populated scratch.
            for(size_t cycle=0;cycle<2;++cycle)for(size_t m=0;m<16;++m) {
                size_t mask=cycle?15-m:m;
                report["cells"].push_back(fixture.Cell(mask,cycle?96:0,false,cycle!=0,1,3,{}));
            }
            report["cells"].push_back(fixture.Cell(1,96,true,true,1,3,{}));
            report["cells"].push_back(fixture.Cell(0,0,true,false,1,3,{}));
            Need(fixture.gpu.Download(fixture.inputs[3])==fixture.kc&&fixture.gpu.Download(fixture.inputs[4])==fixture.vc,
                "Read-only request schedule changed KV ownership/storage");
        }
    }
    const auto owners=MetalContext::ProfileOwnerStats();
    Need(!owners.matrices&&!owners.affineWeights&&!owners.bf16Weights&&!owners.pipelineSets,"Request schedules leaked native owners");
    report["native_owners_after_close"]=0;
    if(benchmark){std::ofstream stream(folder/"report.json");stream<<report.dump(2)<<'\n';Need(bool(stream),"Cannot save request schedule report");}
    std::cout<<report.dump()<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
