// SPDX-License-Identifier: Apache-2.0
// Native64/128-row comparison, complete word parity and independent FP64 samples.
#include "../../src/backends/metal/metal_context.h"
#include "../../src/backends/metal/affine_reuse128_policy.h"
#include "../../src/model/affine_int4_codec.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
using namespace Garnet;using JSON=nlohmann::json;
namespace {
void Need(bool v,const char* why){if(!v)throw std::runtime_error(why);}
float Half(float x){return Float16ToFloat32(Float32ToFloat16RNE(x));}
struct Fixture {
    MetalContext gpu;size_t physical,n,k;AffineInt4Layout layout;
    std::vector<uint32_t> packed;std::vector<uint16_t> scales,biases;std::vector<float> input;
    MetalAffineInt4Weight weight;MetalMatrix a;std::array<MetalMatrix,2> output;
    Fixture(size_t rows,size_t columns,size_t inner,bool prepare=true):physical(rows),n(columns),k(inner),layout(AffineInt4Layout::Create(n,k,AffineInt4CoefficientType::FP16)) {
        packed.resize(layout.weightBytes/4);scales.resize(layout.coefficientBytes/2);biases.resize(scales.size());input.resize(physical*k);
        for(size_t i=0;i<packed.size();++i)packed[i]=uint32_t(i*2654435761u)^0x76543210u;
        for(size_t i=0;i<scales.size();++i){scales[i]=Float32ToFloat16RNE(float(int(i%13)-6)*.00171f);biases[i]=Float32ToFloat16RNE(float(int(i%17)-8)*.00313f);}
        for(size_t i=0;i<input.size();++i)input[i]=float(std::sin(double(i)*.173)*.05);
        weight=gpu.UploadAffineInt4(packed.data(),layout.weightBytes,scales.data(),layout.coefficientBytes,biases.data(),layout.coefficientBytes,n,k,true);
        a=gpu.Upload(input,physical,k);for(auto& out:output)out=gpu.Upload(std::vector<float>(physical*n,11.f),physical,n);
        if(prepare)Need(gpu.PrepareAffineInt4Profile(a,weight,output[0],physical),"Cannot prepare128-row affine profiles");
    }
    JSON Cell(size_t logical,bool benchmark) {
        std::vector<size_t> indices(logical);std::vector<bool> active(physical);
        const size_t multiplier=physical==511?137:127;
        for(size_t i=0;i<logical;++i){indices[i]=(i*multiplier+5)%physical;Need(!active[indices[i]],"Fixture row permutation duplicates");active[indices[i]]=true;}
        const bool eligible=UseAffineReuse128(logical,n,GARNET_METAL_REUSE64_MIN_OUTPUTS),family=logical>=64&&n>=GARNET_METAL_REUSE64_MIN_OUTPUTS;
        auto before=MetalContext::ProfileOwnerStats();const auto pipelines=gpu.AffineProfilePipelineCount();std::array<std::vector<double>,2> times,wall;
        for(int repeat=benchmark?-3:0;repeat<(benchmark?9:1);++repeat)for(int step=0;step<2;++step) {
            const int variant=(repeat+3+step)%2;auto start=std::chrono::steady_clock::now();gpu.Begin();
            if(variant)gpu.EncodeAffineInt4IndexedMatMul(a,weight,output[variant],indices);
            else gpu.EncodeAffineInt4IndexedMatMulControl(a,weight,output[variant],indices);
            gpu.SubmitAndWait();const auto elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            Need(gpu.LastReuse128AffineDispatches()==size_t(variant&&eligible),"Actual128-row route differs");
            Need(gpu.LastReuse64AffineDispatches()==size_t(family&&!(variant&&eligible)),"Actual64-row fallback differs");
            Need(gpu.LastIndexedAffineDispatches()==size_t(logical>0)&&gpu.LastMPPAffineDispatches()==size_t(logical>0),"Indexed/tensor coverage differs");
            Need(gpu.LastAffineProfileDispatches()==size_t(logical>0)&&gpu.LastAffineProfileMisses()==0&&gpu.AffineProfilePipelineCount()==pipelines,"Runtime profile miss/compilation");
            if(repeat>=0){times[variant].push_back(gpu.LastGPUMs());wall[variant].push_back(elapsed);}
        }
        auto control=gpu.Download(output[0]),candidate=gpu.Download(output[1]);Need(control.size()==candidate.size()&&std::memcmp(control.data(),candidate.data(),control.size()*4)==0,"Complete64/128 output words differ");
        size_t inactive=0;for(size_t i=0;i<candidate.size();++i){Need(std::isfinite(candidate[i]),"Nonfinite128-row projection");if(!active[i/n]){Need(candidate[i]==0,"Inactive physical projection retained scratch");++inactive;}}
        size_t samples=0;double maximum=0;
        if(logical) {
            AffineInt4View view(layout,packed.data(),layout.weightBytes,scales.data(),layout.coefficientBytes,biases.data(),layout.coefficientBytes);
            std::set<size_t> rows{0,logical/2,logical-1},columns{0,1,15,16,31,32,63,64,n/2,n-1};
            for(size_t r:{size_t(15),size_t(16),size_t(31),size_t(32),size_t(63),size_t(64),size_t(127),size_t(128),size_t(255)})if(r<logical)rows.insert(r);
            std::vector<float> decoded(k);
            for(auto col:columns){view.DecodeRow(col,decoded.data(),k);for(auto row:rows){double expected=0,absolute=0;for(size_t inner=0;inner<k;++inner){const double product=double(Half(input[indices[row]*k+inner]))*Half(decoded[inner]);expected+=product;absolute+=std::abs(product);}const double fraction=std::abs(double(candidate[indices[row]*n+col])-expected)/(2e-6+4e-6*absolute);Need(fraction<=1,"Projection exceeds independent FP64 bound");maximum=std::max(maximum,fraction);++samples;}}
        }
        const auto after=MetalContext::ProfileOwnerStats();Need(before.matrices==after.matrices&&before.affineWeights==after.affineWeights&&before.pipelineSets==after.pipelineSets,"Projection allocated new native owners");
        return JSON{{"logical_rows",logical},{"physical_rows",physical},{"outputs",n},{"inner",k},{"candidate128",eligible},{"output_words",candidate.size()},{"inactive_zero_words",inactive},{"exact_control_words",true},{"fp64_samples",samples},{"fp64_maximum_error_fraction",maximum},{"control_gpu_ms",times[0]},{"candidate_gpu_ms",times[1]},{"control_wall_ms",wall[0]},{"candidate_wall_ms",wall[1]},{"warmups",benchmark?3:0},{"repetitions",benchmark?9:1},{"profile_pipelines",pipelines},{"profile_misses",0}};
    }
};
JSON Guards() {
    Fixture f(512,8192,192,false);std::vector<size_t> rows(128);for(size_t i=0;i<rows.size();++i)rows[i]=i*3;
    auto before=f.gpu.Download(f.output[1]);bool failed=false;f.gpu.Begin();
    try{f.gpu.EncodeAffineInt4IndexedMatMul(f.a,f.weight,f.output[1],rows);}catch(const std::logic_error& e){failed=std::string(e.what()).find("Prepare the cooperative128")!=std::string::npos;}
    f.gpu.Abort();Need(failed&&f.gpu.Download(f.output[1])==before,"Unprepared128-row call encoded a write");
    Need(f.gpu.PrepareAffineInt4Profile(f.a,f.weight,f.output[1],512),"Guard profile preparation failed");JSON rejected=JSON::array({"unprepared-before-zero"});
    auto reject=[&](const char* label,auto call){auto saved=f.gpu.Download(f.output[1]);bool bad=false;f.gpu.Begin();try{call();}catch(const std::exception&){bad=true;}f.gpu.Abort();Need(bad&&f.gpu.Download(f.output[1])==saved,"Invalid projection changed output");rejected.push_back(label);};
    auto duplicate=rows;duplicate[1]=duplicate[0];reject("duplicate-index",[&]{f.gpu.EncodeAffineInt4IndexedMatMul(f.a,f.weight,f.output[1],duplicate);});
    auto outside=rows;outside[1]=512;reject("out-of-range-index",[&]{f.gpu.EncodeAffineInt4IndexedMatMul(f.a,f.weight,f.output[1],outside);});
    auto unready=f.gpu.Allocate(512,192);reject("uninitialized-input",[&]{f.gpu.EncodeAffineInt4IndexedMatMul(unready,f.weight,f.output[1],rows);});
    f.gpu.Begin();f.gpu.EncodeAffineInt4IndexedMatMul(f.a,f.weight,f.output[1],rows);f.gpu.Abort();
    failed=false;try{f.gpu.Download(f.output[1]);}catch(const std::logic_error&){failed=true;}Need(failed,"Aborted128-row output remained readable");
    f.gpu.Begin();f.gpu.EncodeAffineInt4IndexedMatMul(f.a,f.weight,f.output[1],rows);f.gpu.SubmitAndWait();Need(f.gpu.LastReuse128AffineDispatches()==1,"128-row recovery failed");rejected.push_back("abort-and-recovery");
#ifdef GARNET_METAL_VERIFY_ATTENTION
    const auto pipelines=f.gpu.AffineProfilePipelineCount();
    auto run=[&](bool enabled) {
        f.gpu.VerificationSetAffineReuse128(enabled);f.gpu.Begin();
        f.gpu.EncodeAffineInt4IndexedMatMul(f.a,f.weight,f.output[1],rows);f.gpu.SubmitAndWait();
        Need(f.gpu.LastReuse128AffineDispatches()==size_t(enabled)&&f.gpu.LastReuse64AffineDispatches()==size_t(!enabled),"Verification route selection differs");
        Need(f.gpu.AffineProfilePipelineCount()==pipelines&&f.gpu.LastAffineProfileMisses()==0,"Verification selection compiled or missed");
        return f.gpu.Download(f.output[1]);
    };
    auto candidate=run(true);Need(run(false)==candidate&&run(true)==candidate,"Verification selection changes complete output");
    f.gpu.VerificationSetAffineReuse128(false);f.gpu.Begin();bool blocked=false;
    try{f.gpu.VerificationSetAffineReuse128(true);}catch(const std::logic_error&){blocked=true;}
    Need(blocked,"Verification route changed in an open command batch");
    f.gpu.EncodeAffineInt4IndexedMatMul(f.a,f.weight,f.output[1],rows);f.gpu.Flush();blocked=false;
    try{f.gpu.VerificationSetAffineReuse128(true);}catch(const std::logic_error&){blocked=true;}
    Need(blocked,"Verification route changed across partial submission");f.gpu.SubmitAndWait();
    Need(f.gpu.LastReuse64AffineDispatches()==1&&f.gpu.LastReuse128AffineDispatches()==0,"Rejected selection mutated current policy");
    {Fixture other(128,8192,192);other.Cell(128,false);}
    f.gpu.Begin();f.gpu.EncodeAffineInt4IndexedMatMul(f.a,f.weight,f.output[1],rows);f.gpu.SubmitAndWait();
    Need(f.gpu.LastReuse64AffineDispatches()==1&&f.gpu.LastReuse128AffineDispatches()==0&&f.gpu.Download(f.output[1])==candidate,"Independent context changed verification policy");
    f.gpu.Begin();f.gpu.Abort();Need(run(true)==candidate,"Verification selection failed after abort");
    rejected.push_back("verification-route-parity");rejected.push_back("verification-open-batch-rejected");
    rejected.push_back("verification-partial-batch-rejected");rejected.push_back("verification-context-isolation");
#endif
    return rejected;
}
}
int main(int argc,char** argv){try{
    const bool benchmark=argc==3&&std::string(argv[1])=="--benchmark";Need(argc==1||benchmark,"Usage: reuse128 [--benchmark NEW_DIRECTORY]");
    JSON report={{"scope","Native strict FP16 projection operator64/128 tile experiment; synthetic weights/inputs, not model speed or quality"},{"cases",JSON::array()},{"rejections",JSON::array()}};
    if(benchmark) {
        Need(std::filesystem::create_directory(argv[2]),"New benchmark directory required");
        for(auto shape:std::array<std::array<size_t,2>,4>{{{1024,4096},{4096,4096},{14336,4096},{4096,14336}}}){Fixture f(512,shape[0],shape[1]);for(size_t m:{size_t(64),size_t(84),size_t(96),size_t(128),size_t(160),size_t(192),size_t(256),size_t(512)}){report["cases"].push_back(f.Cell(m,true));std::cout<<"reuse128_benchmark "<<m<<" "<<shape[0]<<" "<<shape[1]<<" PASS\n"<<std::flush;}}
    }else {
        report["rejections"]=Guards();
        {Fixture f(512,8192,64);for(size_t m:{size_t(0),size_t(4),size_t(63),size_t(64),size_t(84),size_t(96),size_t(128),size_t(129),size_t(160),size_t(192),size_t(193),size_t(255),size_t(256),size_t(384),size_t(512)})report["cases"].push_back(f.Cell(m,false));}
        {Fixture f(511,8193,128);report["cases"].push_back(f.Cell(128,false));}
        {Fixture f(512,1024,128);report["cases"].push_back(f.Cell(256,false));}
    }
    const auto owners=MetalContext::ProfileOwnerStats();Need(!owners.matrices&&!owners.affineWeights&&!owners.bf16Weights&&!owners.pipelineSets,"Projection test retained owners");report["passed"]=true;
    if(benchmark){std::ofstream out(std::filesystem::path(argv[2])/"results.json");out<<report.dump(2)<<'\n';Need(bool(out),"Cannot write benchmark results");}
    else std::cout<<report.dump()<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
