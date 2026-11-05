// SPDX-License-Identifier: Apache-2.0
// Real Mistral projection dimensions, synthetic FP16 coefficients/inputs.
// Operator diagnostic only: not model quality, TTFT or serving throughput.
#include "../../src/backends/metal/metal_context.h"
#include "../../src/model/affine_int4_codec.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <stdexcept>
using namespace Garnet;
namespace {
void Require(bool value,const char* message) {if(!value)throw std::runtime_error(message);}
float Half(float value) {return Float16ToFloat32(Float32ToFloat16RNE(value));}
void Array(const std::vector<double>& values) {
    std::cout<<'[';for(size_t i=0;i<values.size();++i){if(i)std::cout<<',';std::cout<<values[i];}std::cout<<']';
}
void Cell(MetalContext& gpu,size_t logical,size_t n,size_t k,size_t maximumRows,
          bool indexed,int repetitions,int warmups,const std::filesystem::path& outputDirectory) {
    const size_t physical=indexed?maximumRows:logical;
    const auto layout=AffineInt4Layout::Create(n,k,AffineInt4CoefficientType::FP16);
    std::vector<uint32_t> packed(layout.weightBytes/4);
    std::vector<uint16_t> scales(layout.coefficientBytes/2),biases(scales.size());
    for(size_t i=0;i<packed.size();++i)packed[i]=uint32_t(i*2654435761u)^0x76543210u;
    for(size_t i=0;i<scales.size();++i) {
        scales[i]=Float32ToFloat16RNE(float(int(i%13)-6)*.00171f);
        biases[i]=Float32ToFloat16RNE(float(int(i%17)-8)*.00313f);
    }
    auto weight=gpu.UploadAffineInt4(packed.data(),layout.weightBytes,scales.data(),layout.coefficientBytes,
                                    biases.data(),layout.coefficientBytes,n,k,true);
    std::vector<float> input(physical*k);
    for(size_t i=0;i<input.size();++i)input[i]=float(std::sin(double(i)*.173)*.05);
    std::vector<size_t> selected(logical);std::vector<bool> active(physical);
    for(size_t i=0;i<logical;++i) {selected[i]=indexed?(maximumRows-1+37*i)%maximumRows:i;active[selected[i]]=true;}
    auto a=gpu.Upload(input,physical,k),out=gpu.Upload(std::vector<float>(physical*n,17.f),physical,n);
    Require(gpu.PrepareAffineInt4Profile(a,weight,out,maximumRows),"Cannot prepare FP16 projection profile");
    const size_t pipelineCount=gpu.AffineProfilePipelineCount();
    std::vector<double> gpuMs,wallMs;size_t reuseDispatches=0;
    for(int repetition=-warmups;repetition<repetitions;++repetition) {
        const auto start=std::chrono::steady_clock::now();gpu.Begin();
        if(indexed)gpu.EncodeAffineInt4IndexedMatMul(a,weight,out,selected);
        else gpu.EncodeAffineInt4MatMul(a,weight,out);
        gpu.SubmitAndWait();const auto finish=std::chrono::steady_clock::now();
        Require(gpu.AffineProfilePipelineCount()==pipelineCount,"Runtime compiled an FP16 projection pipeline");
        Require(gpu.LastAffineProfileDispatches()==1&&gpu.LastAffineProfileMisses()==0,"Prepared FP16 projection profile missed");
        Require(gpu.LastMPPAffineDispatches()==size_t(indexed||logical>4),"FP16 projection tensor route mismatch");
#ifdef GARNET_METAL_MPP_AFFINE_REUSE64
        reuseDispatches=gpu.LastReuse64AffineDispatches();
        Require(reuseDispatches==size_t(logical>=64&&n>=GARNET_METAL_REUSE64_MIN_OUTPUTS),"FP16 cooperative64 route mismatch");
#endif
        if(repetition>=0) {
            gpuMs.push_back(gpu.LastGPUMs());
            wallMs.push_back(std::chrono::duration<double,std::milli>(finish-start).count());
        }
    }
    const auto actual=gpu.Download(out);size_t inactiveWords=0;
    for(size_t row=0;row<physical;++row)for(size_t col=0;col<n;++col) {
        Require(std::isfinite(actual[row*n+col]),"Nonfinite FP16 projection output");
        if(!active[row]) {Require(actual[row*n+col]==0.f,"Inactive FP16 projection output retained stale data");++inactiveWords;}
    }
    const AffineInt4View view(layout,packed.data(),layout.weightBytes,scales.data(),layout.coefficientBytes,
                             biases.data(),layout.coefficientBytes);
    std::set<size_t> rows{0,logical/2,logical-1},columns{0,1,15,16,31,32,63,64,n/2,n-1};
    for(size_t row:{size_t(15),size_t(16),size_t(31),size_t(32),size_t(63),size_t(64),size_t(127)})if(row<logical)rows.insert(row);
    std::vector<float> decoded(k);size_t samples=0;double maximum=0,maximumRatio=0;
    for(auto col:columns) {
        view.DecodeRow(col,decoded.data(),k);
        for(auto logicalRow:rows) {
            const auto row=selected[logicalRow];double expected=0,absolute=0;
            for(size_t inner=0;inner<k;++inner) {
                float left=input[row*k+inner],right=decoded[inner];
                if(indexed||logical>4){left=Half(left);right=Half(right);}
                const double product=double(left)*right;expected+=product;absolute+=std::abs(product);
            }
            const double error=std::abs(double(actual[row*n+col])-expected),limit=2e-6+4e-6*absolute;
            Require(error<=limit,"Real-shape FP16 projection sample differs from independent FP64 oracle");
            maximum=std::max(maximum,error);maximumRatio=std::max(maximumRatio,error/limit);++samples;
        }
    }
    const auto outputPath=outputDirectory/(std::string(indexed?"indexed":"dense")+"-m"+std::to_string(logical)+
        "-n"+std::to_string(n)+"-k"+std::to_string(k)+".f32");
    std::ofstream stream(outputPath,std::ios::binary);Require(bool(stream),"Cannot open projection word output");
    stream.write(reinterpret_cast<const char*>(actual.data()),std::streamsize(actual.size()*sizeof(float)));
    stream.close();Require(bool(stream),"Cannot write projection word output");
    std::cout<<"{\"kind\":\"cell\",\"mode\":"<<std::quoted(indexed?"indexed":"dense")
        <<",\"logical_rows\":"<<logical<<",\"physical_rows\":"<<physical<<",\"outputs\":"<<n<<",\"inner\":"<<k
        <<",\"repetitions\":"<<repetitions<<",\"warmups\":"<<warmups<<",\"reuse64_dispatches\":"<<reuseDispatches
        <<",\"profile_misses\":0,\"runtime_pipeline_compiles\":0,\"fp64_samples\":"<<samples
        <<",\"fp64_max_abs_error\":"<<maximum<<",\"fp64_max_error_fraction\":"<<maximumRatio
        <<",\"inactive_zero_words\":"<<inactiveWords<<",\"output_words\":"<<actual.size()
        <<",\"output_file\":"<<std::quoted(outputPath.string())<<",\"gpu_ms\":";Array(gpuMs);
    std::cout<<",\"wall_ms\":";Array(wallMs);std::cout<<"}\n"<<std::flush;
}
}
int main(int argc,char** argv) {
    try {
        Require(argc==6,"Usage: benchmark repetitions warmups maximumRows dense|indexed outputDirectory");
        const int repetitions=std::stoi(argv[1]),warmups=std::stoi(argv[2]),maximumRows=std::stoi(argv[3]);
        const std::string mode=argv[4];const std::filesystem::path outputDirectory=argv[5];
        Require(repetitions>=3&&repetitions<=100&&warmups>=1&&warmups<=20,"Invalid benchmark repetitions/warmups");
        Require(maximumRows==128||maximumRows==512,"Invalid FP16 benchmark row bound");
        Require(mode=="dense"||mode=="indexed","Invalid FP16 projection mode");
        Require(std::filesystem::is_directory(outputDirectory)&&std::filesystem::is_empty(outputDirectory),"Output directory must exist and be empty");
        {
            MetalContext gpu;Require(gpu.HasMPPAffine()&&gpu.HasIndexedAffine(),"Actual FP16 tensor/indexed pipelines required");
            int minimumOutputs=0;bool relaxed=false;
#ifdef GARNET_METAL_MPP_AFFINE_REUSE64
            Require(gpu.HasReuse64Affine(),"Actual FP16 cooperative64 pipeline required");
            minimumOutputs=GARNET_METAL_REUSE64_MIN_OUTPUTS;relaxed=GARNET_METAL_REUSE64_RELAXED_VALUE!=0;
#endif
            std::cout<<std::setprecision(12)<<"{\"kind\":\"metadata\",\"device\":"<<std::quoted(gpu.DeviceName())
                <<",\"operand_dtype\":\"float16\",\"coefficient_dtype\":\"float16\",\"accumulator_dtype\":\"float32\""
                <<",\"reuse64_min_outputs\":"<<minimumOutputs<<",\"reuse64_relaxed\":"<<(relaxed?"true":"false")
                <<",\"maximum_rows\":"<<maximumRows<<",\"mode\":"<<std::quoted(mode)
                <<",\"scope\":\"Synthetic Mistral projection operator latency; excludes compile/load/download/oracle; no model speed or quality claim\"}\n";
            for(size_t rows:{size_t(4),size_t(16),size_t(32),size_t(63),size_t(64),size_t(65),size_t(127),size_t(128),size_t(256),size_t(512)})
                if(rows<=size_t(maximumRows))for(auto shape:std::array<std::array<size_t,2>,4>{{{1024,4096},{4096,4096},{14336,4096},{4096,14336}}})
                    Cell(gpu,rows,shape[0],shape[1],size_t(maximumRows),mode=="indexed",repetitions,warmups,outputDirectory);
        }
        const auto owners=MetalContext::ProfileOwnerStats();
        Require(!owners.matrices&&!owners.affineWeights&&!owners.bf16Weights&&!owners.pipelineSets,"Projection benchmark leaked owners");
        std::cout<<"{\"kind\":\"complete\",\"owners\":0}\n";return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
