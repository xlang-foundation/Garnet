// SPDX-License-Identifier: Apache-2.0
// Exact FP32 operator diagnostics. All engine variants run in serial; complete
// output files and original FP64 bounds qualify arithmetic before timings.
#include "metal_context.h"
#include "../../src/model/affine_int4_codec.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace Garnet;
namespace {
void Need(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
void Numbers(const std::vector<double>& values){std::cout<<'[';for(size_t i=0;i<values.size();++i){if(i)std::cout<<',';std::cout<<values[i];}std::cout<<']';}
void Cell(MetalContext& gpu,const std::filesystem::path& root,size_t index,size_t rows,size_t n,size_t k,
          const std::vector<size_t>* selected,int repetitions) {
    const auto layout=AffineInt4Layout::Create(n,k,AffineInt4CoefficientType::FP16);
    std::vector<uint32_t> packed(layout.weightBytes/4);
    std::vector<uint16_t> scales(layout.coefficientBytes/2),biases(scales.size());
    for(size_t i=0;i<packed.size();++i)packed[i]=uint32_t(i)*2654435761u+131u;
    for(size_t i=0;i<scales.size();++i){scales[i]=uint16_t(0x2c00+i%8);biases[i]=uint16_t(0xb000+i%7);}
    uint32_t seed=37;std::vector<float> input(rows*k);
    for(auto& value:input){seed=1664525u*seed+1013904223u;value=float(int((seed>>8)%65521)-32760)*0x1p-18f;}
    auto weight=gpu.UploadAffineInt4(packed.data(),layout.weightBytes,scales.data(),layout.coefficientBytes,biases.data(),layout.coefficientBytes,n,k,true);
    auto a=gpu.Upload(input,rows,k),out=gpu.Allocate(rows,n);
    Need(gpu.PrepareAffineInt4Profile(a,weight,out,4),"Cannot prepare bounded projection profile");
    gpu.Update(out,std::vector<float>(rows*n,-99.f));
    std::vector<double> device,wall;
    for(int repeat=-3;repeat<repetitions;++repeat) {
        const auto begin=std::chrono::steady_clock::now();gpu.Begin();
        if(selected)gpu.EncodeAffineInt4IndexedDecode(a,weight,out,*selected);
        else gpu.EncodeAffineInt4MatMul(a,weight,out);
        gpu.SubmitAndWait();const auto end=std::chrono::steady_clock::now();
        Need(gpu.LastAffineProfileMisses()==0,"Unprepared projection dispatch");
        Need(gpu.LastAffineProfileDispatches()==size_t(!selected||!selected->empty()),"Profile dispatch count differs");
        Need(gpu.LastMPPAffineDispatches()==0,"Small-row projection changed arithmetic route");
        if(repeat>=0){device.push_back(gpu.LastGPUMs());wall.push_back(std::chrono::duration<double,std::milli>(end-begin).count());}
    }
    const auto output=gpu.Download(out);
    for(float value:output)Need(std::isfinite(value),"Nonfinite projection output");
    const AffineInt4View view(layout,packed.data(),layout.weightBytes,scales.data(),layout.coefficientBytes,biases.data(),layout.coefficientBytes);
    std::vector<float> decoded(k);size_t checked=0;double maximum=0;
    for(size_t row=0;row<rows;++row) {
        const bool active=!selected||std::find(selected->begin(),selected->end(),row)!=selected->end();
        if(!active){for(size_t c=0;c<n;++c)Need(output[row*n+c]==0.f,"Inactive output row not cleared");continue;}
        for(size_t c:std::array<size_t,3>{0,n/2,n-1}) {
            view.DecodeRow(c,decoded.data(),k);double wanted=0,absolute=0;
            for(size_t j=0;j<k;++j){const double product=double(input[row*k+j])*decoded[j];wanted+=product;absolute+=std::abs(product);}
            const double limit=2e-6+4e-6*absolute,error=std::abs(double(output[row*n+c])-wanted);
            Need(error<=limit,"Projection exceeds original FP64 bound");maximum=std::max(maximum,error/limit);++checked;
        }
    }
    std::ofstream binary(root/("case-"+std::to_string(index)+".bin"),std::ios::binary|std::ios::out);
    binary.write(reinterpret_cast<const char*>(output.data()),std::streamsize(output.size()*sizeof(float)));
    Need(bool(binary),"Cannot preserve complete projection output");binary.close();
    std::cout<<"{\"kind\":\"cell\",\"index\":"<<index<<",\"rows\":"<<rows<<",\"outputs\":"<<n<<",\"inner\":"<<k
             <<",\"indexed\":"<<(selected?"true":"false")<<",\"selected\":[";
    if(selected)for(size_t i=0;i<selected->size();++i){if(i)std::cout<<',';std::cout<<(*selected)[i];}
    std::cout<<"],\"words\":"<<output.size()<<",\"fp64_samples\":"<<checked<<",\"maximum_error_fraction\":"<<maximum
             <<",\"gpu_ms\":";Numbers(device);std::cout<<",\"wall_ms\":";Numbers(wall);std::cout<<"}\n"<<std::flush;
}
}
int main(int argc,char** argv){try{
    Need(argc==3,"Usage: benchmark NEW_OUTPUT_DIRECTORY REPETITIONS");
    const auto root=std::filesystem::path(argv[1]);const int repetitions=std::stoi(argv[2]);
    Need(repetitions>=3&&repetitions<=51,"Invalid repetition count");
    Need(std::filesystem::create_directory(root),"Output directory must be new");size_t index=0;
    {
        MetalContext gpu;std::cout<<std::setprecision(12)<<"{\"kind\":\"metadata\",\"device\":"<<std::quoted(gpu.DeviceName())
            <<",\"warmups\":3,\"repetitions\":"<<repetitions<<",\"maximum_rows\":4,\"profile_is_upper_bound\":true,\"coefficient_dtype\":\"FP16\",\"scope\":\"Synthetic exact FP32 projection diagnostics; not model throughput\"}\n";
        for(const auto& shape:std::vector<std::array<size_t,2>>{{1,64},{7,192},{17,64},{33,128},
            {4096,4096},{1024,4096},{14336,4096},{4096,14336},{32768,4096}}) {
            for(size_t rows:{1,2,3,4})Cell(gpu,root,index++,rows,shape[0],shape[1],nullptr,repetitions);
            for(size_t mask=0;mask<16;++mask) {
                std::vector<size_t> selected;for(size_t row=0;row<4;++row)if(mask&(size_t(1)<<row))selected.push_back(row);
                Cell(gpu,root,index++,4,shape[0],shape[1],&selected,repetitions);
            }
        }
    }
    const auto owners=MetalContext::ProfileOwnerStats();
    Need(!owners.matrices&&!owners.affineWeights&&!owners.bf16Weights&&!owners.pipelineSets,"Native owners remain after close");
    std::cout<<"{\"kind\":\"closed\",\"cells\":"<<index<<",\"owners\":0}\n";return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
