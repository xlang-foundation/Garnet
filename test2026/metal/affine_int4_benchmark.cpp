// SPDX-License-Identifier: Apache-2.0
// Synthetic operator diagnostics, never end-to-end model throughput evidence.
#include "metal_context.h"
#include "../../src/model/affine_int4_codec.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace Garnet;
namespace {
uint16_t BF16(float value) {
    uint32_t bits;std::memcpy(&bits,&value,4);return uint16_t((bits+0x7fff+((bits>>16)&1))>>16);
}
void Require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
void Array(const std::vector<double>& values) {
    std::cout<<"[";for(size_t i=0;i<values.size();++i){if(i)std::cout<<",";std::cout<<values[i];}std::cout<<"]";
}
void Cell(MetalContext& gpu,size_t m,size_t n,size_t k,int warmups,int repetitions) {
    const auto layout=AffineInt4Layout::Create(n,k);
    std::vector<uint32_t> packed(layout.weightBytes/4);
    std::vector<uint16_t> scales(layout.coefficientBytes/2),biases(scales.size());
    for(size_t i=0;i<packed.size();++i)packed[i]=uint32_t(i*2654435761u)^0x76543210u;
    for(size_t i=0;i<scales.size();++i){scales[i]=BF16(float(int(i%13)-6)*.017f);biases[i]=BF16(float(int(i%17)-8)*.031f);}
    std::vector<float> input(m*k);
    for(size_t i=0;i<input.size();++i)input[i]=float(std::sin(double(i)*.173)*.05);
    auto weight=gpu.UploadAffineInt4(packed.data(),layout.weightBytes,scales.data(),layout.coefficientBytes,biases.data(),layout.coefficientBytes,n,k);
    auto a=gpu.Upload(input,m,k),out=gpu.Allocate(m,n);
    std::vector<double> gpuMs,wallMs;size_t mppDispatches=0,reuse64Dispatches=0;
    for(int repeat=-warmups;repeat<repetitions;++repeat) {
        const auto start=std::chrono::steady_clock::now();
        gpu.Begin();gpu.EncodeAffineInt4MatMul(a,weight,out);gpu.SubmitAndWait();
        const auto finish=std::chrono::steady_clock::now();
#ifdef GARNET_METAL_MPP_AFFINE
        mppDispatches=gpu.LastMPPAffineDispatches();
        Require(mppDispatches==size_t(m>4&&gpu.HasMPPAffine()),"Operator MPP route/counter mismatch");
#endif
#ifdef GARNET_METAL_MPP_AFFINE_REUSE64
        reuse64Dispatches=gpu.LastReuse64AffineDispatches();
        Require(reuse64Dispatches==size_t(m>=64&&n>=5120&&gpu.HasReuse64Affine()),"Operator reuse64 route/counter mismatch");
#endif
        if(repeat>=0){gpuMs.push_back(gpu.LastGPUMs());wallMs.push_back(std::chrono::duration<double,std::milli>(finish-start).count());}
    }
    const auto actual=gpu.Download(out);
    for(float value:actual)Require(std::isfinite(value),"Nonfinite benchmark output");
    AffineInt4View view(layout,packed.data(),layout.weightBytes,scales.data(),layout.coefficientBytes,biases.data(),layout.coefficientBytes);
    std::vector<float> row(k);double maximum=0;
    for(auto sample:std::array<std::array<size_t,2>,3>{{{0,0},{m/2,n/2},{m-1,n-1}}}) {
        view.DecodeRow(sample[1],row.data(),k);double expected=0;
        for(size_t inner=0;inner<k;++inner) {
            float left=input[sample[0]*k+inner],right=row[inner];
#ifdef GARNET_METAL_MPP_AFFINE_BF16
            if(m>4&&gpu.HasMPPAffine()){left=AffineInt4RoundBF16(left);right=AffineInt4RoundBF16(right);}
#endif
            expected+=double(left)*right;
        }
        const auto error=std::abs(double(actual[sample[0]*n+sample[1]])-expected);maximum=std::max(maximum,error);
        Require(error<=std::max(1e-4,std::abs(expected)*3e-4),"Large-shape FP64 sample differs");
    }
    bool relaxed=false,nMajor=false,registers=false;
#ifdef GARNET_METAL_MPP_AFFINE
    if(m>4&&gpu.HasMPPAffine()){relaxed=GARNET_METAL_MPP_AFFINE_RELAXED_VALUE!=0;nMajor=GARNET_METAL_MPP_AFFINE_N_MAJOR_VALUE!=0;registers=GARNET_METAL_MPP_AFFINE_REGISTERS_VALUE!=0;}
#endif
    if(reuse64Dispatches){registers=true;nMajor=false;}
    const char* operand="float32";
#ifdef GARNET_METAL_MPP_AFFINE_BF16
    if(m>4&&gpu.HasMPPAffine())operand="bfloat16";
#endif
    std::cout<<"{\"kind\":\"cell\",\"relaxed_precision\":"<<(relaxed?"true":"false")<<",\"mpp_n_major\":"<<(nMajor?"true":"false")<<",\"mpp_registers\":"<<(registers?"true":"false")<<",\"operand_dtype\":"<<std::quoted(operand)<<",\"rows\":"<<m<<",\"outputs\":"<<n<<",\"inner\":"<<k
             <<",\"warmups\":"<<warmups<<",\"repetitions\":"<<repetitions<<",\"mpp_dispatches_per_call\":"<<mppDispatches<<",\"reuse64_dispatches_per_call\":"<<reuse64Dispatches
             <<",\"sample_max_abs\":"<<maximum<<",\"gpu_ms\":";Array(gpuMs);std::cout<<",\"wall_ms\":";Array(wallMs);std::cout<<"}\n"<<std::flush;
}
}
int main(int argc,char** argv) {
    try {
        const int repetitions=argc>1?std::stoi(argv[1]):9,warmups=argc>2?std::stoi(argv[2]):3;
        Require(repetitions>=3&&repetitions<=100&&warmups>=1&&warmups<=20,"Invalid benchmark repetitions/warmups");
        const int maximumRows=argc>3?std::stoi(argv[3]):64;
        Require(maximumRows==64||maximumRows==128||maximumRows==512,"Invalid maximum packed row count");
        const std::string shapeSet=argc>4?argv[4]:"synthetic";
        Require(shapeSet=="synthetic"||shapeSet=="qwen27b","Invalid projection shape set");
        const std::vector<std::array<size_t,2>> dimensions=shapeSet=="qwen27b"?
            std::vector<std::array<size_t,2>>{{48,5120},{1024,5120},{5120,6144},{5120,17408},{6144,5120},{10240,5120},{12288,5120},{17408,5120}}:
            std::vector<std::array<size_t,2>>{{5120,5120},{17408,5120},{5120,17408}};
        MetalContext gpu;bool mpp=false;
#ifdef GARNET_METAL_MPP_AFFINE
        mpp=gpu.HasMPPAffine();
#endif
        int tileK=0;bool relaxed=false,nMajor=false,registers=false;
#ifdef GARNET_METAL_MPP_AFFINE
        if(mpp){tileK=GARNET_METAL_MPP_AFFINE_TILE_K;relaxed=GARNET_METAL_MPP_AFFINE_RELAXED_VALUE!=0;nMajor=GARNET_METAL_MPP_AFFINE_N_MAJOR_VALUE!=0;registers=GARNET_METAL_MPP_AFFINE_REGISTERS_VALUE!=0;}
#endif
        const char* operand="float32";
        bool reuse64=false;
#ifdef GARNET_METAL_MPP_AFFINE_REUSE64
        reuse64=gpu.HasReuse64Affine();
#endif
#ifdef GARNET_METAL_MPP_AFFINE_BF16
        if(mpp)operand="bfloat16";
#endif
        std::cout<<std::setprecision(12)<<"{\"kind\":\"metadata\",\"device\":"<<std::quoted(gpu.DeviceName())<<",\"mpp_available\":"<<(mpp?"true":"false")
                 <<",\"shape_set\":"<<std::quoted(shapeSet)
                 <<",\"reuse64_available\":"<<(reuse64?"true":"false")<<",\"reuse64_min_outputs\":"<<(reuse64?5120:0)<<",\"reuse64_registers\":"<<(reuse64?"true":"false")<<",\"relaxed_precision\":"<<(relaxed?"true":"false")<<",\"mpp_n_major\":"<<(nMajor?"true":"false")<<",\"mpp_registers\":"<<(registers?"true":"false")<<",\"maximum_rows\":"<<maximumRows<<",\"mpp_tensor_k\":"<<(registers?16:tileK)<<",\"mpp_tile_k\":"<<tileK<<",\"operand_dtype\":"<<std::quoted(operand)<<",\"accumulator_dtype\":\"float32\",\"scope\":\"Synthetic affine projection GPU/wall latency; excludes loading, compilation and downloads; not model throughput\"}\n";
        for(size_t m:{1,2,4,8,16,32,64,128,256,512})if(m<=size_t(maximumRows))for(auto shape:dimensions)
            Cell(gpu,m,shape[0],shape[1],warmups,repetitions);
        return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<"\n";return 1;}
}
