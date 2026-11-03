// SPDX-License-Identifier: Apache-2.0
// Numerical/ownership gates, not throughput evidence or full-model validation.
#include "metal_context.h"
#include "../../src/model/affine_int4_codec.h"
#include "../../src/model/mlx_vision_patch.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>

using namespace Garnet;
namespace {
uint16_t BF16(float value) {
    uint32_t bits;std::memcpy(&bits,&value,4);
    return uint16_t((bits+0x7fff+((bits>>16)&1))>>16);
}
void Require(bool condition,const char* message) {if(!condition)throw std::runtime_error(message);}
template<class Action> void Reject(Action action) {
    bool rejected=false;try{action();}catch(const std::invalid_argument&){rejected=true;}
    Require(rejected,"invalid affine INT4 input was accepted");
}
void CodecGolden() {
    const auto layout=AffineInt4Layout::Create(2,128);
    std::vector<uint32_t> packed(32,0x76543210u);
    packed[0]=0xfedcba98u;packed[8]=0xfedcba98u;
    for(size_t i=16;i<32;++i)packed[i]=0x11111111u;
    std::vector<uint16_t> scales={BF16(.5f),BF16(-.25f),BF16(1.5f),BF16(.25f)};
    std::vector<uint16_t> biases={BF16(-1),BF16(.125f),BF16(-2),BF16(-.125f)};
    AffineInt4View view(layout,packed.data(),128,scales.data(),8,biases.data(),8);
    std::vector<float> row(128);
    view.DecodeRow(0,row.data(),row.size());
    const std::array<float,8> first={3,3.5f,4,4.5f,5,5.5f,6,6.5f};
    const std::array<float,8> second={-1.875f,-2.125f,-2.375f,-2.625f,-2.875f,-3.125f,-3.375f,-3.625f};
    for(size_t i=0;i<8;++i){Require(row[i]==first[i],"Q4 golden nibble/endian ordering mismatch");
        Require(row[64+i]==second[i],"Q4 golden group boundary mismatch");}
    view.DecodeRow(1,row.data(),row.size());
    for(size_t i=0;i<128;++i)Require(row[i]==(i<64?-.5f:.125f),"Q4 golden row stride mismatch");
    Reject([&]{view.DecodeRow(2,row.data(),128);});
    Reject([&]{view.DecodeRow(0,row.data(),127);});
    Reject([&]{AffineInt4View invalid(layout,packed.data(),127,scales.data(),8,biases.data(),8);});
    Require(AffineInt4RoundBF16(2.0078125f)==2.f,"BF16 even tie rounded upward");
    Require(AffineInt4RoundBF16(2.0234375f)==2.03125f,"BF16 odd tie did not round upward");
    Require(AffineInt4RoundBF16(-2.0234375f)==-2.03125f,"negative BF16 tie mismatch");
    const auto embeddingLayout=AffineInt4Layout::Create(1,64);
    std::vector<uint32_t> embeddingWords(8,0x22222222u);
    const uint16_t embeddingScale=0x3f80,embeddingBias=0x3c00;
    AffineInt4View embedding(embeddingLayout,embeddingWords.data(),32,&embeddingScale,2,&embeddingBias,2);
    std::vector<float> embeddingRow(64);
    embedding.DecodeRow(0,embeddingRow.data(),64);
    Require(embeddingRow[0]==2.0078125f,"projection dequantization was rounded prematurely");
    embedding.DecodeEmbeddingRow(0,embeddingRow.data(),64);
    for(auto value:embeddingRow)Require(value==2.f,"embedding did not preserve BF16 dequantization semantics");
}
void VisionPatchGolden() {
    std::vector<float> source(24);
    for(size_t i=0;i<source.size();++i)source[i]=float(i);
    const std::vector<float> expected={0,3,6,9,1,4,7,10,2,5,8,11,
                                     12,15,18,21,13,16,19,22,14,17,20,23};
    Require(MlxVisionPatchToCanonical(source,2,2,1,2,3)==expected,
            "converted vision patch channel/temporal order mismatch");
    Reject([&]{MlxVisionPatchToCanonical(source,2,2,1,2,2);});
    Reject([&]{MlxVisionPatchToCanonical(source,SIZE_MAX,2,1,2,3);});
}
void Case(MetalContext& gpu,size_t m,size_t n,size_t k) {
    const auto layout=AffineInt4Layout::Create(n,k);
    std::vector<uint32_t> packed(layout.weightBytes/4,0);
    std::vector<uint16_t> scales(layout.coefficientBytes/2),biases(scales.size());
    for(size_t row=0;row<n;++row)for(size_t column=0;column<k;++column) {
        const auto q=uint32_t((row*7+column*11+column/17)%16);
        packed[row*(k/8)+column/8]|=q<<((column%8)*4);
    }
    for(size_t i=0;i<scales.size();++i) {
        scales[i]=BF16(float(int(i%13)-6)*.017f);
        biases[i]=BF16(float(int(i%17)-8)*.031f);
    }
    AffineInt4View view(layout,packed.data(),layout.weightBytes,scales.data(),layout.coefficientBytes,
                        biases.data(),layout.coefficientBytes);
    std::vector<float> decoded(n*k),input(m*k);
    for(size_t row=0;row<n;++row)view.DecodeRow(row,decoded.data()+row*k,k);
    for(size_t i=0;i<input.size();++i)input[i]=float(std::sin(double(i)*.173)*.05);
    auto weight=gpu.UploadAffineInt4(packed.data(),layout.weightBytes,scales.data(),layout.coefficientBytes,
                                    biases.data(),layout.coefficientBytes,n,k);
    const auto info=MetalContext::AffineInt4BufferInfo(weight);
    Require(info.size()==3,"packed weight must own exactly three buffers");
    uint64_t bytes=0;std::set<uintptr_t> owners;
    for(const auto& buffer:info){bytes+=buffer.lengthBytes;owners.insert(buffer.identity);}
    Require(bytes==layout.weightBytes+2*layout.coefficientBytes&&owners.size()==3,
            "packed weights were expanded or storage aliased");
    auto a=gpu.Upload(input,m,k),out=gpu.Allocate(m,n);
    gpu.Begin();gpu.EncodeAffineInt4MatMul(a,weight,out);
    // Encoded commands must retain their resources through submission.
    weight=MetalAffineInt4Weight();
    gpu.SubmitAndWait();const auto actual=gpu.Download(out);
    double maximum=0;
    for(size_t row=0;row<m;++row)for(size_t column=0;column<n;++column) {
        double expected=0;
        for(size_t inner=0;inner<k;++inner)expected+=double(input[row*k+inner])*decoded[column*k+inner];
        const auto error=std::abs(double(actual[row*n+column])-expected);
        maximum=std::max(maximum,error);
        Require(std::isfinite(actual[row*n+column])&&error<=std::max(1e-4,std::abs(expected)*3e-4),
                "affine INT4 projection differs from independent FP64 contraction");
    }
    std::cout<<"affine_int4 rows="<<m<<" outputs="<<n<<" inner="<<k<<" max_abs="<<maximum<<"\n";
}
}
int main() {
    try {
        Reject([]{AffineInt4Layout::Create(1,63);});
        Reject([]{AffineInt4Layout::Create(SIZE_MAX,64);});
        CodecGolden();
        VisionPatchGolden();
        MetalContext gpu;
        for(auto shape:std::array<std::array<size_t,3>,16>{{
            {1,37,128},{2,48,5120},{4,37,17408},{4,257,512},{8,64,128},{9,37,128},
            {5,1,64},{5,31,128},{8,37,5120},{8,33,17408},
            {16,80,64},{17,37,128},{32,48,512},{64,257,128},{128,64,64},{512,48,128}}})
            Case(gpu,shape[0],shape[1],shape[2]);
        std::vector<uint32_t> packed(64*64/8,0);
        std::vector<uint16_t> scales(64,BF16(.25f)),biases(64,BF16(-.5f));
        Reject([&]{gpu.UploadAffineInt4(packed.data(),packed.size()*4-1,scales.data(),128,biases.data(),128,64,64);});
        scales[0]=0x7f80;
        Reject([&]{gpu.UploadAffineInt4(packed.data(),packed.size()*4,scales.data(),128,biases.data(),128,64,64);});
        scales[0]=BF16(.25f);
        auto weight=gpu.UploadAffineInt4(packed.data(),packed.size()*4,scales.data(),128,biases.data(),128,64,64);
        auto a=gpu.Upload(std::vector<float>(64,1),1,64);
        gpu.Begin();Reject([&]{gpu.EncodeAffineInt4MatMul(a,weight,a);});gpu.Abort();
        Case(gpu,1,37,128);
        std::cout<<"affine_int4 numerical, tail, ownership and rejection gates passed\n";return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<"\n";return 1;}
}
