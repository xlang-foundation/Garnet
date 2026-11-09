// SPDX-License-Identifier: Apache-2.0
#include "metal_context.h"
#include "indexed_decode_clear_policy.h"
#include "../../src/model/affine_int4_codec.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
using namespace Garnet;
namespace {
size_t checks=0,words=0,rejected=0;
void Need(bool value,const char* why){if(!value)throw std::runtime_error(why);}
void Same(const std::vector<float>& actual,const std::vector<float>& expected) {
    Need(actual.size()==expected.size(),"Output length differs");
    for(size_t i=0;i<actual.size();++i)
        if(!std::isfinite(actual[i])||std::memcmp(&actual[i],&expected[i],4))
            throw std::runtime_error("Projection differs/missing write at word "+std::to_string(i));
    ++checks;words+=actual.size();
}
bool Blocked(size_t n,size_t k,size_t selected) {
    const bool shape=(k==4096&&(n==4096||n==14336||n==32768))||(k==14336&&n==4096);
    bool count=selected==4;
#ifdef GARNET_METAL_MISTRAL_THREE_ROW_REUSE
    count=count||selected==3;
#endif
    return shape&&count;
}
void Route(MetalContext& gpu,size_t n,size_t k,size_t selected) {
    Need(gpu.LastAffineProfileMisses()==0,"Prepared profile missed");
    Need(gpu.LastAffineProfileDispatches()==size_t(selected!=0),"Wrong prepared dispatch count");
    Need(gpu.LastAffineDecodeBlockDispatches()==size_t(Blocked(n,k,selected)),"Wrong occupancy layout");
    Need(gpu.LastMPPAffineDispatches()==0,"FP32 decode used MPP");
}
void Case(size_t n,size_t k) {
    MetalContext gpu;
    const auto layout=AffineInt4Layout::Create(n,k,AffineInt4CoefficientType::FP16);
    std::vector<uint32_t> packed(layout.weightBytes/4);
    for(size_t i=0;i<packed.size();++i)packed[i]=uint32_t(i)*2654435761u+131u;
    std::vector<uint16_t> scales(layout.coefficientBytes/2),biases(scales.size());
    for(size_t i=0;i<scales.size();++i){scales[i]=uint16_t(0x2180+i%53);biases[i]=uint16_t(0xa900+i%47);}
    auto weight=gpu.UploadAffineInt4(packed.data(),layout.weightBytes,scales.data(),layout.coefficientBytes,biases.data(),layout.coefficientBytes,n,k,true);
    std::vector<float> input(4*k),expected(4*n);
    uint32_t seed=37;
    for(auto& value:input){seed=1664525u*seed+1013904223u;value=float(int((seed>>8)%65521)-32760)*.000004f;}
    const float poison=std::numeric_limits<float>::quiet_NaN();
    auto source=gpu.Upload(input,4,k),output=gpu.Upload(std::vector<float>(4*n,poison),4,n);
    // Unprepared four-row scalar path is the bitwise control, then check every
    // physical row independently with the unchanged single-row shader.
    gpu.Begin();gpu.EncodeAffineInt4MatMul(source,weight,output);gpu.SubmitAndWait();
    Need(gpu.LastAffineProfileMisses()==1&&gpu.LastAffineDecodeBlockDispatches()==0,"Unprepared route changed");
    expected=gpu.Download(output);
    for(size_t row=0;row<4;++row) {
        auto one=gpu.Upload(std::vector<float>(input.begin()+row*k,input.begin()+(row+1)*k),1,k);
        auto out=gpu.Upload(std::vector<float>(n,poison),1,n);
        gpu.Begin();gpu.EncodeAffineInt4MatMul(one,weight,out);gpu.SubmitAndWait();
        Same(gpu.Download(out),std::vector<float>(expected.begin()+row*n,expected.begin()+(row+1)*n));
    }
    Need(gpu.PrepareAffineInt4Profile(source,weight,output,4),"Cannot prepare native profile");
    size_t expectedPipelines=9;
#ifdef GARNET_METAL_INDEXED_DECODE_FUSED_CLEAR
    const auto info=MetalContext::MatrixBufferInfo(output);
    Need(info.lengthBytes%(4*sizeof(float))==0,"Invalid four-row output span");
    const size_t stride=info.lengthBytes/(4*sizeof(float));
    if(IndexedDecodeClearSupported(4,4,1,n,stride))expectedPipelines+=3;
#endif
    const size_t pipelines=gpu.AffineProfilePipelineCount();
    Need(pipelines==expectedPipelines,"Wrong precompiled occupancy family");
    Need(gpu.PrepareAffineInt4Profile(source,weight,output,4)&&gpu.AffineProfilePipelineCount()==pipelines,"Repeated preparation grew cache");
    for(size_t rows=1;rows<=4;++rows) {
        auto dense=gpu.Upload(std::vector<float>(input.begin(),input.begin()+rows*k),rows,k);
        auto out=gpu.Upload(std::vector<float>(rows*n,poison),rows,n);
        Need(gpu.PrepareAffineInt4Profile(dense,weight,out,4),"Cannot prepare dense occupancy");
        gpu.Update(out,std::vector<float>(rows*n,poison));
        gpu.Begin();gpu.EncodeAffineInt4MatMul(dense,weight,out);gpu.SubmitAndWait();
        Route(gpu,n,k,rows);
        Same(gpu.Download(out),std::vector<float>(expected.begin(),expected.begin()+rows*n));
    }
    // All sixteen masks, in every permutation: includes noncontiguous row3,
    // physical capacity4/active3, every omitted row, and full/empty occupancies.
    for(unsigned mask=0;mask<16;++mask) {
        std::vector<size_t> selected;
        for(size_t row=0;row<4;++row)if(mask&(1u<<row))selected.push_back(row);
        do {
            std::vector<float> sparse(4*n,0.f);
            for(auto row:selected)std::copy_n(expected.data()+row*n,n,sparse.data()+row*n);
            gpu.Update(output,std::vector<float>(4*n,poison));
            gpu.Begin();gpu.EncodeAffineInt4IndexedDecode(source,weight,output,selected);gpu.SubmitAndWait();
            Route(gpu,n,k,selected.size());
            Need(gpu.LastIndexedDecodeAffineDispatches()==size_t(!selected.empty()),"Wrong indexed route");
            Same(gpu.Download(output),sparse);
        }while(std::next_permutation(selected.begin(),selected.end()));
    }
    for(const std::vector<size_t>& selected:std::vector<std::vector<size_t>>{{0,0,2},{0,2,4},{0,1,2,3,4}}) {
        gpu.Update(output,std::vector<float>(4*n,-99.f));gpu.Begin();bool threw=false;
        try{gpu.EncodeAffineInt4IndexedDecode(source,weight,output,selected);}catch(const std::invalid_argument&){threw=true;++rejected;}
        gpu.Abort();Need(threw,"Invalid active rows accepted");
        Same(gpu.Download(output),std::vector<float>(4*n,-99.f));
    }
    Need(gpu.AffineProfilePipelineCount()==pipelines,"Dispatch compiled new profiles");
    std::cout<<"mistral occupancy PASS K="<<k<<" N="<<n<<" masks=16 all_permutations=true poisoned=true\n"<<std::flush;
}
}
int main(){try{
    for(auto shape:std::vector<std::pair<size_t,size_t>>{{17,64},{33,128},{1024,4096},{4096,4096},{14336,4096},{4096,14336},{32768,4096}})
        Case(shape.first,shape.second);
    const auto owners=MetalContext::ProfileOwnerStats();
    Need(!owners.matrices&&!owners.affineWeights&&!owners.bf16Weights&&!owners.pipelineSets,"Owners remain after close");
    std::cout<<"Mistral projection occupancy PASS checks="<<checks<<" fp32_words="<<words<<" rejected="<<rejected<<" owners=0\n";
    return 0;
}catch(const std::exception& e){std::cerr<<"Mistral projection occupancy FAILED: "<<e.what()<<"\n";return 1;}}
