// SPDX-License-Identifier: Apache-2.0
#include "metal_context.h"
#include "../../src/model/affine_int4_codec.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace Garnet;
namespace {
void Need(bool value,const char* why){if(!value)throw std::runtime_error(why);}
size_t checks=0,words=0;
bool NoMpp(MetalContext& gpu) {
#ifdef GARNET_METAL_MPP_AFFINE
    return gpu.LastMPPAffineDispatches()==0;
#else
    (void)gpu;return true;
#endif
}
void Same(const std::vector<float>& actual,const std::vector<float>& expected) {
    Need(actual.size()==expected.size(),"Verification affine output size differs");
    for(size_t i=0;i<actual.size();++i)if(std::memcmp(&actual[i],&expected[i],4))
        throw std::runtime_error("Packed FP32 verification check"+std::to_string(checks)+" word"+std::to_string(i)+" differs: "+std::to_string(actual[i])+" vs "+std::to_string(expected[i]));
    ++checks;words+=actual.size();
}
void Case(MetalContext& gpu,size_t n,size_t k) {
    const auto layout=AffineInt4Layout::Create(n,k);
    std::vector<uint32_t> packed(layout.weightBytes/4);
    for(size_t i=0;i<packed.size();++i)packed[i]=uint32_t(i)*2654435761u+131u;
    std::vector<uint16_t> scales(layout.coefficientBytes/2),biases(scales.size());
    for(size_t i=0;i<scales.size();++i){scales[i]=uint16_t(0x3b80+i%8);biases[i]=uint16_t(0xbd00+i%7);}
    auto weight=gpu.UploadAffineInt4(packed.data(),layout.weightBytes,scales.data(),layout.coefficientBytes,biases.data(),layout.coefficientBytes,n,k);
    std::vector<float> input(4*k),expected(4*n);
    uint32_t seed=37;
    for(auto& value:input){seed=1664525u*seed+1013904223u;value=float(int((seed>>8)%65521)-32760)*.000004f;}
    for(size_t row=0;row<4;++row) {
        auto source=gpu.Upload(std::vector<float>(input.begin()+row*k,input.begin()+(row+1)*k),1,k),out=gpu.Allocate(1,n);
        Need(gpu.PrepareAffineInt4Profile(source,weight,out,4),"Cannot prepare ordinary FP32 control");
        gpu.Begin();gpu.EncodeAffineInt4MatMul(source,weight,out);gpu.SubmitAndWait();
        Need(NoMpp(gpu)&&gpu.LastAffineProfileDispatches()==1&&gpu.LastAffineProfileMisses()==0,"Control used a different projection policy");
        auto values=gpu.Download(out);std::copy(values.begin(),values.end(),expected.begin()+row*n);
    }
    auto source=gpu.Upload(input,4,k),output=gpu.Allocate(4,n);
    Need(gpu.PrepareAffineInt4Profile(source,weight,output,4),"Cannot prepare four-row FP32 profile");
    gpu.Begin();gpu.EncodeAffineInt4MatMul(source,weight,output);gpu.SubmitAndWait();
    Need(NoMpp(gpu)&&gpu.LastAffineProfileDispatches()==1&&gpu.LastAffineProfileMisses()==0,"Four-row verification used BF16 or unprepared projections");
    Same(gpu.Download(output),expected);
    for(const std::vector<size_t>& selected:std::vector<std::vector<size_t>>{{0},{0,1},{0,1,2},{0,1,2,3},{0,3},{3},{}}) {
        // Indexed decode explicitly zeroes all inactive logical output rows.
        // Seed the actual output with a sentinel to check that full write.
        std::vector<float> sparse(4*n,0.f);
        for(size_t row:selected)std::copy_n(expected.data()+row*n,n,sparse.data()+row*n);
        gpu.Update(output,std::vector<float>(4*n,-99.f));gpu.Begin();
        gpu.EncodeAffineInt4IndexedDecode(source,weight,output,selected);gpu.SubmitAndWait();
        Need(NoMpp(gpu)&&gpu.LastIndexedDecodeAffineDispatches()==size_t(!selected.empty())&&gpu.LastAffineProfileMisses()==0,"Sparse verification route differs");
        Same(gpu.Download(output),sparse);
    }
    std::cout<<"verification affine shape PASS K="<<k<<" N="<<n<<" physical_rows=4 valid_rows=0..4 ordinary_fp32_bits=true\n"<<std::flush;
}
}
int main(){try{
    {
        MetalContext gpu;
        for(auto shape:std::vector<std::pair<size_t,size_t>>{{17,64},{33,128},{5120,5120},{17408,5120},{5120,17408},{248320,5120}})
            Case(gpu,shape.first,shape.second);
    }
    const auto owners=MetalContext::ProfileOwnerStats();Need(!owners.matrices&&!owners.affineWeights&&!owners.bf16Weights&&!owners.pipelineSets,"Verification affine owners remain after close");
    std::cout<<"verification affine FP32 reference PASS checks="<<checks<<" fp32_words="<<words<<" full_vocab_columns=248320 selected_rows=0..4 owners=0\n";
    return 0;
}catch(const std::exception& error){std::cerr<<"verification affine FP32 reference FAILED: "<<error.what()<<"\n";return 1;}}
