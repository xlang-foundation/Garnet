// SPDX-License-Identifier: Apache-2.0
#include "metal_context.h"
#include "mistral_n4_layout.h"
#include "../../src/model/affine_int4_codec.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace Garnet;
void Need(bool v,const char* why){if(!v)throw std::runtime_error(why);}
int main(){try{
 size_t compared=0;
 for(auto shape:std::array<std::array<size_t,2>,2>{{{4096,14336},{14336,4096}}}){{
    const size_t k=shape[0],n=shape[1];MetalContext gpu;
    const auto layout=AffineInt4Layout::Create(n,k,AffineInt4CoefficientType::FP16);
    std::vector<uint32_t> packed(layout.weightBytes/4);std::vector<uint16_t> scales(layout.coefficientBytes/2),biases(scales.size());
    for(size_t i=0;i<packed.size();i++)packed[i]=uint32_t(i)*2654435761u+37;
    for(size_t i=0;i<scales.size();i++){scales[i]=uint16_t(0x2c00+i%8);biases[i]=uint16_t(0xb000+i%7);}
    std::vector<float> input(4*k);uint32_t seed=37;
    for(auto& x:input){seed=1664525u*seed+1013904223u;x=float(int((seed>>8)%65521)-32760)*0x1p-18f;}
    auto weight=gpu.UploadAffineInt4(packed.data(),layout.weightBytes,scales.data(),layout.coefficientBytes,biases.data(),layout.coefficientBytes,n,k,true);
    auto x=gpu.Upload(input,4,k),out=gpu.Allocate(4,n);
    // No profile exists yet: this is the original scalar FP32 native control.
    gpu.Begin();gpu.EncodeAffineInt4MatMul(x,weight,out);gpu.SubmitAndWait();
    Need(gpu.LastAffineProfileMisses()==1&&gpu.LastN4DecodeDispatches()==0,"Control was not the unprepared scalar route");
    const auto gold=gpu.Download(out);
    Need(gpu.PrepareAffineInt4Profile(x,weight,out,4),"Cannot prepare bounded profile");
    const auto ready=MetalContext::N4Stats();
#ifdef GARNET_METAL_MISTRAL_N4_DECODE
    Need(ready.weights==1&&ready.bytes==MistralN4PayloadBytes(k,n),"N4 owner or payload charge differs");
    Need(MetalContext::AffineInt4BufferInfo(weight).size()==6,"N4 buffers absent from inventory");
#else
    Need(ready.weights==0&&ready.bytes==0,"Default-off profile allocated N4 cache");
#endif
    for(size_t mask=0;mask<16;mask++){
        std::vector<size_t> ids;for(size_t row=0;row<4;row++)if(mask&(size_t(1)<<row))ids.push_back(row);
        gpu.Begin();gpu.EncodeAffineInt4IndexedDecode(x,weight,out,ids);gpu.SubmitAndWait();const auto actual=gpu.Download(out);
        Need(gpu.LastAffineProfileMisses()==0,"Prepared profile missed");
#ifdef GARNET_METAL_MISTRAL_N4_DECODE
        Need(gpu.LastN4DecodeDispatches()==size_t(mask==15),"N4 selected outside four-active-row profile");
#else
        Need(gpu.LastN4DecodeDispatches()==0,"Default-off profile selected N4");
#endif
        for(size_t row=0;row<4;row++)for(size_t col=0;col<n;col++){
            float wanted=(mask&(size_t(1)<<row))?gold[row*n+col]:0.f;
            Need(std::memcmp(&wanted,&actual[row*n+col],4)==0,"Native selected row output differs from scalar control");++compared;
        }
    }
    gpu.Begin();gpu.EncodeAffineInt4MatMul(x,weight,out);gpu.SubmitAndWait();const auto dense=gpu.Download(out);
    Need(!std::memcmp(gold.data(),dense.data(),gold.size()*4),"Native four-row output differs from scalar control");compared+=gold.size();
#ifdef GARNET_METAL_MISTRAL_N4_DECODE
    Need(gpu.LastN4DecodeDispatches()==1,"Four-row MLP did not use N4");
#endif
    const auto afterInput=gpu.Download(x);Need(!std::memcmp(input.data(),afterInput.data(),input.size()*4),"Read-only native input changed");
 }}
 const auto closed=MetalContext::N4Stats();const auto owners=MetalContext::ProfileOwnerStats();
 Need(closed.weights==0&&closed.bytes==0,"N4 cache charge retained after owner close");
 Need(!owners.matrices&&!owners.affineWeights&&!owners.pipelineSets,"Native owners retained after close");
 std::cout<<"Native N4 decode PASS exact_words="<<compared<<" masks=16 per_shape=2 occupancy4_only\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
