// SPDX-License-Identifier: Apache-2.0
// Typed FP16 coefficients, strict FP16 tensor operands, FP32 accumulation.
#include "../../src/backends/metal/metal_context.h"
#include "../../src/backends/metal/affine_reuse128_policy.h"
#include "../../src/model/affine_int4_codec.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>

using namespace Garnet;
namespace {
size_t checks=0,words=0;
void Require(bool condition,const char* message) {++checks;if(!condition)throw std::runtime_error(message);}
template<class F> void Reject(F call) {bool rejected=false;try{call();}catch(const std::exception&){rejected=true;}Require(rejected,"missing FP16 representation/control rejection");}
float Half(float x) {return Float16ToFloat32(Float32ToFloat16RNE(x));}
struct Fixture {
    size_t m,n,k;
    std::vector<float> input,decoded;
    MetalMatrix a,out,dense;
    MetalAffineInt4Weight weight;
    Fixture(MetalContext& gpu,size_t rows,size_t columns,size_t inner):m(rows),n(columns),k(inner) {
        auto layout=AffineInt4Layout::Create(n,k,AffineInt4CoefficientType::FP16);
        std::vector<uint32_t> packed(layout.weightBytes/4);
        std::vector<uint16_t> scales(layout.coefficientBytes/2),biases(scales.size());
        for(size_t i=0;i<packed.size();++i)packed[i]=uint32_t(i*2654435761u)^0x76543210u;
        for(size_t i=0;i<scales.size();++i) {
            scales[i]=Float32ToFloat16RNE(float(int(i%13)-6)*.00171f);
            biases[i]=Float32ToFloat16RNE(float(int(i%17)-8)*.00313f);
        }
        AffineInt4View view(layout,packed.data(),layout.weightBytes,scales.data(),layout.coefficientBytes,biases.data(),layout.coefficientBytes);
        decoded.resize(n*k);for(size_t i=0;i<n;++i)view.DecodeRow(i,decoded.data()+i*k,k);
        input.resize(m*k);for(size_t i=0;i<input.size();++i)input[i]=float(std::sin(double(i)*.173)*.05);
        Reject([&]{gpu.UploadAffineInt4(packed.data(),layout.weightBytes,scales.data(),layout.coefficientBytes,biases.data(),layout.coefficientBytes,n,k,false);});
        weight=gpu.UploadAffineInt4(packed.data(),layout.weightBytes,scales.data(),layout.coefficientBytes,biases.data(),layout.coefficientBytes,n,k,true);
        a=gpu.Upload(input,m,k);out=gpu.Upload(std::vector<float>(m*n,17.f),m,n);dense=gpu.Allocate(m,n);
        Require(gpu.PrepareAffineInt4Profile(a,weight,out,m),"cannot prepare FP16 affine profile");
    }
    void Compare(const std::vector<float>& actual,const std::vector<size_t>& selected,bool fp16Operands) {
        Require(actual.size()==m*n,"wrong FP16 projection output size");std::vector<bool> active(m);
        for(auto row:selected)active[row]=true;
        for(size_t row=0;row<m;++row)for(size_t col=0;col<n;++col) {
            double expected=0,absolute=0;
            if(active[row])for(size_t inner=0;inner<k;++inner) {
                double left=input[row*k+inner],right=decoded[col*k+inner];
                if(fp16Operands){left=Half(float(left));right=Half(float(right));}
                expected+=left*right;absolute+=std::abs(left*right);
            }
            const float value=actual[row*n+col];
            Require(std::isfinite(value)&&std::abs(double(value)-expected)<=2e-6+4e-6*absolute,"FP16 projection differs from independent FP64 dot oracle");
            if(!active[row])Require(value==0.f,"inactive FP16 row retained stale values");
            ++words;
        }
    }
};
void CoefficientBits(MetalContext& gpu,bool biasCase) {
    std::vector<uint16_t> finite;
    for(uint32_t value=0;value<65536;++value)if(std::isfinite(Float16ToFloat32(uint16_t(value))))finite.push_back(uint16_t(value));
    const size_t n=finite.size(),k=64;
    auto layout=AffineInt4Layout::Create(n,k,AffineInt4CoefficientType::FP16);
    std::vector<uint32_t> packed(layout.weightBytes/4,0);std::vector<uint16_t> zeros(n);
    if(!biasCase)for(size_t row=0;row<n;++row)packed[row*8]=1;
    auto& scales=biasCase?zeros:finite;auto& biases=biasCase?finite:zeros;
    auto weight=gpu.UploadAffineInt4(packed.data(),layout.weightBytes,scales.data(),layout.coefficientBytes,biases.data(),layout.coefficientBytes,n,k,true);
    std::vector<float> input(k);input[0]=1.f;
    auto a=gpu.Upload(input,1,k),out=gpu.Allocate(1,n);
    if(!biasCase) {
        auto count=gpu.AffineProfilePipelineCount();gpu.Begin();gpu.EncodeAffineInt4MatMul(a,weight,out);gpu.SubmitAndWait();
        Require(gpu.LastAffineProfileDispatches()==0&&gpu.LastAffineProfileMisses()==1&&gpu.AffineProfilePipelineCount()==count,"unprepared FP16 profile did not use noncompiling fallback");
        for(auto value:gpu.Download(out))Require(std::isfinite(value),"unprepared FP16 fallback returned nonfinite data");
    }
    Require(gpu.PrepareAffineInt4Profile(a,weight,out,1),"cannot prepare coefficient-bit profile");
    auto count=gpu.AffineProfilePipelineCount();gpu.Begin();gpu.EncodeAffineInt4MatMul(a,weight,out);gpu.SubmitAndWait();
    Require(gpu.AffineProfilePipelineCount()==count&&gpu.LastAffineProfileMisses()==0,"coefficient profile compiled/missed during encode");
    const auto actual=gpu.Download(out);
    for(size_t i=0;i<n;++i){Require(actual[i]==Float16ToFloat32(finite[i]),"Metal FP16 coefficient expansion is inexact (including subnormals)");++words;}
}
void Bounds(MetalContext& gpu) {
    constexpr size_t n=79,k=192;
    auto layout=AffineInt4Layout::Create(n,k,AffineInt4CoefficientType::FP16);
    std::vector<uint32_t> packed(layout.weightBytes/4,0x76543210u);
    std::vector<uint16_t> scales(layout.coefficientBytes/2,Float32ToFloat16RNE(.03125f)),biases(scales.size(),Float32ToFloat16RNE(-.125f));
    auto w=gpu.UploadAffineInt4(packed.data(),layout.weightBytes,scales.data(),layout.coefficientBytes,biases.data(),layout.coefficientBytes,n,k,true);
    auto a=gpu.Upload(std::vector<float>(k,.0625f),1,k),out=gpu.Allocate(1,n);
    Require(gpu.PrepareAffineInt4Profile(a,w,out,128),"cannot prepare up-to-128 FP16 profile from one actual row");
    auto count=gpu.AffineProfilePipelineCount();
    Require(gpu.PrepareAffineInt4Profile(a,w,out,128)&&gpu.AffineProfilePipelineCount()==count,"FP16 preparation not idempotent");
    Require(!gpu.PrepareAffineInt4Profile(a,w,out,0)&&!gpu.PrepareAffineInt4Profile(a,w,out,513),"FP16 profile accepted invalid upper bounds");
    gpu.Begin();Reject([&]{gpu.PrepareAffineInt4Profile(a,w,out,128);});gpu.Abort();
    for(size_t m:{1,2,4,5,63,127,128,129}) {
        auto input=gpu.Upload(std::vector<float>(m*k,.0625f),m,k),output=gpu.Allocate(m,n);
        gpu.Begin();gpu.EncodeAffineInt4MatMul(input,w,output);gpu.SubmitAndWait();
        Require(gpu.LastAffineProfileDispatches()==size_t(m<=128)&&gpu.LastAffineProfileMisses()==size_t(m>128),"FP16 actual rows did not respect profile maximum");
        Require(gpu.AffineProfilePipelineCount()==count,"FP16 runtime compiled a profile");
        for(auto value:gpu.Download(output)){Require(value==-.1875f,"FP16 upper-bound profile changed an exact projection");++words;}
        if(m==129) {
            Require(gpu.PrepareAffineInt4Profile(input,w,output,512),"cannot widen FP16 profile upper bound");
            gpu.Begin();gpu.EncodeAffineInt4MatMul(input,w,output);gpu.SubmitAndWait();
            Require(gpu.LastAffineProfileDispatches()==1&&gpu.LastAffineProfileMisses()==0,"widened FP16 profile not selected");
        }
    }
    for(uint16_t invalid:{uint16_t(0x7c00),uint16_t(0xfc00),uint16_t(0x7e00)}) {
        auto bad=scales;bad[0]=invalid;
        Reject([&]{gpu.UploadAffineInt4(packed.data(),layout.weightBytes,bad.data(),layout.coefficientBytes,biases.data(),layout.coefficientBytes,n,k,true);});
    }
}
void Dense(MetalContext& gpu,size_t m,size_t n,size_t k) {
    Fixture f(gpu,m,n,k);std::vector<size_t> all(m);std::iota(all.begin(),all.end(),0);
    auto count=gpu.AffineProfilePipelineCount();
    gpu.Begin();gpu.EncodeAffineInt4MatMul(f.a,f.weight,f.dense);gpu.SubmitAndWait();
    Require(gpu.LastAffineProfileDispatches()==1&&gpu.LastAffineProfileMisses()==0,"dense FP16 prepared profile missed");
    Require(gpu.AffineProfilePipelineCount()==count,"dense FP16 encode compiled a pipeline");
    if(m>4)Require(gpu.LastMPPAffineDispatches()==1,"FP16 tensor path was not used");
#ifdef GARNET_METAL_MPP_AFFINE_REUSE64
    size_t reuse128=0;
#ifdef GARNET_METAL_MPP_AFFINE_REUSE128
    reuse128=size_t(UseAffineReuse128(m,n,GARNET_METAL_REUSE64_MIN_OUTPUTS));
    Require(gpu.LastReuse128AffineDispatches()==reuse128,"dense FP16 cooperative128 route mismatch");
#endif
    Require(gpu.LastReuse64AffineDispatches()==size_t(m>=64&&n>=GARNET_METAL_REUSE64_MIN_OUTPUTS)-reuse128,"dense FP16 cooperative64 threshold mismatch");
#endif
    f.Compare(gpu.Download(f.dense),all,m>4);
    std::cout<<"fp16_dense rows="<<m<<" columns="<<n<<" inner="<<k<<'\n';
}
void Indexed(MetalContext& gpu,size_t m,size_t n,size_t k,bool decode) {
    Fixture f(gpu,m,n,k);const auto count=gpu.AffineProfilePipelineCount();
    if(decode){gpu.Begin();gpu.EncodeAffineInt4MatMul(f.a,f.weight,f.dense);gpu.SubmitAndWait();}
    std::vector<float> dense=decode?gpu.Download(f.dense):std::vector<float>();
    std::vector<std::vector<size_t>> selections;
    if(m<=4)for(size_t mask=0;mask<(size_t(1)<<m);++mask) {
        std::vector<size_t> selected;for(size_t i=m;i>0;--i)if(mask&(size_t(1)<<(i-1)))selected.push_back(i-1);
        selections.push_back(selected);
    }else for(size_t active:{size_t(0),size_t(1),size_t(7),size_t(16),size_t(17),size_t(63),size_t(64),size_t(127),size_t(128),m}) {
        if(active>m)continue;std::vector<size_t> selected(active);for(size_t i=0;i<active;++i)selected[i]=m-1-i;selections.push_back(selected);
    }
    for(const auto& selected:selections) {
        gpu.Begin();
        if(decode)gpu.EncodeAffineInt4IndexedDecode(f.a,f.weight,f.out,selected);
        else gpu.EncodeAffineInt4IndexedMatMul(f.a,f.weight,f.out,selected);
        gpu.SubmitAndWait();
        Require(gpu.LastAffineProfileDispatches()==size_t(!selected.empty())&&gpu.LastAffineProfileMisses()==0,"indexed FP16 profile dispatch/miss mismatch");
        Require(gpu.AffineProfilePipelineCount()==count,"indexed FP16 encode compiled a pipeline");
#ifdef GARNET_METAL_MPP_AFFINE_REUSE64
        size_t reuse128=0;
#ifdef GARNET_METAL_MPP_AFFINE_REUSE128
        reuse128=size_t(!decode&&UseAffineReuse128(selected.size(),n,GARNET_METAL_REUSE64_MIN_OUTPUTS));
        Require(gpu.LastReuse128AffineDispatches()==reuse128,"indexed FP16 cooperative128 route mismatch");
#endif
        Require(gpu.LastReuse64AffineDispatches()==size_t(!decode&&selected.size()>=64&&n>=GARNET_METAL_REUSE64_MIN_OUTPUTS)-reuse128,"indexed FP16 cooperative64 threshold mismatch");
#endif
        auto actual=gpu.Download(f.out);f.Compare(actual,selected,!decode);
        if(decode)for(auto row:selected)for(size_t col=0;col<n;++col)
            Require(actual[row*n+col]==dense[row*n+col],"FP16 indexed decode differs bitwise from dense decode");
    }
    const auto before=gpu.Download(f.out);
    gpu.Begin();Reject([&]{if(decode)gpu.EncodeAffineInt4IndexedDecode(f.a,f.weight,f.out,{0,0});else gpu.EncodeAffineInt4IndexedMatMul(f.a,f.weight,f.out,{0,0});});gpu.Abort();
    gpu.Begin();Reject([&]{if(decode)gpu.EncodeAffineInt4IndexedDecode(f.a,f.weight,f.out,{m});else gpu.EncodeAffineInt4IndexedMatMul(f.a,f.weight,f.out,{m});});gpu.Abort();
    Require(gpu.Download(f.out)==before,"rejected FP16 indexed control mutated output");
    std::cout<<"fp16_indexed physical_rows="<<m<<" columns="<<n<<" inner="<<k<<" decode="<<decode<<" masks="<<selections.size()<<'\n';
}
}
int main() {
    try {
        {
            MetalContext gpu;
            Require(gpu.HasMPPAffine()&&gpu.HasIndexedAffine()&&gpu.HasIndexedDecodeAffine(),"actual FP16 tensor/indexed pipelines required");
            Require(MetalContext::AffineProfileIndexBits()==32,"FP16 test requires bounded32-bit profiles");
            CoefficientBits(gpu,false);CoefficientBits(gpu,true);
            Bounds(gpu);
            for(auto m:{size_t(1),size_t(2),size_t(3),size_t(4),size_t(5),size_t(16),size_t(64),size_t(511),size_t(512)})Dense(gpu,m,37,128);
            Dense(gpu,4,1024,4096);Dense(gpu,16,1024,4096);Dense(gpu,4,37,14336);Dense(gpu,16,37,14336);
            for(size_t m=1;m<=4;++m){Indexed(gpu,m,37,128,true);Indexed(gpu,m,37,128,false);}
            Indexed(gpu,512,48,128,false);
#ifdef GARNET_METAL_MPP_AFFINE_REUSE64
            Require(gpu.HasReuse64Affine(),"actual FP16 cooperative64 pipeline required");
            // Output threshold and64-row boundary; ragged final row/output tiles.
            for(size_t m:{size_t(63),size_t(64),size_t(65)})
                for(size_t n:{size_t(GARNET_METAL_REUSE64_MIN_OUTPUTS-1),size_t(GARNET_METAL_REUSE64_MIN_OUTPUTS),size_t(GARNET_METAL_REUSE64_MIN_OUTPUTS+1)})
                    Dense(gpu,m,n,128);
            Indexed(gpu,129,GARNET_METAL_REUSE64_MIN_OUTPUTS+1,64,false);
#endif
        }
        const auto owners=MetalContext::ProfileOwnerStats();
        Require(!owners.matrices&&!owners.affineWeights&&!owners.bf16Weights&&!owners.pipelineSets,"FP16 profile leaked native owners");
        std::cout<<"fp16_affine passed checks="<<checks<<" fp32_words="<<words<<" owners=0\n";return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
