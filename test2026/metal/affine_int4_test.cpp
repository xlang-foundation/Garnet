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
#ifdef GARNET_METAL_AFFINE_PROFILE
    Require(gpu.PrepareAffineInt4Profile(a,weight,out,m),"Cannot prepare affine test profile");
#endif
    gpu.Begin();gpu.EncodeAffineInt4MatMul(a,weight,out);
    // Encoded commands must retain their resources through submission.
    weight=MetalAffineInt4Weight();
    gpu.SubmitAndWait();const auto actual=gpu.Download(out);
#ifdef GARNET_METAL_AFFINE_PROFILE
    Require(gpu.LastAffineProfileDispatches()==1&&gpu.LastAffineProfileMisses()==0,"Prepared affine profile missed");
#endif
#ifdef GARNET_METAL_MPP_AFFINE
    Require(gpu.LastMPPAffineDispatches()==size_t(m>4&&gpu.HasMPPAffine()),"MPP affine route/counter mismatch");
#endif
#ifdef GARNET_METAL_MPP_AFFINE_REUSE64
    Require(gpu.LastReuse64AffineDispatches()==size_t(m>=64&&n>=5120&&gpu.HasReuse64Affine()),"Cooperative64 affine route/counter mismatch");
#endif
    double maximum=0;
    for(size_t row=0;row<m;++row)for(size_t column=0;column<n;++column) {
        double expected=0;
        for(size_t inner=0;inner<k;++inner) {
            float left=input[row*k+inner],right=decoded[column*k+inner];
#ifdef GARNET_METAL_MPP_AFFINE_BF16
            if(m>4&&gpu.HasMPPAffine()){left=AffineInt4RoundBF16(left);right=AffineInt4RoundBF16(right);}
#endif
            expected+=double(left)*right;
        }
        const auto error=std::abs(double(actual[row*n+column])-expected);
        maximum=std::max(maximum,error);
        Require(std::isfinite(actual[row*n+column])&&error<=std::max(1e-4,std::abs(expected)*3e-4),
                "affine INT4 projection differs from independent FP64 contraction");
    }
    std::cout<<"affine_int4 rows="<<m<<" outputs="<<n<<" inner="<<k<<" max_abs="<<maximum<<"\n";
}
#ifdef GARNET_METAL_INDEXED_AFFINE
void IndexedCase(MetalContext& gpu,size_t m,size_t n,size_t k) {
    const auto layout=AffineInt4Layout::Create(n,k);
    std::vector<uint32_t> packed(layout.weightBytes/4,0);
    std::vector<uint16_t> scales(layout.coefficientBytes/2),biases(scales.size());
    for(size_t row=0;row<n;++row)for(size_t column=0;column<k;++column)
        packed[row*(k/8)+column/8]|=uint32_t((row*3+column*7)%16)<<((column%8)*4);
    for(size_t i=0;i<scales.size();++i){scales[i]=BF16(float(int(i%11)-5)*.019f);biases[i]=BF16(float(int(i%7)-3)*.027f);}
    AffineInt4View view(layout,packed.data(),layout.weightBytes,scales.data(),layout.coefficientBytes,biases.data(),layout.coefficientBytes);
    std::vector<float> decoded(n*k),input(m*k);
    for(size_t row=0;row<n;++row)view.DecodeRow(row,decoded.data()+row*k,k);
    for(size_t i=0;i<input.size();++i)input[i]=float(std::sin(double(i)*.139)*.07);
    auto weight=gpu.UploadAffineInt4(packed.data(),layout.weightBytes,scales.data(),layout.coefficientBytes,biases.data(),layout.coefficientBytes,n,k);
    auto a=gpu.Upload(input,m,k),out=gpu.Upload(std::vector<float>(m*n,17.f),m,n);
#ifdef GARNET_METAL_AFFINE_PROFILE
    Require(gpu.PrepareAffineInt4Profile(a,weight,out,m),"Cannot prepare indexed affine test profile");
#endif
    std::vector<size_t> all(m);for(size_t i=0;i<m;++i)all[i]=m-1-i;
    std::vector<std::vector<size_t>> selections={all,{m-1},{0},{},all,{}};
    if(m>4)selections.insert(selections.end()-1,{m-1,1,3});
    for(const auto& selected:selections) {
        gpu.Begin();gpu.EncodeAffineInt4IndexedMatMul(a,weight,out,selected);gpu.SubmitAndWait();
        Require(gpu.LastIndexedAffineDispatches()==size_t(!selected.empty()),"indexed affine route/counter mismatch");
#ifdef GARNET_METAL_AFFINE_PROFILE
        Require(gpu.LastAffineProfileDispatches()==size_t(!selected.empty())&&gpu.LastAffineProfileMisses()==0,"Prepared indexed affine profile missed");
#endif
#ifdef GARNET_METAL_MPP_AFFINE_REUSE64
        Require(gpu.LastReuse64AffineDispatches()==size_t(selected.size()>=64&&n>=5120&&gpu.HasReuse64Affine()),"indexed cooperative64 route mismatch");
#endif
        const auto actual=gpu.Download(out);std::set<size_t> active(selected.begin(),selected.end());
        for(size_t row=0;row<m;++row)for(size_t column=0;column<n;++column) {
            double expected=0;
            if(active.count(row))for(size_t inner=0;inner<k;++inner)
                expected+=double(AffineInt4RoundBF16(input[row*k+inner]))*AffineInt4RoundBF16(decoded[column*k+inner]);
            Require(std::isfinite(actual[row*n+column])&&
                std::abs(double(actual[row*n+column])-expected)<=std::max(1e-4,std::abs(expected)*3e-4),
                "indexed affine projection differs from BF16/FP64 oracle");
            if(!active.count(row))Require(actual[row*n+column]==0.f,"inactive indexed output retained old values");
        }
    }
    gpu.Begin();Reject([&]{gpu.EncodeAffineInt4IndexedMatMul(a,weight,out,{0,0});});gpu.Abort();
    gpu.Begin();Reject([&]{gpu.EncodeAffineInt4IndexedMatMul(a,weight,out,{m});});gpu.Abort();
    const auto zero=gpu.Download(out);for(auto value:zero)Require(value==0.f,"rejected indexed selection mutated output");
    std::cout<<"indexed_affine physical_rows="<<m<<" outputs="<<n<<" inner="<<k<<" masks="<<selections.size()<<"\n";
}
#endif
}
#ifdef GARNET_METAL_INDEXED_DECODE
void IndexedDecodeCase(MetalContext& gpu,size_t m,size_t n,size_t k) {
    const auto layout=AffineInt4Layout::Create(n,k);
    std::vector<uint32_t> packed(layout.weightBytes/4);
    std::vector<uint16_t> scales(layout.coefficientBytes/2),biases(scales.size());
    for(size_t i=0;i<packed.size();++i)packed[i]=uint32_t(i*2654435761u)^0x76543210u;
    for(size_t i=0;i<scales.size();++i){scales[i]=BF16(float(int(i%13)-6)*.017f);biases[i]=BF16(float(int(i%17)-8)*.031f);}
    std::vector<float> input(m*k),decoded(n*k);
    for(size_t i=0;i<input.size();++i)input[i]=float(std::sin(double(i)*.173)*.05);
    AffineInt4View view(layout,packed.data(),layout.weightBytes,scales.data(),layout.coefficientBytes,biases.data(),layout.coefficientBytes);
    for(size_t row=0;row<n;++row)view.DecodeRow(row,decoded.data()+row*k,k);
    auto weight=gpu.UploadAffineInt4(packed.data(),layout.weightBytes,scales.data(),layout.coefficientBytes,biases.data(),layout.coefficientBytes,n,k);
    auto a=gpu.Upload(input,m,k),out=gpu.Upload(std::vector<float>(m*n,17.f),m,n),dense=gpu.Allocate(m,n);
#ifdef GARNET_METAL_AFFINE_PROFILE
    Require(gpu.PrepareAffineInt4Profile(a,weight,out,m),"Cannot prepare indexed decode test profile");
#endif
    gpu.Begin();gpu.EncodeAffineInt4MatMul(a,weight,dense);gpu.SubmitAndWait();
    const auto reference=gpu.Download(dense);
    for(size_t mask=0;mask<(size_t(1)<<m);++mask) {
        std::vector<size_t> selected;
        for(size_t row=m;row>0;--row)if(mask&(size_t(1)<<(row-1)))selected.push_back(row-1);
        gpu.Begin();gpu.EncodeAffineInt4IndexedDecode(a,weight,out,selected);gpu.SubmitAndWait();
        Require(gpu.LastIndexedDecodeAffineDispatches()==size_t(!selected.empty()),"Indexed FP32 decode route/counter mismatch");
#ifdef GARNET_METAL_AFFINE_PROFILE
        Require(gpu.LastAffineProfileDispatches()==size_t(!selected.empty())&&gpu.LastAffineProfileMisses()==0,"Prepared indexed decode profile missed");
#endif
        const auto actual=gpu.Download(out);
        for(size_t row=0;row<m;++row)for(size_t column=0;column<n;++column) {
            double expected=0;const bool active=mask&(size_t(1)<<row);
            if(active)for(size_t inner=0;inner<k;++inner)expected+=double(input[row*k+inner])*decoded[column*k+inner];
            Require(std::isfinite(actual[row*n+column])&&std::abs(double(actual[row*n+column])-expected)<=std::max(1e-4,std::abs(expected)*3e-4),
                "Indexed decode differs from unrounded FP32/FP64 oracle");
            Require(actual[row*n+column]==(active?reference[row*n+column]:0.f),"Indexed decode differs from ordinary decode or retained inactive values");
        }
    }
    const auto before=gpu.Download(out);
    gpu.Begin();Reject([&]{gpu.EncodeAffineInt4IndexedDecode(a,weight,out,{0,0});});gpu.Abort();
    gpu.Begin();Reject([&]{gpu.EncodeAffineInt4IndexedDecode(a,weight,out,{m});});gpu.Abort();
    gpu.Begin();Reject([&]{gpu.EncodeAffineInt4IndexedDecode(a,weight,a,{0});});gpu.Abort();
    auto uninitialized=gpu.Allocate(m,k);
    bool uninitializedRejected=false;gpu.Begin();
    try{gpu.EncodeAffineInt4IndexedDecode(uninitialized,weight,out,{0});}catch(const std::logic_error&){uninitializedRejected=true;}
    gpu.Abort();Require(uninitializedRejected,"Uninitialized indexed FP32 decode input was accepted");
    Require(gpu.Download(out)==before,"Rejected indexed decode mutated output");
    std::cout<<"indexed_decode physical_rows="<<m<<" outputs="<<n<<" inner="<<k<<" masks="<<(size_t(1)<<m)<<" exact_dense=1\n";
}
#endif
#ifdef GARNET_METAL_AFFINE_PROFILE
void ProfileBoundsCase(MetalContext& gpu) {
    constexpr size_t n=79,k=192;
    const auto layout=AffineInt4Layout::Create(n,k);
    std::vector<uint32_t> packed(layout.weightBytes/4,0x76543210u);
    std::vector<uint16_t> scales(layout.coefficientBytes/2,BF16(.03125f)),biases(scales.size(),BF16(-.125f));
    auto weight=gpu.UploadAffineInt4(packed.data(),layout.weightBytes,scales.data(),layout.coefficientBytes,biases.data(),layout.coefficientBytes,n,k);
    auto first=gpu.Upload(std::vector<float>(k,.0625f),1,k),firstOut=gpu.Allocate(1,n);
    gpu.Begin();gpu.EncodeAffineInt4MatMul(first,weight,firstOut);gpu.SubmitAndWait();
    Require(gpu.LastAffineProfileDispatches()==0&&gpu.LastAffineProfileMisses()==1,"Unprepared profile did not use fallback");
    const auto reference=gpu.Download(firstOut);const auto before=gpu.AffineProfilePipelineCount();
    Require(gpu.PrepareAffineInt4Profile(first,weight,firstOut,128),"Cannot prepare up-to-128 profile from one actual row");
    const auto prepared=gpu.AffineProfilePipelineCount();
    Require(prepared>before,"Profile preparation did not compile pipelines");
    Require(gpu.PrepareAffineInt4Profile(first,weight,firstOut,128)&&gpu.AffineProfilePipelineCount()==prepared,"Profile preparation was not idempotent");
    Require(!gpu.PrepareAffineInt4Profile(first,weight,firstOut,0)&&!gpu.PrepareAffineInt4Profile(first,weight,firstOut,513),"Unsupported profile bounds were accepted");
    gpu.Begin();bool rejected=false;
    try{gpu.PrepareAffineInt4Profile(first,weight,firstOut,128);}catch(const std::logic_error&){rejected=true;}
    gpu.Abort();Require(rejected,"Profile compilation was accepted inside an open command");
    for(size_t rows:{1,2,4,5,63,127,128,129}) {
        auto a=gpu.Upload(std::vector<float>(rows*k,.0625f),rows,k),out=gpu.Allocate(rows,n);
        gpu.Begin();gpu.EncodeAffineInt4MatMul(a,weight,out);gpu.SubmitAndWait();
        Require(gpu.LastAffineProfileDispatches()==size_t(rows<=128)&&gpu.LastAffineProfileMisses()==size_t(rows>128),"Actual rows did not respect the compiled upper bound");
        for(float value:gpu.Download(out))Require(value==reference[0],"Bounded profile changed exact representable projection");
        Require(gpu.AffineProfilePipelineCount()==prepared,"Encoding lazily compiled a pipeline");
        if(rows==129) {
            Require(gpu.PrepareAffineInt4Profile(a,weight,out,512),"Cannot widen the prepared profile bound");
            gpu.Begin();gpu.EncodeAffineInt4MatMul(a,weight,out);gpu.SubmitAndWait();
            Require(gpu.LastAffineProfileDispatches()==1&&gpu.LastAffineProfileMisses()==0,"Widened profile did not accept shorter input");
        }
    }
    std::cout<<"affine_profile up_to_128 accepts=1,2,4,5,63,127,128 fallback=129 no_lazy_compile=1 widened_bound=512\n";
}
#endif
int main() {
    try {
        Reject([]{AffineInt4Layout::Create(1,63);});
        Reject([]{AffineInt4Layout::Create(SIZE_MAX,64);});
        CodecGolden();
        VisionPatchGolden();
        MetalContext gpu;
#ifdef GARNET_METAL_AFFINE_PROFILE
        ProfileBoundsCase(gpu);
#endif
#ifdef GARNET_METAL_INDEXED_DECODE
        Require(gpu.HasIndexedDecodeAffine(),"Indexed FP32 decode pipeline unavailable");
        for(auto shape:std::array<std::array<size_t,3>,5>{{{1,37,64},{2,48,5120},{3,33,17408},{4,17408,64},{4,5127,128}}})
            IndexedDecodeCase(gpu,shape[0],shape[1],shape[2]);
#endif
#ifdef GARNET_METAL_MPP_AFFINE_REUSE64
        Require(gpu.HasReuse64Affine(),"cooperative64 affine unavailable");
        Case(gpu,63,5121,128);Case(gpu,64,5121,128);Case(gpu,65,5127,128);
        IndexedCase(gpu,128,5127,128);
#endif
#ifdef GARNET_METAL_INDEXED_AFFINE
        Require(gpu.HasIndexedAffine(),"indexed affine candidate unavailable");
        for(auto shape:std::array<std::array<size_t,3>,6>{{{1,37,64},{17,33,128},{32,37,5120},
            {65,33,17408},{128,35,64},{512,37,128}}})IndexedCase(gpu,shape[0],shape[1],shape[2]);
#endif
#ifdef GARNET_METAL_MPP_AFFINE
#ifdef GARNET_METAL_MPP_AFFINE_BF16
        std::cout<<"mpp_affine_bf16 ";
#else
        std::cout<<"mpp_affine_fp32 ";
#endif
        std::cout<<"device="<<gpu.DeviceName()<<" available="<<gpu.HasMPPAffine()<<" relaxed="<<GARNET_METAL_MPP_AFFINE_RELAXED_VALUE<<" registers="<<GARNET_METAL_MPP_AFFINE_REGISTERS_VALUE<<" tensor_k="<<(GARNET_METAL_MPP_AFFINE_REGISTERS_VALUE?16:GARNET_METAL_MPP_AFFINE_TILE_K)<<" n_major="<<GARNET_METAL_MPP_AFFINE_N_MAJOR_VALUE<<" tile_k="<<GARNET_METAL_MPP_AFFINE_TILE_K<<"\n";
        Require(gpu.DeviceName().find("M5")==std::string::npos||gpu.HasMPPAffine(),"M5 did not select its MPP candidate");
#endif
        for(auto shape:std::array<std::array<size_t,3>,24>{{
            {1,37,128},{2,48,5120},{4,37,17408},{4,257,512},{8,64,128},{9,37,128},
            {5,1,64},{5,31,128},{8,37,5120},{8,33,17408},
            {16,37,5120},{32,33,17408},{64,37,5120},{64,33,17408},
            {16,80,64},{17,37,128},{32,48,512},{33,37,128},{63,35,64},{65,37,128},{127,33,64},
            {64,257,128},{128,64,64},{512,48,128}}})
            Case(gpu,shape[0],shape[1],shape[2]);
#ifdef GARNET_METAL_QWEN38_LARGE_PREFILL
        Case(gpu,256,37,5120);Case(gpu,512,33,17408);
#endif
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
