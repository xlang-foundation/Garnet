// SPDX-License-Identifier: Apache-2.0
#include "metal_context.h"
#include "indexed_decode_clear_policy.h"
#include "../../src/model/affine_int4_codec.h"
#include "../../src/model/float16_codec.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>
using namespace Garnet;
namespace {
size_t cases=0,words=0,fp64=0,rejections=0;
void Need(bool value,const char* reason){if(!value)throw std::runtime_error(reason);}
template<class F>void Reject(F f){bool rejected=false;try{f();}catch(const std::exception&){rejected=true;}Need(rejected,"Expected native rejection missing");++rejections;}
void Same(const std::vector<float>& actual,const std::vector<float>& expected){Need(actual.size()==expected.size()&&!std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float)),"Complete native output/input parent differs");words+=actual.size();}
void Enable(MetalContext& gpu,bool enabled){
#ifdef GARNET_METAL_INDEXED_DECODE_FUSED_CLEAR
    gpu.SetIndexedDecodeFusedClearEnabled(enabled);
#else
    if(enabled)Reject([&]{gpu.SetIndexedDecodeFusedClearEnabled(true);});
    gpu.SetIndexedDecodeFusedClearEnabled(false);
#endif
}
void Case(MetalContext& gpu,size_t rows,size_t n,size_t k,bool prepare=true){
    constexpr size_t guard=16;
    constexpr float poison=-19.375f;
    auto layout=AffineInt4Layout::Create(n,k,AffineInt4CoefficientType::FP16);
    std::vector<uint32_t> packed(layout.weightBytes/4);
    std::vector<uint16_t> scales(layout.coefficientBytes/2),biases(scales.size());
    for(size_t i=0;i<packed.size();++i)packed[i]=uint32_t(i)*2654435761u+131u;
    for(size_t i=0;i<scales.size();++i){
        scales[i]=Float32ToFloat16RNE(float(int(i%13)-6)*.00171f);
        biases[i]=Float32ToFloat16RNE(float(int(i%17)-8)*.00313f);
        if(i%97==0)scales[i]=uint16_t(i%2?0x8000:0x0001);
    }
    auto weight=gpu.UploadAffineInt4(packed.data(),layout.weightBytes,scales.data(),layout.coefficientBytes,biases.data(),layout.coefficientBytes,n,k,true);
    auto parentLength=[&](size_t columns){return ((guard+rows*columns+guard+127)/128)*128;};
    std::vector<float> input(parentLength(k),poison),blank(parentLength(n),poison);
    uint32_t seed=37;for(size_t i=0;i<rows*k;++i){seed=1664525u*seed+1013904223u;input[guard+i]=float(int((seed>>8)%65521)-32760)*0x1p-18f;}
    auto ip=gpu.Upload(input,1,input.size()),op=gpu.Upload(blank,1,blank.size());
    auto a=gpu.PrepareDiagnosticDenseView(ip,guard,rows,k),out=gpu.PrepareDiagnosticDenseView(op,guard,rows,n);
    if(prepare)Need(gpu.PrepareAffineInt4Profile(a,weight,out,rows),"Cannot prepare native affine profiles");
    const auto preparedCount=gpu.AffineProfilePipelineCount();
    auto execute=[&](const std::vector<size_t>& ids,bool candidate){
        Enable(gpu,candidate);gpu.Update(op,blank);gpu.Begin();
        gpu.EncodeAffineInt4IndexedDecode(a,weight,out,ids);
        Reject([&]{gpu.Download(out);});
        Reject([&]{gpu.SetIndexedDecodeFusedClearEnabled(false);});
        gpu.SubmitAndWait();
        bool fused=false;
#ifdef GARNET_METAL_INDEXED_DECODE_FUSED_CLEAR
        fused=candidate&&prepare&&IndexedDecodeClearSupported(rows,rows,ids.size(),n,n);
#endif
        Need(gpu.LastIndexedDecodeFusedClearDispatches()==size_t(fused),"Wrong actual fused inactive-zero route");
        Need(gpu.LastIndexedDecodeBlitClears()==size_t(ids.size()<rows&&!fused),"Wrong native blit-clear count");
        Need(gpu.LastIndexedDecodeAffineDispatches()==size_t(!ids.empty()),"Wrong native projection count");
        Need(gpu.LastAffineProfileDispatches()==size_t(prepare&&!ids.empty())&&gpu.LastAffineProfileMisses()==size_t(!prepare&&!ids.empty()),"Wrong prepared profile route/miss count");
        Need(gpu.AffineProfilePipelineCount()==preparedCount,"Request-time native profile compilation");
        return gpu.Download(op);
    };
    std::vector<size_t> all(rows);std::iota(all.begin(),all.end(),0);
    const auto dense=execute(all,false);
    AffineInt4View view(layout,packed.data(),layout.weightBytes,scales.data(),layout.coefficientBytes,biases.data(),layout.coefficientBytes);
    std::vector<float> decoded(k);
    std::vector<size_t> columns{0,n-1,n/2,std::min(n-1,size_t(3)),std::min(n-1,size_t(31)),std::min(n-1,size_t(32)),std::min(n-1,size_t(127)),std::min(n-1,size_t(128))};
    std::sort(columns.begin(),columns.end());columns.erase(std::unique(columns.begin(),columns.end()),columns.end());
    for(auto col:columns){
        view.DecodeRow(col,decoded.data(),k);
        for(size_t row=0;row<rows;++row){
            double expected=0,absolute=0;
            for(size_t j=0;j<k;++j){const double term=double(input[guard+row*k+j])*double(decoded[j]);expected+=term;absolute+=std::abs(term);}
            const float actual=dense[guard+row*n+col];
            Need(std::isfinite(actual)&&std::abs(double(actual)-expected)<=2e-6+4e-6*absolute,"Independent FP64 dot bound failed");++fp64;
        }
    }
    std::vector<std::vector<size_t>> selections;
    if(rows<=4){
        for(size_t mask=0;mask<(size_t(1)<<rows);++mask){
            std::vector<size_t> ids;for(size_t row=rows;row>0;--row)if(mask&(size_t(1)<<(row-1)))ids.push_back(row-1);
            selections.push_back(ids);
        }
    }else selections={{},{rows-1},{rows-1,0},{3,7,11},all};
    for(const auto& ids:selections){
        const auto control=execute(ids,false),candidate=execute(ids,true);
        auto expected=blank;
        for(size_t row=0;row<rows;++row){
            const bool selected=std::find(ids.begin(),ids.end(),row)!=ids.end();
            for(size_t col=0;col<n;++col)expected[guard+row*n+col]=selected?dense[guard+row*n+col]:0.0f;
        }
        Same(control,expected);Same(candidate,control);Same(gpu.Download(ip),input);++cases;
    }
    // Inactive inputs may be poisoned; the zero writer must neither use them
    // in a product nor change their stored bits.
    auto inactivePoison=input;
    for(size_t row=0;row+1<rows;++row)std::fill_n(inactivePoison.data()+guard+row*k,k,std::numeric_limits<float>::quiet_NaN());
    gpu.Update(ip,inactivePoison);
    Same(execute({rows-1},true),execute({rows-1},false));Same(gpu.Download(ip),inactivePoison);
    gpu.Update(ip,input);
    // Invalid controls must fail before clearing any byte, even with the
    // candidate enabled and an otherwise available prepared pipeline.
    Enable(gpu,true);gpu.Update(op,blank);
    for(auto ids:std::vector<std::vector<size_t>>{{0,0},{rows},{SIZE_MAX}}){
        gpu.Begin();Reject([&]{gpu.EncodeAffineInt4IndexedDecode(a,weight,out,ids);});
        Need(gpu.LastIndexedDecodeFusedClearDispatches()==0&&gpu.LastIndexedDecodeBlitClears()==0,"Invalid control encoded a writer");
        gpu.SubmitAndWait();Same(gpu.Download(op),blank);
    }
    auto aliasOwner=gpu.Upload(std::vector<float>(parentLength(std::max(n,k)),poison),1,parentLength(std::max(n,k)));
    auto aliasInput=gpu.PrepareDiagnosticDenseView(aliasOwner,guard,rows,k),aliasOutput=gpu.PrepareDiagnosticDenseView(aliasOwner,guard,rows,n);
    const auto aliasBefore=gpu.Download(aliasOwner);
    gpu.Begin();Reject([&]{gpu.EncodeAffineInt4IndexedDecode(aliasInput,weight,aliasOutput,{0});});gpu.SubmitAndWait();Same(gpu.Download(aliasOwner),aliasBefore);
    // Abort and partial-submit recovery retain the original pending-write gate.
    std::vector<size_t> partial{rows-1};
    gpu.Update(op,blank);gpu.Begin();gpu.EncodeAffineInt4IndexedDecode(a,weight,out,partial);gpu.Abort();
    Need(gpu.LastIndexedDecodeFusedClearDispatches()==0&&gpu.LastIndexedDecodeBlitClears()==0,"Abort retained dispatch counters");
    Reject([&]{gpu.Download(out);});
    Same(execute(partial,true),execute(partial,false));
    gpu.Update(op,blank);gpu.Begin();gpu.EncodeAffineInt4IndexedDecode(a,weight,out,partial);gpu.Flush();gpu.Abort();
    Reject([&]{gpu.Download(out);});
    Same(execute(partial,true),execute(partial,false));
    std::cout<<"indexed fused clear case PASS B="<<rows<<" K="<<k<<" N="<<n<<" prepared="<<prepare<<" masks="<<selections.size()<<" full_parent_exact=true\n"<<std::flush;
}
void Policy(){
    for(size_t b:{0ul,1ul,2ul,3ul,4ul,16ul})for(size_t out:{0ul,2ul,4ul,16ul})
    for(size_t selected=0;selected<=5;++selected)for(size_t width:{0ul,1ul,17ul,4096ul,32768ul,SIZE_MAX})
    for(size_t padding:{0ul,1ul}){
        const size_t stride=width==SIZE_MAX?SIZE_MAX:width+padding;
        const bool expected=b==4&&out==4&&selected>0&&selected<4&&width>0&&!padding&&width<=UINT32_MAX/4;
        Need(IndexedDecodeClearSupported(b,out,selected,width,stride)==expected,"Fused-clear geometry predicate mismatch");
    }
}
void PaddedFallback(MetalContext& gpu){
    // Use actual native allocation policy, including its recommended row stride.
    // Logical words are checked here; the fused path cannot touch row padding.
    const size_t rows=4,n=17,k=192;
    auto layout=AffineInt4Layout::Create(n,k,AffineInt4CoefficientType::FP16);
    std::vector<uint32_t> packed(layout.weightBytes/4,0x76543210u);
    std::vector<uint16_t> scales(layout.coefficientBytes/2,0x2980),biases(scales.size(),0xa900);
    auto weight=gpu.UploadAffineInt4(packed.data(),layout.weightBytes,scales.data(),layout.coefficientBytes,biases.data(),layout.coefficientBytes,n,k,true);
    auto input=gpu.Upload(std::vector<float>(rows*k,.0625f),rows,k),output=gpu.Allocate(rows,n);
    gpu.Zero(output);
    Need(gpu.PrepareAffineInt4Profile(input,weight,output,rows),"Padded profile preparation failed");
    const auto info=MetalContext::MatrixBufferInfo(output);
    Need(info.lengthBytes%(rows*sizeof(float))==0,"Unexpected native allocation extent");
    const size_t stride=info.lengthBytes/(rows*sizeof(float));
    const std::vector<size_t> selected{3,0};
    auto run=[&](bool enabled){Enable(gpu,enabled);gpu.Update(output,std::vector<float>(rows*n,-99));gpu.Begin();gpu.EncodeAffineInt4IndexedDecode(input,weight,output,selected);gpu.SubmitAndWait();bool fused=false;
#ifdef GARNET_METAL_INDEXED_DECODE_FUSED_CLEAR
        fused=enabled&&IndexedDecodeClearSupported(rows,rows,selected.size(),n,stride);
#endif
        Need(gpu.LastIndexedDecodeFusedClearDispatches()==size_t(fused)&&gpu.LastIndexedDecodeBlitClears()==size_t(!fused),"Padded allocation selected incorrect zero writer");return gpu.Download(output);};
    Same(run(true),run(false));++cases;
    std::cout<<"indexed clear allocation fallback PASS columns="<<n<<" stride="<<stride<<" logical_words_exact=true\n";
}
}
int main(){try{
    Policy();
    {MetalContext gpu;
        for(auto shape:std::vector<std::pair<size_t,size_t>>{{20,64},{36,128},{1024,4096},{4096,4096},{14336,4096},{4096,14336},{32768,4096}})Case(gpu,4,shape.first,shape.second);
        // Previously prepared B4 geometry must never authorize extra B1/B2 rows.
        Case(gpu,1,20,64);Case(gpu,2,20,64);Case(gpu,4,40,320,false);
        PaddedFallback(gpu);
#ifdef GARNET_METAL_WIDE_DENSE_VERIFICATION
        Case(gpu,16,20,64);
#endif
    }
    auto owners=MetalContext::ProfileOwnerStats();Need(!owners.matrices&&!owners.affineWeights&&!owners.bf16Weights&&!owners.pipelineSets,"Native owners retained");
    std::cout<<"indexed fused clear PASS cases="<<cases<<" complete_parent_words="<<words<<" FP64_samples="<<fp64<<" rejections="<<rejections<<" owners=0\n";return 0;
}catch(const std::exception& error){std::cerr<<"indexed fused clear FAILED: "<<error.what()<<'\n';return 1;}}
