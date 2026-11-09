// SPDX-License-Identifier: Apache-2.0
#include "metal_context.h"
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
size_t cases=0,words=0,rejections=0,samples=0;
void Need(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
template<class F>void Reject(F f){bool rejected=false;try{f();}catch(const std::invalid_argument&){rejected=true;}catch(const std::logic_error&){rejected=true;}Need(rejected,"Invalid lookahead projection accepted");++rejections;}
void Equal(const std::vector<float>& a,const std::vector<float>& b){Need(a.size()==b.size(),"Projection output size differs");for(size_t i=0;i<a.size();++i)if(std::memcmp(&a[i],&b[i],4))throw std::runtime_error("Projection FP32 word"+std::to_string(i)+" differs");words+=a.size();}
size_t Count(uint16_t mask){size_t n=0;while(mask){n+=mask&1;mask>>=1;}return n;}
LookaheadLayout::View Occupancy(const LookaheadLayout& layout,size_t row,size_t wanted) {
    for(size_t remaining=1;remaining<=8;++remaining)for(size_t filled=0;filled<=2;++filled)for(size_t guesses=0;guesses<=4;++guesses) {
        auto v=layout.Request(row,71+row,7,remaining,guesses,filled);if(Count(v.Active())==wanted)return v;
    }
    throw std::runtime_error("Missing actual layout occupancy"+std::to_string(wanted));
}
struct Weight {
    size_t n,k;AffineInt4Layout layout;std::vector<uint32_t> packed;std::vector<uint16_t> scales,biases;
    Weight(size_t outputs,size_t inner):n(outputs),k(inner),layout(AffineInt4Layout::Create(n,k,AffineInt4CoefficientType::FP16)),packed(layout.weightBytes/4),scales(layout.coefficientBytes/2),biases(scales.size()) {
        for(size_t i=0;i<packed.size();++i)packed[i]=uint32_t(i)*2654435761u+131u;
        for(size_t i=0;i<scales.size();++i){scales[i]=uint16_t(0x2980+i%8);biases[i]=uint16_t(0xa900+i%7);}
    }
    MetalAffineInt4Weight Upload(MetalContext& gpu) const {return gpu.UploadAffineInt4(packed.data(),layout.weightBytes,scales.data(),layout.coefficientBytes,biases.data(),layout.coefficientBytes,n,k,true);}
};
std::vector<float> Values(size_t count){std::vector<float> out(count);uint32_t seed=37;for(auto& v:out){seed=1664525u*seed+1013904223u;v=float(int((seed>>8)%65521)-32760)*.000004f;}return out;}
std::vector<float> Ordinary(MetalContext& gpu,const Weight& w,const MetalAffineInt4Weight& weight,const std::vector<float>& input) {
    std::vector<float> reference(64*w.n);
    for(size_t base=0;base<64;base+=4) {
        auto x=gpu.Upload(std::vector<float>(input.begin()+base*w.k,input.begin()+(base+4)*w.k),4,w.k),y=gpu.Allocate(4,w.n);
        Need(gpu.PrepareAffineInt4Profile(x,weight,y,4),"Ordinary projection profile missing");
        gpu.Begin();gpu.EncodeAffineInt4MatMul(x,weight,y);gpu.SubmitAndWait();
        Need(gpu.LastMPPAffineDispatches()==0&&gpu.LastAffineProfileMisses()==0,"Ordinary reference math route changed");
        auto result=gpu.Download(y);std::copy(result.begin(),result.end(),reference.begin()+base*w.n);
    }
    AffineInt4View codec(w.layout,w.packed.data(),w.layout.weightBytes,w.scales.data(),w.layout.coefficientBytes,w.biases.data(),w.layout.coefficientBytes);
    std::vector<float> decoded(w.k);
    for(size_t col:{size_t(0),w.n/2,w.n-1}) {
        codec.DecodeRow(col,decoded.data(),decoded.size());
        for(size_t row:{size_t(0),size_t(17),size_t(63)}) {
            double value=0;for(size_t k=0;k<w.k;++k)value+=double(input[row*w.k+k])*decoded[k];
            Need(std::abs(double(reference[row*w.n+col])-value)<=2e-5+1e-4*std::abs(value),"Ordinary projection FP64 sanity failed");++samples;
        }
    }
    return reference;
}
void Shape(MetalContext& gpu,size_t n,size_t k) {
    Weight data(n,k);auto weight=data.Upload(gpu);auto allInput=Values(64*k);auto reference=Ordinary(gpu,data,weight,allInput);
    for(size_t batch=1;batch<=4;++batch) {
        LookaheadLayout profile(batch,4096,4,3,4);const size_t rows=batch*16;
        auto source=gpu.Upload(std::vector<float>(allInput.begin(),allInput.begin()+rows*k),rows,k),out=gpu.Allocate(rows,n),
            pending=gpu.Allocate(rows,k),zero=gpu.Upload(std::vector<float>(rows*k,0),rows,k);
        gpu.PrepareLookaheadProjection(source,weight,out,profile);const auto prepared=gpu.LookaheadProjectionPipelineCount();
        gpu.PrepareLookaheadProjection(source,weight,out,profile);Need(gpu.LookaheadProjectionPipelineCount()==prepared,"Projection prepare not idempotent");
        for(size_t count=0;count<=rows;++count) {
            std::vector<LookaheadLayout::View> views;std::vector<float> expected(rows*n,0),input(rows*k,std::numeric_limits<float>::quiet_NaN());size_t remaining=count;
            // Reverse owner order exercises sparse physical rows and index63.
            for(size_t row=batch;row-->0;) {
                const size_t wanted=std::min(size_t(16),remaining);remaining-=wanted;if(!wanted)continue;
                views.push_back(Occupancy(profile,row,wanted));
                for(size_t node=0;node<16;++node)if(views.back().Active()&(uint16_t(1)<<node)) {
                    const size_t flat=row*16+node;std::copy_n(reference.data()+flat*n,n,expected.data()+flat*n);std::copy_n(allInput.data()+flat*k,k,input.data()+flat*k);
                }
            }
            Need(!remaining,"Occupancy allocation exceeds prepared rows");gpu.Update(source,input);gpu.Update(out,std::vector<float>(rows*n,-99));
            gpu.Begin();Need(gpu.LastLookaheadProjectionDispatches()==0,"Begin retained projection counter");auto operand=source;
            if(count%7==2){gpu.EncodeAdd(source,zero,pending);operand=pending;if(count%14==2)gpu.Flush();}
            gpu.EncodeLookaheadProjection(operand,weight,out,profile,views);
            Need(gpu.LastLookaheadProjectionDispatches()==size_t(count>0),"Projection dispatch count differs");
            Reject([&]{gpu.Download(out);});Reject([&]{gpu.Begin();});
            Need(gpu.LastLookaheadProjectionDispatches()==size_t(count>0),"Invalid Begin reset pending counter");
            gpu.SubmitAndWait();Equal(gpu.Download(out),expected);Equal(gpu.Download(source),input);
            Need(gpu.LookaheadProjectionPipelineCount()==prepared&&gpu.LastMPPAffineDispatches()==0,"Projection compiled/fell back during execution");++cases;
        }
    }
    std::cout<<"lookahead projection shape PASS K="<<k<<" N="<<n<<" B=1..4 selected=0..B16 poison_inactive=true\n"<<std::flush;
}
void Invalid(MetalContext& gpu,MetalContext& other) {
    Weight data(4096,4096);auto weight=data.Upload(gpu);LookaheadLayout profile(4,4096,4,3,4),foreign(4,4096,4,4,1);
    auto input=Values(64*4096);auto x=gpu.Upload(input,64,4096),y=gpu.Upload(std::vector<float>(64*4096,-99),64,4096);
    std::vector<LookaheadLayout::View> views{Occupancy(profile,3,7)};const auto before=gpu.Download(y);
    Reject([&]{gpu.EncodeLookaheadProjection(x,weight,y,profile,views);});
    gpu.Begin();Reject([&]{gpu.EncodeLookaheadProjection(x,weight,y,profile,views);});Need(gpu.LastLookaheadProjectionDispatches()==0,"Unprepared projection dispatched");gpu.Abort();Equal(gpu.Download(y),before);
    gpu.PrepareLookaheadProjection(x,weight,y,profile);other.PrepareLookaheadProjection(x,weight,y,profile);
    for(size_t mutation=0;mutation<7;++mutation) {
        auto operand=x,output=y;auto selected=views;auto w=weight;
        if(mutation==0)selected.push_back(selected[0]);
        if(mutation==1)selected[0]=foreign.Request(3,77,7,8,1,3);
        if(mutation==2)operand=gpu.Allocate(64,4096);
        if(mutation==3)operand=MetalMatrix();
        if(mutation==4)w=MetalAffineInt4Weight();
        if(mutation==5)output=x;
        gpu.Begin();
        if(mutation==6)Reject([&]{gpu.PrepareLookaheadProjection(x,weight,y,profile);});
        else Reject([&]{gpu.EncodeLookaheadProjection(operand,w,output,profile,selected);});
        Need(gpu.LastLookaheadProjectionDispatches()==0,"Rejected projection dispatched");gpu.Abort();Equal(gpu.Download(y),before);Equal(gpu.Download(x),input);
    }
    gpu.Begin();Reject([&]{gpu.EncodeAffineInt4IndexedDecode(x,weight,y,{0});});gpu.Abort();Equal(gpu.Download(y),before);
    gpu.Begin();gpu.EncodeLookaheadProjection(x,weight,y,profile,views);
    other.Begin();Reject([&]{other.EncodeLookaheadProjection(x,weight,y,profile,views);});other.Abort();
    gpu.Abort();Need(gpu.LastLookaheadProjectionDispatches()==0,"Abort retained projection counter");Reject([&]{gpu.Download(y);});
    gpu.Begin();gpu.EncodeLookaheadProjection(x,weight,y,profile,views);gpu.SubmitAndWait();const auto recovered=gpu.Download(y);
    auto pending=other.Allocate(64,4096),zero=other.Upload(std::vector<float>(64*4096,0),64,4096);
    other.Begin();other.EncodeAdd(x,zero,pending);gpu.Begin();Reject([&]{gpu.EncodeLookaheadProjection(pending,weight,y,profile,views);});gpu.Abort();other.Abort();Equal(gpu.Download(y),recovered);
}
}
int main(){try{
    {MetalContext gpu,other;Invalid(gpu,other);for(auto shape:std::vector<std::pair<size_t,size_t>>{{1024,4096},{4096,4096},{14336,4096},{4096,14336},{32768,4096}})Shape(gpu,shape.first,shape.second);}
    auto owners=MetalContext::ProfileOwnerStats();Need(!owners.matrices&&!owners.affineWeights&&!owners.bf16Weights&&!owners.pipelineSets,"Lookahead projection owners retained");
    std::cout<<"lookahead projection PASS cases="<<cases<<" fp32_words="<<words<<" fp64_components="<<samples<<" rejections="<<rejections<<" owners=0 ordinary_limits_preserved=true no_model_or_speed_claim=true\n";return 0;
}catch(const std::exception& e){std::cerr<<"lookahead projection FAILED: "<<e.what()<<'\n';return 1;}}
