// SPDX-License-Identifier: Apache-2.0
#include "metal_context.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
using namespace Garnet;
namespace {
constexpr size_t T=16,H=32,K=8,D=128,QW=H*D,KW=K*D;
size_t cases=0,words=0,rejections=0,fp64=0,activeQueries=0;
void Need(bool ok,const std::string& why){if(!ok)throw std::runtime_error(why);}
std::vector<float> Values(size_t n,uint32_t seed) {
    std::vector<float> out(n);
    for(auto& x:out){seed=1664525u*seed+1013904223u;x=float(int((seed>>8)%65521)-32760)*.00001f;}
    return out;
}
void Equal(const std::vector<float>& a,const std::vector<float>& b,const std::string& why) {
    Need(a.size()==b.size(),why+" size differs");
    for(size_t i=0;i<a.size();++i)if(std::memcmp(&a[i],&b[i],sizeof(float))) {
        uint32_t actual=0,expected=0;std::memcpy(&actual,&a[i],4);std::memcpy(&expected,&b[i],4);
        throw std::runtime_error(why+" word"+std::to_string(i)+" differs: "+std::to_string(a[i])+" vs "+std::to_string(b[i])+" raw_uint32="+std::to_string(actual)+" vs "+std::to_string(expected));
    }
    words+=a.size();
}
template<class F>void Reject(F f,const std::string& why) {
    bool rejected=false;try{f();}catch(const std::invalid_argument&){rejected=true;}catch(const std::logic_error&){rejected=true;}
    Need(rejected,why);++rejections;
}
struct Job {
    LookaheadLayout profile;
    std::vector<LookaheadLayout::View> views;
    std::vector<float> q,k,v,kc,vc;
    Job(size_t b,size_t c,size_t w,size_t n,size_t g)
        :profile(b,c,w,n,g),q(Values(b*T*QW,13)),k(Values(b*T*KW,37)),v(Values(b*T*KW,97)),
         kc(Values(b*K*c*D,31)),vc(Values(b*K*c*D,71)){}
    void Poison() {
        const float nan=std::numeric_limits<float>::quiet_NaN();
        for(size_t row=0;row<profile.Batch();++row) {
            const LookaheadLayout::View* view=nullptr;
            for(const auto& v:views)if(v.Row()==row)view=&v;
            for(size_t t=0;t<T;++t)if(!view||!(view->Active()&(uint16_t(1)<<t))) {
                std::fill_n(q.data()+(row*T+t)*QW,QW,nan);
                std::fill_n(k.data()+(row*T+t)*KW,KW,nan);
                std::fill_n(v.data()+(row*T+t)*KW,KW,nan);
            }
            for(size_t head=0;head<K;++head) {
                const size_t first=view?view->Slot():0;
                std::fill_n(kc.data()+(row*K+head)*profile.Capacity()*D+first*D,(profile.Capacity()-first)*D,nan);
                std::fill_n(vc.data()+(row*K+head)*profile.Capacity()*D+first*D,(profile.Capacity()-first)*D,nan);
            }
        }
    }
};
std::vector<MetalMatrix> Upload(MetalContext& gpu,const Job& j) {
    const size_t b=j.profile.Batch(),c=j.profile.Capacity();
    return {gpu.Upload(j.q,b*T,QW),gpu.Upload(j.k,b*T,KW),gpu.Upload(j.v,b*T,KW),
        gpu.Upload(j.kc,b,K*c*D),gpu.Upload(j.vc,b,K*c*D)};
}
// Each active query is an independent ordinary decode row with its complete
// causal K/V sequence materialized by the CPU. The reference uses existing
// native score/value kernels, not either new lookahead shader.
struct Reference {std::vector<float> probabilities,output;};
Reference Ordinary(MetalContext& gpu,const Job& j) {
    const size_t c=j.profile.Capacity(),b=j.profile.Batch();
    Reference result{std::vector<float>(b*T*H*c),std::vector<float>(b*T*QW)};
    std::vector<size_t> flat;std::vector<float> q,k,v,slots;
    const float nan=std::numeric_limits<float>::quiet_NaN();
    for(const auto& view:j.views)for(size_t t=0;t<T;++t)if(view.Active()&(uint16_t(1)<<t)) {
        const size_t row=view.Row(),index=flat.size();flat.push_back(row*T+t);
        q.insert(q.end(),j.q.begin()+(row*T+t)*QW,j.q.begin()+(row*T+t+1)*QW);
        k.resize((index+1)*K*c*D,nan);v.resize(k.size(),nan);
        std::vector<size_t> local;
        for(size_t key=0;key<T;++key)if(j.profile.Sees(view,t,key))local.push_back(key);
        Need(view.Slot()+local.size()==j.profile.Position(view,t)+1,"CPU branch ancestor count differs");
        for(size_t head=0;head<K;++head) {
            const size_t dest=(index*K+head)*c*D,source=(row*K+head)*c*D;
            std::copy_n(j.kc.data()+source,view.Slot()*D,k.data()+dest);
            std::copy_n(j.vc.data()+source,view.Slot()*D,v.data()+dest);
            for(size_t a=0;a<local.size();++a) {
                const size_t at=(row*T+local[a])*KW+head*D,to=dest+(view.Slot()+a)*D;
                std::copy_n(j.k.data()+at,D,k.data()+to);std::copy_n(j.v.data()+at,D,v.data()+to);
            }
        }
        slots.push_back(float(view.Slot()+local.size()-1));
    }
    activeQueries+=flat.size();
    if(flat.empty())return result;
    const size_t count=flat.size();
    auto qg=gpu.Upload(q,count,QW),kg=gpu.Upload(k,count,K*c*D),vg=gpu.Upload(v,count,K*c*D),
        sg=gpu.Upload(slots,count,1),ag=gpu.Upload(std::vector<float>(count,1),count,1),
        gg=gpu.Upload(std::vector<float>(count*QW,0),count,QW),
        pg=gpu.Allocate(count,H*c),og=gpu.Allocate(count,QW);
    gpu.Begin();gpu.EncodeAttentionScores(qg,kg,sg,ag,pg,H,K,D,c);
    gpu.EncodeAttentionContext(pg,vg,gg,sg,og,H,K,D,c,false);gpu.SubmitAndWait();
    auto p=gpu.Download(pg),o=gpu.Download(og);
    for(size_t i=0;i<count;++i) {
        std::copy_n(p.data()+i*H*c,H*c,result.probabilities.data()+flat[i]*H*c);
        std::copy_n(o.data()+i*QW,QW,result.output.data()+flat[i]*QW);
        for(size_t head:{size_t(0),size_t(7),size_t(31)}) {
            const size_t kh=head/4,visible=size_t(slots[i])+1;
            std::vector<double> scores(visible);double maximum=-std::numeric_limits<double>::infinity();
            for(size_t key=0;key<visible;++key) {
                double score=0;for(size_t d=0;d<D;++d)score+=double(q[i*QW+head*D+d])*k[(i*K+kh)*c*D+key*D+d];
                scores[key]=score/std::sqrt(double(D));maximum=std::max(maximum,scores[key]);
            }
            double denominator=0;for(auto& s:scores){s=std::exp(s-maximum);denominator+=s;}
            for(size_t d:{size_t(0),size_t(63),size_t(127)}) {
                double expected=0;for(size_t key=0;key<visible;++key)expected+=scores[key]/denominator*v[(i*K+kh)*c*D+key*D+d];
                Need(std::abs(double(o[i*QW+head*D+d])-expected)<=3e-6+2e-5*std::abs(expected),"FP64 ordinary reference differs");++fp64;
            }
        }
    }
    Equal(gpu.Download(kg),k,"Ordinary reference changed key");Equal(gpu.Download(vg),v,"Ordinary reference changed value");
    return result;
}
void Case(MetalContext& gpu,Job j,size_t variant) {
    j.Poison();const auto expected=Ordinary(gpu,j);const auto originals=Upload(gpu,j);auto inputs=originals;
    const size_t rows=j.profile.Batch()*T,c=j.profile.Capacity();
    auto p=gpu.Upload(std::vector<float>(rows*H*c,.25f),rows,H*c),o=gpu.Upload(std::vector<float>(rows*QW,.5f),rows,QW);
    const size_t count=gpu.LookaheadAttentionPipelineCount();gpu.PrepareLookaheadAttention(j.profile);
    const size_t prepared=gpu.LookaheadAttentionPipelineCount();
    Need(prepared==count||prepared==count+2,"Prepared pair count differs");gpu.PrepareLookaheadAttention(j.profile);
    Need(gpu.LookaheadAttentionPipelineCount()==prepared,"Idempotent prepare added pipelines");
    auto pending=gpu.Allocate(rows,QW),zero=gpu.Upload(std::vector<float>(rows*QW,0),rows,QW);
    gpu.Begin();Need(gpu.LastLookaheadAttentionDispatches()==0,"Begin retained lookahead dispatch count");
    if(variant){gpu.EncodeAdd(inputs[0],zero,pending);inputs[0]=pending;if(variant==2)gpu.Flush();}
    gpu.EncodeLookaheadAttention(inputs,p,o,j.profile,j.views);
    Need(gpu.LastLookaheadAttentionDispatches()==1,"Lookahead pair missing");
    Reject([&]{gpu.Download(p);},"Pending probabilities readable");Reject([&]{gpu.Download(o);},"Pending context readable");
    Reject([&]{gpu.Begin();},"Second Begin accepted");Need(gpu.LastLookaheadAttentionDispatches()==1,"Invalid Begin reset counter");
    gpu.SubmitAndWait();Equal(gpu.Download(p),expected.probabilities,"Lookahead probabilities");Equal(gpu.Download(o),expected.output,"Lookahead context");
    const std::vector<std::vector<float>> before{j.q,j.k,j.v,j.kc,j.vc};
    for(size_t i=0;i<originals.size();++i)Equal(gpu.Download(originals[i]),before[i],"Read-only operand"+std::to_string(i));
    ++cases;
}
void Invalid(MetalContext& gpu,MetalContext& other) {
    Job j(2,128,4,4,1);j.views={j.profile.Request(0,7,9,19,1,3),j.profile.Request(1,11,3,8,1,2)};j.Poison();
    gpu.PrepareLookaheadAttention(j.profile);other.PrepareLookaheadAttention(j.profile);auto inputs=Upload(gpu,j);
    auto p=gpu.Upload(std::vector<float>(2*T*QW,.25f),2*T,QW),o=gpu.Upload(std::vector<float>(2*T*QW,.5f),2*T,QW);
    const auto beforeP=gpu.Download(p),beforeO=gpu.Download(o);
    Reject([&]{gpu.EncodeLookaheadAttention(inputs,p,o,j.profile,j.views);},"Encode without Begin accepted");
    for(size_t mutation=0;mutation<10;++mutation) {
        auto args=inputs;auto views=j.views;auto pp=p,oo=o;
        LookaheadLayout different(2,128,3,4,1);
        if(mutation==0)views[1]=views[0];
        if(mutation==1)views[1]=different.Request(1,11,3,8,1,2);
        if(mutation==2)args.pop_back();
        if(mutation==3)args[0]=MetalMatrix();
        if(mutation==4)args[0]=gpu.Allocate(2*T,QW);
        if(mutation==5)args[1]=gpu.Upload(std::vector<float>(KW,0),1,KW);
        if(mutation==6)oo=pp;
        if(mutation==7)oo=inputs[0];
        if(mutation==8)pp=inputs[0];
        gpu.Begin();
        if(mutation==9)Reject([&]{gpu.PrepareLookaheadAttention(j.profile);},"Prepare inside batch accepted");
        else Reject([&]{gpu.EncodeLookaheadAttention(args,pp,oo,j.profile,views);},"Invalid lookahead operands accepted");
        Need(gpu.LastLookaheadAttentionDispatches()==0,"Rejected operands dispatched");gpu.Abort();
        Equal(gpu.Download(p),beforeP,"Rejected operation changed probabilities");Equal(gpu.Download(o),beforeO,"Rejected operation changed context");
    }
    LookaheadLayout unprepared(2,127,4,4,1);
    gpu.Begin();Reject([&]{gpu.EncodeLookaheadAttention(inputs,p,o,unprepared,{});},"Unprepared profile accepted");gpu.Abort();
    gpu.Begin();gpu.EncodeLookaheadAttention(inputs,p,o,j.profile,j.views);
    other.Begin();Reject([&]{other.EncodeLookaheadAttention(inputs,p,o,j.profile,j.views);},"Foreign pending writer accepted");other.Abort();
    gpu.Abort();Need(gpu.LastLookaheadAttentionDispatches()==0,"Abort retained lookahead count");
    Reject([&]{gpu.Download(p);},"Aborted probabilities readable");Reject([&]{gpu.Download(o);},"Aborted context readable");
    // Rewriting the invalidated outputs in a fresh own batch must recover.
    gpu.Begin();gpu.EncodeLookaheadAttention(inputs,p,o,j.profile,j.views);gpu.SubmitAndWait();
    const auto expected=Ordinary(gpu,j);Equal(gpu.Download(p),expected.probabilities,"Abort recovery probabilities");Equal(gpu.Download(o),expected.output,"Abort recovery context");
    // A read-only operand produced by a different pending context is rejected.
    auto pending=other.Allocate(2*T,QW),zero=other.Upload(std::vector<float>(2*T*QW,0),2*T,QW);
    other.Begin();other.EncodeAdd(inputs[0],zero,pending);auto args=inputs;args[0]=pending;
    gpu.Begin();Reject([&]{gpu.EncodeLookaheadAttention(args,p,o,j.profile,j.views);},"Foreign pending input accepted");gpu.Abort();other.Abort();
    Equal(gpu.Download(p),expected.probabilities,"Rejected foreign input changed probabilities");
    const std::vector<std::vector<float>> before{j.q,j.k,j.v,j.kc,j.vc};
    for(size_t i=0;i<inputs.size();++i)Equal(gpu.Download(inputs[i]),before[i],"Invalid controls changed read-only operand");
}
}
int main(){try{
    {
        MetalContext gpu,other;
        for(size_t b:{size_t(1),size_t(4)})for(size_t n:{size_t(2),size_t(3),size_t(4)}) {
            const size_t guesses=n==4?1:4;
            for(size_t mode=0;mode<6;++mode) {
                Job j(b,37,4,n,guesses);
                for(size_t row=0;row<b;++row) {
                    if(mode==4&&row%2==0)continue;
                    const size_t slot=mode==1?35:(mode==2?0:3+row),remaining=mode==3?1:17;
                    j.views.push_back(j.profile.Request(row,19+row,slot,remaining,mode==4?0:guesses,mode==4?0:(mode==5?1:n-1)));
                }
                Case(gpu,std::move(j),cases%3);
            }
        }
        Job none(4,37,4,4,1);Case(gpu,std::move(none),0);
        Job one(4,1,4,4,1);one.views={one.profile.Request(3,91,0,1,1,3)};Case(gpu,std::move(one),0);
        Job large(4,4096,4,4,1);large.views={large.profile.Request(3,92,4090,20,1,3)};Case(gpu,std::move(large),2);
        Job last(4,4096,4,4,1);last.views={last.profile.Request(3,93,4095,20,1,3)};Case(gpu,std::move(last),0);
        // Exercise every admitted geometry in the actual specialized shader,
        // including narrow windows and several independent verification paths.
        for(size_t n=2;n<=4;++n)for(size_t w=1;w<=4;++w)for(size_t g=1;g<=4;++g) {
            if((n-1)*(w+g)>T)continue;
            Job shape(1,19,w,n,g);shape.views={shape.profile.Request(0,100+n*16+w*4+g,2,13,g,n-1)};
            Case(gpu,std::move(shape),cases%3);
        }
        Invalid(gpu,other);
    }
    const auto owners=MetalContext::ProfileOwnerStats();Need(!owners.matrices&&!owners.affineWeights&&!owners.bf16Weights&&!owners.pipelineSets,"Lookahead owners remain after close");
    std::cout<<"lookahead attention reference PASS cases="<<cases<<" active_queries="<<activeQueries<<" fp32_words="<<words<<" fp64_components="<<fp64<<" rejections="<<rejections
        <<" native_decode_bitwise=true readonly_all_operands=true poison_inactive_prefix_tail=true abort_pending_flush_foreign=true owners=0\n";return 0;
}catch(const std::exception& error){std::cerr<<"lookahead attention reference FAILED: "<<error.what()<<"\n";return 1;}}
