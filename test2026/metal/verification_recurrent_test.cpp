// SPDX-License-Identifier: Apache-2.0
// Correctness only: S1 capture and state-only accepted-prefix replay. No timings.
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
size_t checks=0,words=0,rejections=0,cases=0;
void Need(bool ok,const std::string& why){if(!ok)throw std::runtime_error(why);}
void Equal(const std::vector<float>& a,const std::vector<float>& b,const std::string& label) {
    Need(a.size()==b.size(),label+" size");
    for(size_t i=0;i<a.size();++i)if(std::memcmp(&a[i],&b[i],4))
        throw std::runtime_error(label+" word"+std::to_string(i)+" "+std::to_string(a[i])+" != "+std::to_string(b[i]));
    ++checks;words+=a.size();
}
template<class F>void Reject(F&& f,const std::string& why){bool bad=false;try{f();}catch(const std::exception&){bad=true;}Need(bad,"Missing rejection "+why);++rejections;}
std::vector<float> Values(size_t n,uint32_t seed,float scale=.00001f) {
    std::vector<float> v(n);for(auto& x:v){seed=1664525u*seed+1013904223u;x=float(int((seed>>8)%65521)-32760)*scale;}return v;
}
struct Job {
    size_t b,t,kh,h,kd=128,vd,kernel,c,qw,vw,pw,sw;
    std::vector<float> tokens,weights,conv,state,a,beta,alog,dt,mask;
    Job(size_t batch,size_t chunk,size_t keyHeads,size_t heads,size_t valueDim,size_t k,size_t mode)
        :b(batch),t(chunk),kh(keyHeads),h(heads),vd(valueDim),kernel(k),c(2*kh*128+h*vd),qw(h*128),vw(h*vd),pw(2*qw+vw+2*h),sw(qw*vd),
         tokens(Values(b*t*c,13)),weights(Values(c*k,31)),conv(Values(b*c*k,71,.000001f)),state(Values(b*sw,97,.000001f)),
         a(Values(b*t*h,43)),beta(Values(b*t*h,89)),alog(h,-.5f),dt(h,-.25f),mask(b*t) {
        for(size_t row=0;row<b;++row){size_t length=mode==0?t:(mode==1?0:(mode-2+row)%(t+1));
            for(size_t token=0;token<t;++token){mask[row*t+token]=float(token<length);if(token>=length){
                std::fill_n(tokens.data()+(row*t+token)*c,c,std::numeric_limits<float>::quiet_NaN());
                std::fill_n(a.data()+(row*t+token)*h,h,std::numeric_limits<float>::quiet_NaN());
                std::fill_n(beta.data()+(row*t+token)*h,h,std::numeric_limits<float>::quiet_NaN());}}
        }
        for(size_t i=0;i<conv.size();i+=97)conv[i]=-0.f;
        for(size_t i=0;i<state.size();i+=101)state[i]=-0.f;
    }
};
struct Reference {std::vector<float> convOut,cores;std::vector<std::vector<float>> conv,state;};
Reference Sequential(MetalContext& gpu,const Job& j) {
    Reference r{std::vector<float>(j.b*j.t*j.c),std::vector<float>(j.b*j.t*j.vw),{}, {}};
    auto cs=gpu.Upload(j.conv,j.b,j.c*j.kernel),s=gpu.Upload(j.state,j.b,j.sw),w=gpu.Upload(j.weights,j.c,j.kernel),
        al=gpu.Upload(j.alog,1,j.h),dt=gpu.Upload(j.dt,1,j.h);
    for(size_t t=0;t<j.t;++t) {
        std::vector<float> token(j.b*j.c),ap(j.b*j.h),bp(ap.size()),mask(j.b);
        for(size_t row=0;row<j.b;++row){std::copy_n(j.tokens.data()+(row*j.t+t)*j.c,j.c,token.data()+row*j.c);
            std::copy_n(j.a.data()+(row*j.t+t)*j.h,j.h,ap.data()+row*j.h);
            std::copy_n(j.beta.data()+(row*j.t+t)*j.h,j.h,bp.data()+row*j.h);mask[row]=j.mask[row*j.t+t];}
        auto x=gpu.Upload(token,j.b,j.c),a=gpu.Upload(ap,j.b,j.h),b=gpu.Upload(bp,j.b,j.h),m=gpu.Upload(mask,j.b,1);
        auto cv=gpu.Allocate(j.b,j.c*(j.kernel+1)),co=gpu.Allocate(j.b,j.c),nc=gpu.Allocate(j.b,j.c*j.kernel),p=gpu.Allocate(j.b,j.pw),
            q=gpu.Allocate(j.b,j.qw),k=gpu.Allocate(j.b,j.qw),v=gpu.Allocate(j.b,j.vw),g=gpu.Allocate(j.b,j.h),bt=gpu.Allocate(j.b,j.h),out=gpu.Allocate(j.b,j.vw),ns=gpu.Allocate(j.b,j.sw);
        gpu.Begin();gpu.EncodeCausalConv(x,w,cs,m,cv,j.kernel);
        gpu.EncodeSliceGroups(cv,co,j.c,j.kernel+1,0,1);gpu.EncodeSliceGroups(cv,nc,j.c,j.kernel+1,1,j.kernel);
        gpu.EncodeLinearPrepare(cv,a,b,al,dt,p,j.kh,j.h,128,j.vd,j.kernel);
        gpu.EncodeSliceGroups(p,q,1,j.pw,0,j.qw);gpu.EncodeSliceGroups(p,k,1,j.pw,j.qw,j.qw);
        gpu.EncodeSliceGroups(p,v,1,j.pw,2*j.qw,j.vw);gpu.EncodeSliceGroups(p,g,1,j.pw,2*j.qw+j.vw,j.h);
        gpu.EncodeSliceGroups(p,bt,1,j.pw,2*j.qw+j.vw+j.h,j.h);
        gpu.EncodeGatedDeltaSplit(q,k,v,g,bt,s,m,out,ns,j.h,128,j.vd);gpu.SubmitAndWait();
        auto coHost=gpu.Download(co),outHost=gpu.Download(out);
        for(size_t row=0;row<j.b;++row){std::copy_n(coHost.data()+row*j.c,j.c,r.convOut.data()+(row*j.t+t)*j.c);std::copy_n(outHost.data()+row*j.vw,j.vw,r.cores.data()+(row*j.t+t)*j.vw);}
        r.conv.push_back(gpu.Download(nc));r.state.push_back(gpu.Download(ns));cs=nc;s=ns;
    }
    return r;
}
void Case(MetalContext& gpu,const Job& j,size_t variant) {
    const auto expected=Sequential(gpu,j);
    auto tokens=gpu.Upload(j.tokens,j.b*j.t,j.c),weights=gpu.Upload(j.weights,j.c,j.kernel),cs=gpu.Upload(j.conv,j.b,j.c*j.kernel),
        state=gpu.Upload(j.state,j.b,j.sw),ap=gpu.Upload(j.a,j.b*j.t,j.h),bp=gpu.Upload(j.beta,j.b*j.t,j.h),
        al=gpu.Upload(j.alog,1,j.h),dt=gpu.Upload(j.dt,1,j.h),mask=gpu.Upload(j.mask,j.b,j.t);
    auto co=gpu.Allocate(j.b*j.t,j.c),nc=gpu.Allocate(j.b,j.c*j.kernel),c1=gpu.Allocate(j.b,j.c*j.kernel),
        prepared=gpu.Allocate(j.b*j.t,j.pw),cores=gpu.Allocate(j.b*j.t,j.vw),ns=gpu.Allocate(j.b,j.sw),s1=gpu.Allocate(j.b,j.sw);
    std::vector<MetalMatrix> rc,rs,rm,rcScratch,rsScratch;std::vector<std::vector<float>> ec,es;
    auto zero=gpu.Upload(std::vector<float>(j.b*j.t*j.c,0),j.b*j.t,j.c),pending=gpu.Allocate(j.b*j.t,j.c);
    // Every mask and workspace is prepared while the command batch is closed.
    for(size_t accepted=0;accepted<j.t;++accepted) {
        std::vector<float> flags(j.b*j.t,0),convExpected(j.b*j.c*j.kernel),stateExpected(j.b*j.sw);
        for(size_t row=0;row<j.b;++row){size_t length=0;while(length<j.t&&j.mask[row*j.t+length]!=0)++length;
            const size_t draft=length?(accepted+row)%length:0;
            for(size_t t=1;t<=draft;++t)flags[row*j.t+t]=1;
            std::copy_n(expected.conv[draft].data()+row*j.c*j.kernel,j.c*j.kernel,convExpected.data()+row*j.c*j.kernel);
            std::copy_n(expected.state[draft].data()+row*j.sw,j.sw,stateExpected.data()+row*j.sw);}
        rm.push_back(gpu.Upload(flags,j.b,j.t));rcScratch.push_back(gpu.Allocate(j.b*j.t,j.c));rsScratch.push_back(gpu.Allocate(j.b*j.t,j.vw));
        rc.push_back(gpu.Allocate(j.b,j.c*j.kernel));rs.push_back(gpu.Allocate(j.b,j.sw));
        ec.push_back(std::move(convExpected));es.push_back(std::move(stateExpected));
    }
    const auto before=MetalContext::ProfileOwnerStats();
    // Keep the original projection owner alive when rebinding the local input
    // to a GPU producer, so owner-count equality measures encoding alone.
    const auto originalTokens=tokens;
    gpu.Begin();Need(gpu.LastVerificationConvCheckpoints()==0&&gpu.LastVerificationDeltaCheckpoints()==0,"Begin counter reset");
    if(variant==1){gpu.EncodeAdd(tokens,zero,pending);tokens=pending;}
    gpu.EncodeVerificationConvCheckpoint(tokens,weights,cs,mask,co,nc,c1,j.t,j.kernel);
    if(variant==2)gpu.Flush();
    gpu.EncodeLinearPrepare(co,ap,bp,al,dt,prepared,j.kh,j.h,128,j.vd,0);
    gpu.EncodeVerificationDeltaCheckpoint(prepared,state,mask,cores,ns,s1,j.t,j.h,128,j.vd);
    Need(gpu.LastVerificationConvCheckpoints()==1&&gpu.LastVerificationDeltaCheckpoints()==1,"Checkpoint dispatch coverage");
    Reject([&]{gpu.Begin();},"nested Begin");
    Need(gpu.LastVerificationConvCheckpoints()==1&&gpu.LastVerificationDeltaCheckpoints()==1,"Invalid Begin changed counters");
    for(const auto* output:{&co,&nc,&c1,&cores,&ns,&s1})Reject([&]{gpu.Download(*output);},"pending checkpoint readback");
    // Replay state only from S1 and saved QKV/prepared rows. No matrix products,
    // host state upload, or CPU readback is involved in these GPU dependencies.
    for(size_t accepted=0;accepted<j.t;++accepted) {
        gpu.EncodeCausalConvChunk(tokens,weights,c1,rm[accepted],rcScratch[accepted],rc[accepted],j.t,j.kernel);
        gpu.EncodeGatedDeltaChunk(prepared,s1,rm[accepted],rsScratch[accepted],rs[accepted],j.t,j.h,128,j.vd);
    }
    const auto after=MetalContext::ProfileOwnerStats();
    Need(before.matrices==after.matrices&&before.pipelineSets==after.pipelineSets,"Encoding allocated native workspace/pipeline owners");++checks;
    gpu.SubmitAndWait();Equal(gpu.Download(co),expected.convOut,"all convolution outputs");Equal(gpu.Download(cores),expected.cores,"all recurrent outputs");
    Equal(gpu.Download(nc),expected.conv.back(),"full convolution state");Equal(gpu.Download(ns),expected.state.back(),"full GDN state");
    Equal(gpu.Download(c1),expected.conv[0],"S1 convolution state");Equal(gpu.Download(s1),expected.state[0],"S1 GDN state");
    for(size_t i=0;i<rc.size();++i){Equal(gpu.Download(rc[i]),ec[i],"accepted convolution prefix");Equal(gpu.Download(rs[i]),es[i],"accepted GDN prefix");}
    ++cases;
}
void ConvTail(MetalContext& gpu) {
    constexpr size_t b=2,t=2,c=37,k=3;
    const auto tokens=Values(b*t*c,17),weights=Values(c*k,71),initial=Values(b*c*k,43);
    const std::vector<float> flags={1,1,0,0};
    auto x=gpu.Upload(tokens,b*t,c),w=gpu.Upload(weights,c,k),s=gpu.Upload(initial,b,c*k),mask=gpu.Upload(flags,b,t),
        out=gpu.Allocate(b*t,c),next=gpu.Allocate(b,c*k),first=gpu.Allocate(b,c*k);
    gpu.Begin();gpu.EncodeVerificationConvCheckpoint(x,w,s,mask,out,next,first,t,k);gpu.SubmitAndWait();
    std::vector<float> expected(b*t*c),firstHost,finalHost;auto state=s;
    for(size_t token=0;token<t;++token) {
        std::vector<float> data(b*c),active(b);
        for(size_t row=0;row<b;++row){std::copy_n(tokens.data()+(row*t+token)*c,c,data.data()+row*c);active[row]=flags[row*t+token];}
        auto input=gpu.Upload(data,b,c),m=gpu.Upload(active,b,1),packed=gpu.Allocate(b,c*(k+1)),co=gpu.Allocate(b,c),ns=gpu.Allocate(b,c*k);
        gpu.Begin();gpu.EncodeCausalConv(input,w,state,m,packed,k);gpu.EncodeSliceGroups(packed,co,c,k+1,0,1);gpu.EncodeSliceGroups(packed,ns,c,k+1,1,k);gpu.SubmitAndWait();
        auto result=gpu.Download(co);for(size_t row=0;row<b;++row)std::copy_n(result.data()+row*c,c,expected.data()+(row*t+token)*c);
        finalHost=gpu.Download(ns);if(!token)firstHost=finalHost;state=ns;
    }
    Equal(gpu.Download(out),expected,"tail37 convolution outputs");Equal(gpu.Download(next),finalHost,"tail37 final state");Equal(gpu.Download(first),firstHost,"tail37 first state");++cases;
}
void Invalid(MetalContext& gpu) {
    Job j(1,4,1,1,16,4,0);
    auto tok=gpu.Upload(j.tokens,4,j.c),w=gpu.Upload(j.weights,j.c,4),cs=gpu.Upload(j.conv,1,j.c*4),
        state=gpu.Upload(j.state,1,j.sw),mask=gpu.Upload(j.mask,1,4),prepared=gpu.Upload(Values(4*j.pw,13),4,j.pw);
    auto co=gpu.Upload(std::vector<float>(4*j.c,-99),4,j.c),nc=gpu.Upload(std::vector<float>(j.c*4,-99),1,j.c*4),c1=gpu.Upload(std::vector<float>(j.c*4,-99),1,j.c*4),
        out=gpu.Upload(std::vector<float>(4*j.vw,-99),4,j.vw),ns=gpu.Upload(std::vector<float>(j.sw,-99),1,j.sw),s1=gpu.Upload(std::vector<float>(j.sw,-99),1,j.sw);
    auto uninit=gpu.Allocate(4,j.c),uninitPrepared=gpu.Allocate(4,j.pw);
    auto zeroMask=gpu.Upload(std::vector<float>(4,0),1,4),pendingMask=gpu.Allocate(1,4);
    auto conv=[&](const MetalMatrix& x,const MetalMatrix& m,const MetalMatrix& o,const MetalMatrix& n,const MetalMatrix& c,size_t t=4,size_t k=4){gpu.EncodeVerificationConvCheckpoint(x,w,cs,m,o,n,c,t,k);};
    auto delta=[&](const MetalMatrix& p,const MetalMatrix& m,const MetalMatrix& o,const MetalMatrix& n,const MetalMatrix& c,size_t t=4,size_t h=1,size_t kd=128,size_t vd=16){gpu.EncodeVerificationDeltaCheckpoint(p,state,m,o,n,c,t,h,kd,vd);};
    Reject([&]{conv(tok,mask,co,nc,c1);},"conv without Begin");Reject([&]{delta(prepared,mask,out,ns,s1);},"delta without Begin");
    for(auto flags:std::vector<std::vector<float>>{{0,1,1,0},{1,0,1,0},{1,.5,0,0},{1,-1,0,0},{1,INFINITY,0,0},{1,NAN,0,0}}) {
        auto m=gpu.Upload(flags,1,4);gpu.Begin();Reject([&]{conv(tok,m,co,nc,c1);},"invalid conv mask");Reject([&]{delta(prepared,m,out,ns,s1);},"invalid delta mask");
        Need(!gpu.LastVerificationConvCheckpoints()&&!gpu.LastVerificationDeltaCheckpoints(),"Rejected call changed counters");gpu.Abort();
        Equal(gpu.Download(c1),std::vector<float>(j.c*4,-99),"rejected S1 conv unchanged");Equal(gpu.Download(s1),std::vector<float>(j.sw,-99),"rejected S1 GDN unchanged");
    }
    gpu.Begin();
    Reject([&]{conv(tok,mask,tok,nc,c1);},"conv input alias");Reject([&]{conv(tok,mask,co,cs,c1);},"conv state alias");
    Reject([&]{conv(tok,mask,co,nc,nc);},"conv output alias");Reject([&]{conv(tok,mask,co,nc,cs);},"conv checkpoint alias");
    Reject([&]{delta(prepared,mask,out,state,s1);},"delta state alias");Reject([&]{delta(prepared,mask,out,ns,ns);},"delta output alias");
    Reject([&]{delta(prepared,mask,out,ns,state);},"delta checkpoint alias");
    for(size_t t:{0,5,8}){Reject([&]{conv(tok,mask,co,nc,c1,t);},"conv temporal bound");Reject([&]{delta(prepared,mask,out,ns,s1,t);},"delta temporal bound");}
    for(size_t k:{0,9})Reject([&]{conv(tok,mask,co,nc,c1,4,k);},"conv kernel bound");
    for(size_t h:{0,49})Reject([&]{delta(prepared,mask,out,ns,s1,4,h);},"delta head bound");
    for(size_t vd:{0,17,144})Reject([&]{delta(prepared,mask,out,ns,s1,4,1,128,vd);},"delta width bound");
    Reject([&]{delta(prepared,mask,out,ns,s1,4,1,64,16);},"delta key bound");
    auto empty=MetalMatrix();Reject([&]{conv(tok,mask,co,nc,empty);},"missing conv checkpoint");Reject([&]{delta(prepared,mask,out,ns,empty);},"missing delta checkpoint");
    Reject([&]{conv(uninit,mask,co,nc,c1);},"uninitialized conv input");Reject([&]{delta(uninitPrepared,mask,out,ns,s1);},"uninitialized prepared input");
    gpu.EncodeAdd(mask,zeroMask,pendingMask);
    Reject([&]{conv(tok,pendingMask,co,nc,c1);},"pending own conv mask");Reject([&]{delta(prepared,pendingMask,out,ns,s1);},"pending own delta mask");gpu.Abort();
    Equal(gpu.Download(co),std::vector<float>(4*j.c,-99),"all rejected conv output unchanged");Equal(gpu.Download(nc),std::vector<float>(j.c*4,-99),"all rejected conv final unchanged");
    Equal(gpu.Download(out),std::vector<float>(4*j.vw,-99),"all rejected delta output unchanged");Equal(gpu.Download(ns),std::vector<float>(j.sw,-99),"all rejected delta final unchanged");
    MetalContext other;auto zeros=gpu.Upload(std::vector<float>(j.sw,0),1,j.sw),foreign=other.Allocate(1,j.sw),
        zt=gpu.Upload(std::vector<float>(4*j.c,0),4,j.c),zp=gpu.Upload(std::vector<float>(4*j.pw,0),4,j.pw),
        foreignTokens=other.Allocate(4,j.c),foreignPrepared=other.Allocate(4,j.pw);
    other.Begin();other.EncodeAdd(state,zeros,foreign);other.EncodeAdd(tok,zt,foreignTokens);other.EncodeAdd(prepared,zp,foreignPrepared);
    gpu.Begin();Reject([&]{delta(prepared,mask,out,ns,foreign);},"foreign checkpoint output");
    Reject([&]{delta(prepared,mask,out,foreign,s1);},"foreign final output");
    Reject([&]{conv(foreignTokens,mask,co,nc,c1);},"foreign projected input");
    Reject([&]{delta(foreignPrepared,mask,out,ns,s1);},"foreign prepared input");gpu.Abort();other.Abort();
    Equal(gpu.Download(c1),std::vector<float>(j.c*4,-99),"foreign reject S1 conv unchanged");Equal(gpu.Download(s1),std::vector<float>(j.sw,-99),"foreign reject S1 GDN unchanged");
    Job wide(2,4,1,1,16,4,0);
    auto wx=gpu.Upload(wide.tokens,8,j.c),ws=gpu.Upload(wide.conv,2,j.c*4),wm=gpu.Upload(wide.mask,2,4),
        wo=gpu.Allocate(8,j.c),wn=gpu.Allocate(2,j.c*4),wc=gpu.Allocate(2,j.c*4),
        wp=gpu.Upload(Values(8*j.pw,71),8,j.pw),wd=gpu.Upload(wide.state,2,j.sw),
        wdo=gpu.Allocate(8,j.vw),wdn=gpu.Allocate(2,j.sw),wdc=gpu.Allocate(2,j.sw);
    gpu.Begin();Reject([&]{gpu.EncodeVerificationConvCheckpoint(wx,w,ws,wm,wo,wn,wc,4,4);},"eight physical conv rows");
    Reject([&]{gpu.EncodeVerificationDeltaCheckpoint(wp,wd,wm,wdo,wdn,wdc,4,1,128,16);},"eight physical delta rows");gpu.Abort();
    for(bool partial:{false,true}) {
        gpu.Begin();conv(tok,mask,co,nc,c1);delta(prepared,mask,out,ns,s1);
        Need(gpu.LastVerificationConvCheckpoints()==1&&gpu.LastVerificationDeltaCheckpoints()==1,"recovery dispatches");
        if(partial)gpu.Flush();gpu.Abort();
        Need(!gpu.LastVerificationConvCheckpoints()&&!gpu.LastVerificationDeltaCheckpoints(),"Abort counter reset");
        for(auto* m:{&co,&nc,&c1,&out,&ns,&s1})Reject([&]{gpu.Download(*m);},"aborted output readable");
    }
    gpu.Begin();conv(tok,mask,co,nc,c1);delta(prepared,mask,out,ns,s1);gpu.SubmitAndWait();
    Need(gpu.Download(c1).size()==j.c*4&&gpu.Download(s1).size()==j.sw,"retry after abort");++checks;
}
}
int main(){try{
    {
        MetalContext gpu;
        for(auto shape:std::vector<std::pair<size_t,size_t>>{{1,1},{1,2},{1,3},{1,4},{2,1},{2,2},{4,1}})
            for(size_t vd:{16,32,128})for(size_t mode=0;mode<shape.second+3;++mode)
                Case(gpu,Job(shape.first,shape.second,1,3,vd,4,mode),mode%3);
        for(size_t t:{1,2,3,4})for(size_t mode=0;mode<t+3;++mode)
            Case(gpu,Job(1,t,16,48,128,4,mode),mode%3);
        for(size_t k:{1,8})Case(gpu,Job(1,4,1,1,16,k,3),2);
        ConvTail(gpu);
        Invalid(gpu);
    }
    auto owners=MetalContext::ProfileOwnerStats();Need(!owners.matrices&&!owners.affineWeights&&!owners.bf16Weights&&!owners.pipelineSets,"Owners remain after checkpoint gate");
    std::cout<<"verification_recurrent_checkpoint PASS cases="<<cases<<" checks="<<checks<<" bitwise_words="<<words<<" rejections="<<rejections<<" owners=0; no model speed claim\n";return 0;
}catch(const std::exception& error){std::cerr<<"verification_recurrent_checkpoint FAILED: "<<error.what()<<"\n";return 1;}}
