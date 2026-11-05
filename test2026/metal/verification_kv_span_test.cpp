// SPDX-License-Identifier: Apache-2.0
// Exact KV backup/restore and independent ordinary-prefix reference; no timings.
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
size_t cases=0,checks=0,words=0,rejections=0;
void Need(bool ok,const std::string& why){if(!ok)throw std::runtime_error(why);}
void Equal(const std::vector<float>& a,const std::vector<float>& b,const std::string& why) {
    Need(a.size()==b.size(),why+" size");
    if(std::memcmp(a.data(),b.data(),a.size()*4))for(size_t i=0;i<a.size();++i)if(std::memcmp(&a[i],&b[i],4)) {
        uint32_t x,y;std::memcpy(&x,&a[i],4);std::memcpy(&y,&b[i],4);
        throw std::runtime_error(why+" word"+std::to_string(i)+" bits"+std::to_string(x)+" != "+std::to_string(y));
    }
    ++checks;words+=a.size();
}
template<class F>void Reject(F&& f,const std::string& why){bool bad=false;try{f();}catch(const std::exception&){bad=true;}Need(bad,"Missing rejection: "+why);++rejections;}
std::vector<float> Raw(size_t n,uint32_t seed) {
    std::vector<float> out(n);
    const uint32_t special[]={0,0x80000000u,0x7fc12345u,0xffc54321u,0x7fa12345u,0xffa54321u,0x7f800000u,0xff800000u,1,0x80000001u};
    for(size_t i=0;i<n;++i){seed=1664525u*seed+1013904223u;uint32_t bits=i%17<10?special[i%17]:seed;std::memcpy(out.data()+i,&bits,4);}return out;
}
struct Job {
    size_t b,t,h,d,c,w,cw;std::vector<float> initial,tokens,slots,submitted;
    Job(size_t batch,size_t chunk,size_t heads,size_t dim,size_t capacity,size_t base,size_t mode)
      :b(batch),t(chunk),h(heads),d(dim),c(capacity),w(h*d),cw(h*c*d),initial(Raw(b*cw,17)),tokens(Raw(b*t*w,71)),slots(b*t),submitted(b*t) {
        for(size_t row=0;row<b;++row){const size_t length=mode==0?t:(mode==1?0:(mode-2+row)%(t+1));
            const size_t first=base-std::min(base,row*2);
            for(size_t tok=0;tok<t;++tok){submitted[row*t+tok]=float(tok<length);slots[row*t+tok]=tok<length?float(first+tok):std::numeric_limits<float>::quiet_NaN();}}
    }
};
std::vector<float> Expected(const Job& j,const std::vector<float>& accepted) {
    auto result=j.initial;
    for(size_t row=0;row<j.b;++row)for(size_t t=0;t<j.t;++t)if(accepted[row*j.t+t])
        for(size_t h=0;h<j.h;++h)for(size_t d=0;d<j.d;++d)
            result[row*j.cw+(h*j.c+size_t(j.slots[row*j.t+t]))*j.d+d]=j.tokens[(row*j.t+t)*j.w+h*j.d+d];
    return result;
}
std::vector<float> Sequential(MetalContext& gpu,const Job& j,const std::vector<float>& accepted) {
    auto cache=gpu.Upload(j.initial,j.b,j.cw);
    for(size_t t=0;t<j.t;++t) {
        std::vector<float> token(j.b*j.w),slots(j.b),mask(j.b);
        for(size_t row=0;row<j.b;++row){std::copy_n(j.tokens.data()+(row*j.t+t)*j.w,j.w,token.data()+row*j.w);mask[row]=accepted[row*j.t+t];slots[row]=mask[row]?j.slots[row*j.t+t]:0;}
        auto x=gpu.Upload(token,j.b,j.w),s=gpu.Upload(slots,j.b,1),m=gpu.Upload(mask,j.b,1),next=gpu.Allocate(j.b,j.cw);
        gpu.Begin();gpu.EncodeKVUpdate(x,cache,s,m,next,j.h,j.d,j.c);gpu.SubmitAndWait();cache=next;
    }
    return gpu.Download(cache);
}
void Speculate(MetalContext& gpu,const Job& j,const MetalMatrix& tokens,const MetalMatrix& cache,
    const MetalMatrix& slots,const MetalMatrix& submitted,const MetalMatrix& scratch) {
#if defined(GARNET_METAL_RESIDENT_KV_INPLACE)&&defined(GARNET_METAL_CHUNK_KV_INPLACE)
    gpu.EncodeKVChunk(tokens,cache,slots,submitted,cache,j.h,j.d,j.c,j.t,true);
#else
    // The span primitive itself needs no in-place-update build extension.
    gpu.EncodeKVChunk(tokens,cache,slots,submitted,scratch,j.h,j.d,j.c,j.t,true);
    gpu.EncodeLayout(scratch,scratch,cache,0);
#endif
}
void Case(MetalContext& gpu,const Job& j,size_t keep,size_t variant) {
    std::vector<float> accepted(j.b*j.t,0),backupExpected(j.b*j.t*j.w,0);
    for(size_t row=0;row<j.b;++row){size_t length=0;while(length<j.t&&j.submitted[row*j.t+length])++length;
        const size_t count=keep==j.t?length:std::min(length,(keep+row)%(j.t+1));
        for(size_t t=0;t<j.t;++t){accepted[row*j.t+t]=float(t<count);if(j.submitted[row*j.t+t])
            for(size_t h=0;h<j.h;++h)for(size_t d=0;d<j.d;++d)
                backupExpected[(row*j.t+t)*j.w+h*j.d+d]=j.initial[row*j.cw+(h*j.c+size_t(j.slots[row*j.t+t]))*j.d+d];}}
    const auto expected=Expected(j,accepted);Equal(Sequential(gpu,j,accepted),expected,"ordinary prefix vs CPU reference");
    auto original=gpu.Upload(j.initial,j.b,j.cw),cache=gpu.Upload(j.initial,j.b,j.cw),tokens=gpu.Upload(j.tokens,j.b*j.t,j.w),
        slots=gpu.Upload(j.slots,j.b,j.t),submitted=gpu.Upload(j.submitted,j.b,j.t),flags=gpu.Upload(accepted,j.b,j.t),
        backup=gpu.Allocate(j.b*j.t,j.w),consumer=gpu.Allocate(j.b,j.cw),scratch=gpu.Allocate(j.b,j.cw);
    if(keep==j.t&&variant==0)flags=submitted; // Read-only controls may alias.
    const auto before=MetalContext::ProfileOwnerStats();
    gpu.Begin();Need(!gpu.LastVerificationKVBackups()&&!gpu.LastVerificationKVRestores(),"Begin reset counters");
    if(variant==1)gpu.EncodeLayout(original,original,cache,0); // Own pending cache producer.
    gpu.EncodeVerificationKVBackup(cache,slots,submitted,backup,j.h,j.d,j.c,j.t);
    if(variant==2)gpu.Flush();
    Speculate(gpu,j,tokens,cache,slots,submitted,scratch);
    gpu.EncodeVerificationKVRestore(backup,slots,submitted,flags,cache,j.h,j.d,j.c,j.t);
    if(variant==3)gpu.Flush();
    gpu.EncodeLayout(cache,cache,consumer,0); // Consume restored state before any CPU readback.
    Need(gpu.LastVerificationKVBackups()==1&&gpu.LastVerificationKVRestores()==1,"Native span dispatch counters");
    Reject([&]{gpu.Begin();},"nested Begin");Need(gpu.LastVerificationKVBackups()==1&&gpu.LastVerificationKVRestores()==1,"Invalid Begin changed counters");
    for(const auto* value:{&cache,&backup,&consumer})Reject([&]{gpu.Download(*value);},"pending span readback");
    const auto after=MetalContext::ProfileOwnerStats();Need(before.matrices==after.matrices&&before.pipelineSets==after.pipelineSets,"Encoding grew native workspace/pipeline owners");++checks;
    gpu.SubmitAndWait();Equal(gpu.Download(backup),backupExpected,"raw backup including inactive zeros");
    Equal(gpu.Download(cache),expected,"entire restored cache");Equal(gpu.Download(consumer),expected,"GPU restored-cache consumer");
    Equal(gpu.Download(original),j.initial,"other original cache unchanged");++cases;
}
void Carried(MetalContext& gpu) {
    Job j(1,4,4,256,64,2,0);auto expected=j.initial;
    auto cache=gpu.Upload(j.initial,1,j.cw),tokens=gpu.Upload(j.tokens,4,j.w),submitted=gpu.Upload(j.submitted,1,4),
        s0=gpu.Upload(std::vector<float>{2,3,4,5},1,4),s1=gpu.Upload(std::vector<float>{4,5,6,7},1,4),
        a0=gpu.Upload(std::vector<float>{1,1,0,0},1,4),a1=gpu.Upload(std::vector<float>{1,1,1,0},1,4),
        backup0=gpu.Allocate(4,j.w),backup1=gpu.Allocate(4,j.w),scratch=gpu.Allocate(1,j.cw);
    auto b0=std::vector<float>(4*j.w),b1=b0;
    for(size_t step=0;step<2;++step){size_t start=step?4:2,count=step?3:2;
        for(size_t t=0;t<4;++t)for(size_t h=0;h<j.h;++h)for(size_t d=0;d<j.d;++d)
            (step?b1:b0)[t*j.w+h*j.d+d]=expected[(h*j.c+start+t)*j.d+d];
        for(size_t t=0;t<count;++t)for(size_t h=0;h<j.h;++h)for(size_t d=0;d<j.d;++d)
            expected[(h*j.c+start+t)*j.d+d]=j.tokens[t*j.w+h*j.d+d];}
    gpu.Begin();gpu.EncodeVerificationKVBackup(cache,s0,submitted,backup0,j.h,j.d,j.c,4);
    Speculate(gpu,j,tokens,cache,s0,submitted,scratch);
    gpu.EncodeVerificationKVRestore(backup0,s0,submitted,a0,cache,j.h,j.d,j.c,4);
    gpu.Flush();gpu.EncodeVerificationKVBackup(cache,s1,submitted,backup1,j.h,j.d,j.c,4);
    Speculate(gpu,j,tokens,cache,s1,submitted,scratch);
    gpu.EncodeVerificationKVRestore(backup1,s1,submitted,a1,cache,j.h,j.d,j.c,4);gpu.SubmitAndWait();
    Need(gpu.LastVerificationKVBackups()==2&&gpu.LastVerificationKVRestores()==2,"Carried span coverage");
    Equal(gpu.Download(cache),expected,"two carried span transactions");Equal(gpu.Download(backup0),b0,"first carried snapshot");Equal(gpu.Download(backup1),b1,"second snapshot consumes restored cache");++cases;
}
void Invalid(MetalContext& gpu) {
    Job j(1,4,1,7,8,1,0);
    auto cache=gpu.Upload(j.initial,1,j.cw),slots=gpu.Upload(j.slots,1,4),mask=gpu.Upload(j.submitted,1,4),
        backup=gpu.Upload(Raw(4*j.w,89),4,j.w),accepted=gpu.Upload(std::vector<float>{1,1,0,0},1,4);
    const auto oldBackup=gpu.Download(backup);
    auto gather=[&](const MetalMatrix& c,const MetalMatrix& s,const MetalMatrix& m,const MetalMatrix& b,size_t h=1,size_t d=7,size_t cap=8,size_t t=4){gpu.EncodeVerificationKVBackup(c,s,m,b,h,d,cap,t);};
    auto restore=[&](const MetalMatrix& b,const MetalMatrix& s,const MetalMatrix& m,const MetalMatrix& a,const MetalMatrix& c,size_t h=1,size_t d=7,size_t cap=8,size_t t=4){gpu.EncodeVerificationKVRestore(b,s,m,a,c,h,d,cap,t);};
    auto unchanged=[&]{Equal(gpu.Download(cache),j.initial,"rejected restore unchanged");Equal(gpu.Download(backup),oldBackup,"rejected backup unchanged");};
    Reject([&]{gather(cache,slots,mask,backup);},"backup outside Begin");Reject([&]{restore(backup,slots,mask,accepted,cache);},"restore outside Begin");
    for(auto data:std::vector<std::vector<float>>{{1,1,3,4},{1,3,4,5},{2,1,3,4},{-1,2,3,4},{1.5,2,3,4},{5,6,7,8},{NAN,2,3,4},{INFINITY,2,3,4}}) {
        auto s=gpu.Upload(data,1,4);gpu.Begin();Reject([&]{gather(cache,s,mask,backup);},"bad slots backup");Reject([&]{restore(backup,s,mask,accepted,cache);},"bad slots restore");gpu.Abort();unchanged();}
    for(auto data:std::vector<std::vector<float>>{{0,1,0,0},{1,0,1,0},{1,.5,0,0},{1,-1,0,0},{1,NAN,0,0},{1,INFINITY,0,0}}) {
        auto m=gpu.Upload(data,1,4);gpu.Begin();Reject([&]{gather(cache,slots,m,backup);},"bad submitted mask");Reject([&]{restore(backup,slots,mask,m,cache);},"bad accepted mask");gpu.Abort();unchanged();}
    auto shortMask=gpu.Upload(std::vector<float>{1,0,0,0},1,4),uninitCache=gpu.Allocate(1,j.cw),uninitBackup=gpu.Allocate(4,j.w),
        badShape=gpu.Allocate(1,3),zero=gpu.Upload(std::vector<float>(4,0),1,4),pending=gpu.Allocate(1,4);
    gpu.Begin();Reject([&]{restore(backup,slots,shortMask,accepted,cache);},"accepted longer than submitted");
    for(size_t h:{0,9}){Reject([&]{gather(cache,slots,mask,backup,h);},"heads backup");Reject([&]{restore(backup,slots,mask,accepted,cache,h);},"heads restore");}
    for(size_t d:{0,257})Reject([&]{gather(cache,slots,mask,backup,1,d);},"dimension bound");
    for(size_t c:{0,4097})Reject([&]{restore(backup,slots,mask,accepted,cache,1,7,c);},"capacity bound");
    for(size_t t:{0,5}){Reject([&]{gather(cache,slots,mask,backup,1,7,8,t);},"chunk bound backup");Reject([&]{restore(backup,slots,mask,accepted,cache,1,7,8,t);},"chunk bound restore");}
    Reject([&]{gather(uninitCache,slots,mask,backup);},"uninitialized source cache");Reject([&]{restore(backup,slots,mask,accepted,uninitCache);},"uninitialized restore bank");
    Reject([&]{restore(uninitBackup,slots,mask,accepted,cache);},"uninitialized backup input");
    Reject([&]{gather(cache,badShape,mask,backup);},"slot shape mismatch");Reject([&]{restore(backup,slots,mask,badShape,cache);},"accepted shape mismatch");
    MetalMatrix empty;Reject([&]{gather(cache,slots,mask,empty);},"empty backup");Reject([&]{restore(backup,slots,mask,empty,cache);},"empty accepted mask");
    gpu.EncodeAdd(mask,zero,pending);
    Reject([&]{gather(cache,pending,mask,backup);},"pending own slots");Reject([&]{gather(cache,slots,pending,backup);},"pending own submitted");
    Reject([&]{restore(backup,slots,mask,pending,cache);},"pending own accepted");gpu.Abort();unchanged();
    // Degenerate shapes deliberately reach storage-alias checks, not shape errors.
    auto one=gpu.Upload(std::vector<float>{0},1,1),bank=gpu.Upload(std::vector<float>{-99},1,1);
    gpu.Begin();Reject([&]{gather(one,one,one,one,1,1,1,1);},"cache backup alias");
    Reject([&]{gather(one,one,one,bank,1,1,1,1);},"data control alias");
    Reject([&]{restore(one,bank,bank,bank,one,1,1,1,1);},"restore backup cache alias");gpu.Abort();
    // Correct eight-row matrices still exceed this verified physical envelope.
    auto wideCache=gpu.Upload(Raw(2*j.cw,17),2,j.cw),wideBackup=gpu.Upload(Raw(8*j.w,31),8,j.w),
        wideSlots=gpu.Upload(std::vector<float>{1,2,3,4,1,2,3,4},2,4),wideMask=gpu.Upload(std::vector<float>(8,1),2,4);
    gpu.Begin();Reject([&]{gather(wideCache,wideSlots,wideMask,wideBackup);},"eight rows backup");Reject([&]{restore(wideBackup,wideSlots,wideMask,wideMask,wideCache);},"eight rows restore");gpu.Abort();
    MetalContext other;auto fCache=other.Allocate(1,j.cw),fBackup=other.Allocate(4,j.w),
        zc=gpu.Upload(std::vector<float>(j.cw,0),1,j.cw),zb=gpu.Upload(std::vector<float>(4*j.w,0),4,j.w);
    other.Begin();other.EncodeAdd(cache,zc,fCache);other.EncodeAdd(backup,zb,fBackup);gpu.Begin();
    Reject([&]{gather(fCache,slots,mask,backup);},"foreign cache input");Reject([&]{gather(cache,slots,mask,fBackup);},"foreign backup output");
    Reject([&]{restore(fBackup,slots,mask,accepted,cache);},"foreign backup input");Reject([&]{restore(backup,slots,mask,accepted,fCache);},"foreign restore bank");
    Need(!gpu.LastVerificationKVBackups()&&!gpu.LastVerificationKVRestores(),"rejected calls changed counters");gpu.Abort();other.Abort();unchanged();
    for(bool partial:{false,true}) {
        gpu.Begin();gather(cache,slots,mask,backup);restore(backup,slots,mask,accepted,cache);if(partial)gpu.Flush();gpu.Abort();
        Need(!gpu.LastVerificationKVBackups()&&!gpu.LastVerificationKVRestores(),"Abort span counter reset");
        Reject([&]{gpu.Download(cache);},"aborted cache readable");Reject([&]{gpu.Download(backup);},"aborted backup readable");
        gpu.Update(cache,j.initial.data(),j.initial.size());gpu.Update(backup,oldBackup.data(),oldBackup.size());
    }
    gpu.Begin();gather(cache,slots,mask,backup);restore(backup,slots,mask,accepted,cache);gpu.SubmitAndWait();Equal(gpu.Download(cache),j.initial,"retry recovery");
}
}
int main(){try{
    {
        MetalContext gpu;
        for(auto shape:std::vector<std::pair<size_t,size_t>>{{1,1},{1,2},{1,3},{1,4},{2,1},{2,2},{4,1}})
            for(size_t mode=0;mode<shape.second+3;++mode)for(size_t keep=0;keep<=shape.second;++keep)
                Case(gpu,Job(shape.first,shape.second,1,7,8,8-shape.second,mode),keep,(mode+keep)%4);
        for(auto shape:std::vector<std::pair<size_t,size_t>>{{1,1},{1,4},{2,2},{4,1}})
            for(size_t mode=0;mode<shape.second+3;++mode)for(size_t keep=0;keep<=shape.second;++keep)
                Case(gpu,Job(shape.first,shape.second,4,256,4096,4096-shape.second,mode),keep,(mode+keep)%4);
        Case(gpu,Job(1,1,8,256,1,0,0),1,1);
        for(size_t keep=0;keep<=4;++keep)Case(gpu,Job(1,4,8,37,65,61,0),keep,keep%4);
        Job single(1,1,1,1,1,0,0);uint32_t minusZero=0x80000000u,nan=0x7fc33333u;
        std::memcpy(single.initial.data(),&minusZero,4);std::memcpy(single.tokens.data(),&nan,4);
        for(size_t keep:{0,1})Case(gpu,single,keep,0);
        Carried(gpu);Invalid(gpu);
    }
    auto owners=MetalContext::ProfileOwnerStats();Need(!owners.matrices&&!owners.affineWeights&&!owners.bf16Weights&&!owners.pipelineSets,"Owners remain after KV-span gate");
    std::cout<<"verification_kv_span PASS cases="<<cases<<" checks="<<checks<<" bitwise_words="<<words<<" rejections="<<rejections<<" owners=0; no model speed claim\n";return 0;
}catch(const std::exception& error){std::cerr<<"verification_kv_span FAILED: "<<error.what()<<"\n";return 1;}}
