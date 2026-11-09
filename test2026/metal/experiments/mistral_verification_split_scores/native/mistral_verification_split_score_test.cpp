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
void Need(bool value,const std::string& why){if(!value)throw std::runtime_error(why);}
std::vector<float> Values(size_t count,uint32_t seed) {
    std::vector<float> values(count);
    for(auto& value:values){seed=1664525u*seed+1013904223u;value=float(int((seed>>8)%65521)-32760)*.00001f;}
    return values;
}
size_t checks=0,words=0,rejections=0;
void Equal(const std::vector<float>& actual,const std::vector<float>& expected,const std::string& label) {
    Need(actual.size()==expected.size(),label+" shape differs");
    for(size_t i=0;i<actual.size();++i)if(std::memcmp(&actual[i],&expected[i],4))
        throw std::runtime_error(label+" word"+std::to_string(i)+" differs: "+std::to_string(actual[i])+" vs "+std::to_string(expected[i]));
    words+=actual.size();
}
struct Job {
    size_t batch,chunk,heads,kv,dim,capacity,qw,kw,cw;
    std::vector<float> q,k,v,kc,vc,slots,active,gate;
    Job(size_t b,size_t t,size_t h,size_t kh,size_t d,size_t c,size_t base,size_t mask)
        :batch(b),chunk(t),heads(h),kv(kh),dim(d),capacity(c),qw(h*d),kw(kh*d),cw(kh*c*d),
         q(Values(b*t*qw,13)),k(Values(b*t*kw,37)),v(Values(b*t*kw,97)),
         kc(Values(b*cw,31)),vc(Values(b*cw,71)),slots(b*t),active(b*t),gate(Values(b*t*qw,131)) {
        for(size_t row=0;row<b;++row) {
            const size_t length=mask==0?t:(mask==1?(row+t-1)%(t+1):(mask==3?1:(mask==4?std::min(size_t(2),t):0)));
            const size_t first=base-std::min(base,row*2);
            for(size_t token=0;token<t;++token) {
                slots[row*t+token]=token<length?float(first+token):std::numeric_limits<float>::quiet_NaN();
                active[row*t+token]=float(token<length);
                if(token>=length){std::fill_n(q.data()+(row*t+token)*qw,qw,std::numeric_limits<float>::quiet_NaN());std::fill_n(k.data()+(row*t+token)*kw,kw,std::numeric_limits<float>::quiet_NaN());std::fill_n(v.data()+(row*t+token)*kw,kw,std::numeric_limits<float>::quiet_NaN());}
            }
        }
    }
};
struct Result{std::vector<float> output,probabilities,key,value;};
Result Sequential(MetalContext& gpu,const Job& j,bool gated) {
    Result result{std::vector<float>(j.batch*j.chunk*j.qw),std::vector<float>(j.batch*j.chunk*j.heads*j.capacity),{}, {}};
    auto key=gpu.Upload(j.kc,j.batch,j.cw),value=gpu.Upload(j.vc,j.batch,j.cw);
    for(size_t t=0;t<j.chunk;++t) {
        std::vector<float> q(j.batch*j.qw),k(j.batch*j.kw),v(k.size()),g(q.size()),slots(j.batch),active(j.batch);
        for(size_t row=0;row<j.batch;++row) {
            std::copy_n(j.q.data()+(row*j.chunk+t)*j.qw,j.qw,q.data()+row*j.qw);
            std::copy_n(j.gate.data()+(row*j.chunk+t)*j.qw,j.qw,g.data()+row*j.qw);
            std::copy_n(j.k.data()+(row*j.chunk+t)*j.kw,j.kw,k.data()+row*j.kw);
            std::copy_n(j.v.data()+(row*j.chunk+t)*j.kw,j.kw,v.data()+row*j.kw);
            slots[row]=j.active[row*j.chunk+t]?j.slots[row*j.chunk+t]:0;active[row]=j.active[row*j.chunk+t];
        }
        auto qg=gpu.Upload(q,j.batch,j.qw),kg=gpu.Upload(k,j.batch,j.kw),vg=gpu.Upload(v,j.batch,j.kw),gg=gpu.Upload(g,j.batch,j.qw),sg=gpu.Upload(slots,j.batch,1),ag=gpu.Upload(active,j.batch,1);
        auto nextKey=gpu.Allocate(j.batch,j.cw),nextValue=gpu.Allocate(j.batch,j.cw),prob=gpu.Allocate(j.batch,j.heads*j.capacity),out=gpu.Allocate(j.batch,j.qw);
        gpu.Begin();gpu.EncodeKVUpdate(kg,key,sg,ag,nextKey,j.kv,j.dim,j.capacity);gpu.EncodeKVUpdate(vg,value,sg,ag,nextValue,j.kv,j.dim,j.capacity);
        gpu.EncodeAttentionScores(qg,nextKey,sg,ag,prob,j.heads,j.kv,j.dim,j.capacity);
        gpu.EncodeAttentionContext(prob,nextValue,gg,sg,out,j.heads,j.kv,j.dim,j.capacity,gated);gpu.SubmitAndWait();
        auto outputs=gpu.Download(out),probabilities=gpu.Download(prob);
        for(size_t row=0;row<j.batch;++row){std::copy_n(outputs.data()+row*j.qw,j.qw,result.output.data()+(row*j.chunk+t)*j.qw);std::copy_n(probabilities.data()+row*j.heads*j.capacity,j.heads*j.capacity,result.probabilities.data()+(row*j.chunk+t)*j.heads*j.capacity);}
        key=nextKey;value=nextValue;
    }
    result.key=gpu.Download(key);result.value=gpu.Download(value);return result;
}
std::vector<MetalMatrix> Upload(MetalContext& gpu,const Job& j) {
    return {gpu.Upload(j.q,j.batch*j.chunk,j.qw),gpu.Upload(j.k,j.batch*j.chunk,j.kw),gpu.Upload(j.v,j.batch*j.chunk,j.kw),
        gpu.Upload(j.kc,j.batch,j.cw),gpu.Upload(j.vc,j.batch,j.cw),gpu.Upload(j.slots,j.batch,j.chunk),gpu.Upload(j.active,j.batch,j.chunk),gpu.Upload(j.gate,j.batch*j.chunk,j.qw)};
}
void Case(MetalContext& gpu,const Job& j,bool gated,size_t variant,bool allowLayout=true,bool allowVector=true,bool allowSplit=true) {
    auto expected=Sequential(gpu,j,gated);auto inputs=Upload(gpu,j);
    auto probabilities=gpu.Allocate(j.batch*j.chunk,j.heads*j.capacity),output=gpu.Allocate(j.batch*j.chunk,j.qw);
    auto nextKey=variant==1?inputs[3]:gpu.Allocate(j.batch,j.cw),nextValue=variant==1?inputs[4]:gpu.Allocate(j.batch,j.cw);
    auto pendingQuery=gpu.Allocate(j.batch*j.chunk,j.qw),zero=gpu.Upload(std::vector<float>(j.batch*j.chunk*j.qw,0),j.batch*j.chunk,j.qw);
    gpu.Begin();Need(gpu.LastVerificationAttentionDispatches()==0,"Begin retained verification counter");Need(gpu.LastVerificationLayoutDispatches()==0,"Begin retained layout counter");Need(gpu.LastVerificationSplitScoreDispatches()==0,"Begin retained split score counter");Need(gpu.LastVerificationValueVectorDispatches()==0,"Begin retained value vector counter");
    if(variant==1){gpu.EncodeKVChunk(inputs[1],inputs[3],inputs[5],inputs[6],nextKey,j.kv,j.dim,j.capacity,j.chunk,true);gpu.EncodeKVChunk(inputs[2],inputs[4],inputs[5],inputs[6],nextValue,j.kv,j.dim,j.capacity,j.chunk,true);}
    if(variant==2){gpu.EncodeAdd(inputs[0],zero,pendingQuery);inputs[0]=pendingQuery;gpu.Flush();}
    gpu.EncodeVerificationAttention(inputs,probabilities,output,j.heads,j.kv,j.dim,j.capacity,j.chunk,gated);
    Need(gpu.LastVerificationAttentionDispatches()==1,"Verification route missing");
    #ifdef GARNET_METAL_MISTRAL_VERIFY_LAYOUT
    constexpr bool enabled=true;
#else
    constexpr bool enabled=false;
#endif
    const bool profiled=enabled&&allowLayout&&j.batch==1&&j.chunk==4&&j.heads==32&&j.kv==8&&j.dim==128&&j.capacity==4096&&!gated;
    Need(gpu.LastVerificationLayoutDispatches()==size_t(profiled),"Compiled layout route differs");
#ifdef GARNET_METAL_MISTRAL_VERIFY_VECTOR4
    const bool vector=profiled&&allowVector;
#else
    const bool vector=false;
#endif
    Need(gpu.LastVerificationValueVectorDispatches()==size_t(vector),"Value vector route differs");
    size_t visible=0;for(size_t t=0;t<j.chunk;++t)if(j.active[t])visible=std::max(visible,size_t(j.slots[t])+1);
#ifdef GARNET_METAL_MISTRAL_VERIFY_SPLIT_SCORES
    const bool split=profiled&&allowSplit&&visible>=512;
#else
    const bool split=false;
#endif
    Need(gpu.LastVerificationSplitScoreDispatches()==size_t(split),"Split verification route differs");
#ifdef GARNET_METAL_MISTRAL_VERIFY_SPLIT_SCORES
    {bool rejected=false;try{gpu.VerificationSetSplitScoresEnabled(false);}catch(const std::logic_error&){rejected=true;}
     Need(rejected&&gpu.LastVerificationSplitScoreDispatches()==size_t(split),"Open/partial batch accepted split score route change");++rejections;}
#endif
#ifdef GARNET_METAL_MISTRAL_VERIFY_VECTOR4
    {bool rejected=false;try{gpu.VerificationSetValueVector4(false);}catch(const std::logic_error&){rejected=true;}
     Need(rejected&&gpu.LastVerificationValueVectorDispatches()==size_t(vector),"Open/partial batch accepted value route change");++rejections;}
#endif
#ifdef GARNET_METAL_MISTRAL_VERIFY_LAYOUT
    {bool rejected=false;try{gpu.VerificationSetMistralLayout(false);}catch(const std::logic_error&){rejected=true;}
     Need(rejected&&gpu.LastVerificationLayoutDispatches()==size_t(profiled),"Open/partial batch accepted layout change");++rejections;}
#endif
    if(variant!=1){gpu.EncodeKVChunk(inputs[1],inputs[3],inputs[5],inputs[6],nextKey,j.kv,j.dim,j.capacity,j.chunk,true);gpu.EncodeKVChunk(inputs[2],inputs[4],inputs[5],inputs[6],nextValue,j.kv,j.dim,j.capacity,j.chunk,true);}
    for(const auto& matrix:{probabilities,output}){bool rejected=false;try{gpu.Download(matrix);}catch(const std::logic_error&){rejected=true;}Need(rejected,"Pending verification output readable");++rejections;}
    gpu.SubmitAndWait();
    const auto label="B"+std::to_string(j.batch)+" T"+std::to_string(j.chunk)+" H"+std::to_string(j.heads)+" D"+std::to_string(j.dim)+" C"+std::to_string(j.capacity)+" variant"+std::to_string(variant);
    Equal(gpu.Download(probabilities),expected.probabilities,label+" probabilities");Equal(gpu.Download(output),expected.output,label+" context");
    Equal(gpu.Download(nextKey),expected.key,label+" KV key");Equal(gpu.Download(nextValue),expected.value,label+" KV value");++checks;
}
void Invalid(MetalContext& gpu,MetalContext& other) {
    Job job(2,4,4,2,8,64,9,0);auto inputs=Upload(gpu,job);
    auto probabilities=gpu.Upload(std::vector<float>(8*4*64,.25f),8,4*64),output=gpu.Upload(std::vector<float>(8*32,.5f),8,32);
    const auto beforeP=gpu.Download(probabilities),beforeO=gpu.Download(output),beforeK=gpu.Download(inputs[3]);
    for(size_t mutation=0;mutation<11;++mutation) {
        auto args=inputs;auto slots=job.slots,active=job.active;
        if(mutation==0)slots[1]=slots[0];
        if(mutation==1)slots[1]=slots[0]-1;
        if(mutation==2)slots[0]=-1;
        if(mutation==3)slots[0]=64;
        if(mutation==4)slots[0]=.5f;
        if(mutation==5)slots[0]=std::numeric_limits<float>::quiet_NaN();
        if(mutation==6)active[1]=0;
        if(mutation==7)active[0]=.5f;
        if(mutation==8)active[0]=std::numeric_limits<float>::quiet_NaN();
        args[5]=gpu.Upload(slots,2,4);args[6]=gpu.Upload(active,2,4);
        auto copied=gpu.Allocate(2,4),zero=gpu.Upload(std::vector<float>(8,0),2,4);
        gpu.Begin();
        if(mutation>=9){const size_t index=mutation==9?5:6;gpu.EncodeAdd(args[index],zero,copied);args[index]=copied;}
        bool rejected=false;try{gpu.EncodeVerificationAttention(args,probabilities,output,4,2,8,64,4,true);}catch(const std::invalid_argument&){rejected=true;}catch(const std::logic_error&){rejected=true;}
        Need(rejected&&gpu.LastVerificationAttentionDispatches()==0,"Invalid or pending controls accepted");gpu.Abort();
        Equal(gpu.Download(probabilities),beforeP,"Rejected controls changed probabilities");Equal(gpu.Download(output),beforeO,"Rejected controls changed output");Equal(gpu.Download(inputs[3]),beforeK,"Rejected controls changed cache");++rejections;
    }
    gpu.Begin();bool rejected=false;try{gpu.EncodeVerificationAttention(inputs,probabilities,inputs[0],4,2,8,64,4,true);}catch(const std::invalid_argument&){rejected=true;}gpu.Abort();Need(rejected,"Input/output alias accepted");++rejections;
    auto aliasInputs=Upload(gpu,Job(2,4,4,2,64,64,9,0));
    gpu.Begin();rejected=false;try{gpu.EncodeVerificationAttention(aliasInputs,probabilities,probabilities,4,2,64,64,4,true);}catch(const std::invalid_argument&){rejected=true;}gpu.Abort();Need(rejected,"Probability/context alias accepted");++rejections;
    auto missing=inputs;missing[0]=gpu.Allocate(8,32);
    gpu.Begin();rejected=false;try{gpu.EncodeVerificationAttention(missing,probabilities,output,4,2,8,64,4,true);}catch(const std::logic_error&){rejected=true;}gpu.Abort();Need(rejected,"Uninitialized query accepted");++rejections;
    gpu.Begin();gpu.EncodeVerificationAttention(inputs,probabilities,output,4,2,8,64,4,true);
    rejected=false;try{gpu.Begin();}catch(const std::logic_error&){rejected=true;}Need(rejected&&gpu.LastVerificationAttentionDispatches()==1,"Invalid Begin changed pending counter");++rejections;
    other.Begin();rejected=false;try{other.EncodeVerificationAttention(inputs,probabilities,output,4,2,8,64,4,true);}catch(const std::logic_error&){rejected=true;}other.Abort();Need(rejected,"Foreign verification output writer accepted");++rejections;
    gpu.Abort();Need(gpu.LastVerificationAttentionDispatches()==0&&gpu.LastVerificationValueVectorDispatches()==0&&gpu.LastVerificationSplitScoreDispatches()==0,"Abort retained verification counter");
    for(const auto& matrix:{probabilities,output}){rejected=false;try{gpu.Download(matrix);}catch(const std::logic_error&){rejected=true;}Need(rejected,"Aborted output readable");++rejections;}
    gpu.Begin();gpu.EncodeVerificationAttention(inputs,probabilities,output,4,2,8,64,4,true);gpu.SubmitAndWait();
    auto expected=Sequential(gpu,job,true);Equal(gpu.Download(output),expected.output,"Abort recovery context");Equal(gpu.Download(probabilities),expected.probabilities,"Abort recovery probabilities");
}
}
int main(){try{
    {
        MetalContext gpu,other;
        for(size_t base:{size_t(0),size_t(384),size_t(4092)})for(size_t mask=0;mask<5;++mask)
            Case(gpu,Job(1,4,32,8,128,4096,base,mask),false,checks%3);
        Case(gpu,Job(1,4,32,8,128,4096,384,0),true,0);
        Case(gpu,Job(1,3,32,8,128,4096,384,0),false,0);
        Case(gpu,Job(1,4,32,8,128,1024,384,0),false,0);
        Case(gpu,Job(2,2,32,8,128,4096,384,0),false,0);
#ifdef GARNET_METAL_MISTRAL_VERIFY_LAYOUT
        gpu.VerificationSetMistralLayout(false);
        Case(gpu,Job(1,4,32,8,128,4096,384,0),false,0,false);
        gpu.VerificationSetMistralLayout(true);
        Case(gpu,Job(1,4,32,8,128,4096,384,0),false,0);
#else
        {bool rejected=false;try{gpu.VerificationSetMistralLayout(false);}catch(const std::logic_error&){rejected=true;}
         Need(rejected,"Disabled build accepted layout control");++rejections;}
#endif
#ifdef GARNET_METAL_MISTRAL_VERIFY_VECTOR4
        gpu.VerificationSetValueVector4(false);
        Case(gpu,Job(1,4,32,8,128,4096,384,0),false,0,true,false);
        gpu.VerificationSetValueVector4(true);
        Case(gpu,Job(1,4,32,8,128,4096,384,0),false,0);
#else
        {bool rejected=false;try{gpu.VerificationSetValueVector4(false);}catch(const std::logic_error&){rejected=true;}
         Need(rejected,"Disabled build accepted value vector control");++rejections;}
#endif
        for(size_t base:{size_t(507),size_t(508),size_t(511),size_t(2048)})for(size_t mask=0;mask<5;++mask)
            Case(gpu,Job(1,4,32,8,128,4096,base,mask),false,checks%3);
#ifdef GARNET_METAL_MISTRAL_VERIFY_SPLIT_SCORES
        gpu.VerificationSetSplitScoresEnabled(false);
        Case(gpu,Job(1,4,32,8,128,4096,2048,0),false,0,true,true,false);
        gpu.VerificationSetSplitScoresEnabled(true);
        Case(gpu,Job(1,4,32,8,128,4096,2048,0),false,2);
#else
        {bool rejected=false;try{gpu.VerificationSetSplitScoresEnabled(true);}catch(const std::logic_error&){rejected=true;}
         Need(rejected,"Disabled build accepted split score selection");++rejections;}
#endif
        Invalid(gpu,other);
    }
    const auto owners=MetalContext::ProfileOwnerStats();Need(!owners.matrices&&!owners.affineWeights&&!owners.bf16Weights&&!owners.pipelineSets,"Verification owners remain after close");
    std::cout<<"Mistral verification split scores PASS cases="<<checks<<" fp32_words="<<words<<" rejections="<<rejections
             <<" probabilities_context_fullkv_bitwise=true cache_before_after=true pending_queries=true partial_flush=true invalid_controls_pending_abort_alias_foreign=true owners=0\n";
    return 0;
}catch(const std::exception& error){std::cerr<<"verification attention decode reference FAILED: "<<error.what()<<"\n";return 1;}}
