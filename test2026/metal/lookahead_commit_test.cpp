// SPDX-License-Identifier: Apache-2.0
#include "metal_context.h"
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace Garnet;
namespace {
size_t cases=0,words=0,rejections=0;
void Need(bool b,const char* why){if(!b)throw std::runtime_error(why);}
template<class F>void Reject(F f){bool b=false;try{f();}catch(const std::invalid_argument&){b=true;}catch(const std::logic_error&){b=true;}Need(b,"Invalid commit accepted");++rejections;}
void Equal(const std::vector<float>& a,const std::vector<float>& b){Need(a.size()==b.size(),"Commit size differs");for(size_t i=0;i<a.size();++i)if(std::memcmp(&a[i],&b[i],4))throw std::runtime_error("Commit word differs at "+std::to_string(i));words+=a.size();}
std::vector<float> Bits(size_t count,uint32_t seed){std::vector<float> v(count);for(size_t i=0;i<count;++i){uint32_t x=i%11==0?0x80000000u:i%13==0?0x7fc00000u|uint32_t(i&0x3fffff):seed+uint32_t(i)*2654435761u;std::memcpy(&v[i],&x,4);}return v;}
void Case(MetalContext& gpu,const LookaheadLayout& p,size_t slot,size_t budget,size_t branch,size_t count,bool sparse,bool flush){
    auto localBits=Bits(p.Batch()*16*1024,123),bankBits=Bits(p.Batch()*8*p.Capacity()*128,987),expected=bankBits;
    auto local=gpu.Upload(localBits,p.Batch()*16,1024),bank=gpu.Upload(bankBits,p.Batch(),8*p.Capacity()*128);
    std::vector<LookaheadLayout::View> views;std::vector<LookaheadLayout::CommitSelection> paths;
    for(size_t row=p.Batch();row-->0;)if(!sparse||row==p.Batch()-1){
        auto view=p.Request(row,81+row,slot,budget,p.Guesses(),p.Levels()-1);auto path=p.SelectCommit(view,branch,count);views.push_back(view);paths.push_back(path);
        for(size_t t=0;t<count;++t)for(size_t h=0;h<8;++h)for(size_t d=0;d<128;++d)
            expected[((row*8+h)*p.Capacity()+slot+t)*128+d]=localBits[((row*16+path.local[t])*8+h)*128+d];
    }
    gpu.Begin();Need(gpu.LastLookaheadKVCommitDispatches()==0,"Begin retained counter");
    gpu.EncodeLookaheadKVCommit(local,bank,p,views,paths);Need(gpu.LastLookaheadKVCommitDispatches()==1,"Commit count differs");
    Reject([&]{gpu.Download(bank);});Reject([&]{gpu.Begin();});Need(gpu.LastLookaheadKVCommitDispatches()==1,"Invalid Begin reset counter");
    if(flush)gpu.Flush();gpu.SubmitAndWait();Equal(gpu.Download(bank),expected);Equal(gpu.Download(local),localBits);++cases;
}
void Invalid(MetalContext& gpu,MetalContext& other){
    LookaheadLayout p(4,17,4,4,1),foreign(4,17,3,4,1);
    auto bits=Bits(4*8*17*128,17),srcBits=Bits(64*1024,11);
    auto bank=gpu.Upload(bits,4,8*17*128),local=gpu.Upload(srcBits,64,1024);
    std::vector<LookaheadLayout::View> views{p.Request(3,71,7,8,1,3)};
    std::vector<LookaheadLayout::CommitSelection> paths{p.SelectCommit(views[0],0,3)};
    Reject([&]{gpu.EncodeLookaheadKVCommit(local,bank,p,views,paths);});
    for(size_t mutation=0;mutation<10;++mutation){
        auto v=views;auto q=paths;auto input=local,output=bank;
        if(mutation==0){v.push_back(v[0]);q.push_back(q[0]);}
        if(mutation==1)v[0]=foreign.Request(3,71,7,8,1,3);
        if(mutation==2)q[0].local[1]=1;
        if(mutation==3)q[0].local[1]++;
        if(mutation==4)q[0].local[3]=1;
        if(mutation==5)q[0].count=0;
        if(mutation==6)q[0].count=5;
        if(mutation==7)input=gpu.Allocate(64,1024);
        if(mutation==8)output=local;
        if(mutation==9)q.clear();
        gpu.Begin();Reject([&]{gpu.EncodeLookaheadKVCommit(input,output,p,v,q);});Need(gpu.LastLookaheadKVCommitDispatches()==0,"Rejected commit dispatched");gpu.Abort();Equal(gpu.Download(bank),bits);
    }
    gpu.Begin();gpu.EncodeLookaheadKVCommit(local,bank,p,{},{});Need(gpu.LastLookaheadKVCommitDispatches()==0,"Empty commit dispatched");gpu.SubmitAndWait();Equal(gpu.Download(bank),bits);
    // Same-command pending operands and canonical writers are legal; foreign
    // pending buffers are rejected. A real Abort poisons only its output.
    auto pending=other.Allocate(64,1024),zero=other.Upload(std::vector<float>(64*1024,0),64,1024);
    other.Begin();other.EncodeAdd(local,zero,pending);gpu.Begin();Reject([&]{gpu.EncodeLookaheadKVCommit(pending,bank,p,views,paths);});gpu.Abort();other.Abort();Equal(gpu.Download(bank),bits);
    gpu.Begin();gpu.EncodeLookaheadKVCommit(local,bank,p,views,paths);
    other.Begin();Reject([&]{other.EncodeLookaheadKVCommit(local,bank,p,views,paths);});other.Abort();
    gpu.Abort();Need(gpu.LastLookaheadKVCommitDispatches()==0,"Abort retained commit counter");Reject([&]{gpu.Download(bank);});
    gpu.Update(bank,bits);gpu.Begin();gpu.EncodeLookaheadKVCommit(local,bank,p,views,paths);gpu.SubmitAndWait();++cases;
}
}
int main(){try{
    {MetalContext gpu,other;Invalid(gpu,other);
    size_t profiles=0;
    for(size_t n=2;n<=4;++n)for(size_t w=1;w<=4;++w)for(size_t g=1;g<=4;++g)if((n-1)*(w+g)<=16){
        ++profiles;for(size_t b:{size_t(1),size_t(4)}){
            LookaheadLayout p(b,17,w,n,g);
            for(size_t branch=0;branch<g;++branch)for(size_t count=1;count<=n;++count){
                Case(gpu,p,3,8,branch,count,false,count%2);
                Case(gpu,p,17-count,count,branch,count,true,count%2==0);
            }
        }
    }
    Need(profiles==42,"Missing layout");
    for(size_t b:{size_t(1),size_t(4)}){LookaheadLayout p(b,4096,4,4,1);for(size_t n=1;n<=4;++n)Case(gpu,p,4096-n,n,0,n,true,true);}
    }
    auto owners=MetalContext::ProfileOwnerStats();Need(!owners.matrices&&!owners.affineWeights&&!owners.bf16Weights&&!owners.pipelineSets,"Commit owners retained");
    std::cout<<"lookahead commit PASS cases="<<cases<<" fp32_words="<<words<<" rejections="<<rejections<<" layouts=42 owners=0 no_model_or_speed_claim=true\n";return 0;
}catch(const std::exception& e){std::cerr<<"lookahead commit FAILED: "<<e.what()<<'\n';return 1;}}
