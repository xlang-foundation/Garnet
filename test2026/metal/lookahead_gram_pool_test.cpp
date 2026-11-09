// SPDX-License-Identifier: Apache-2.0
#include "lookahead_gram_pool.h"
#include <iostream>
#include <stdexcept>
#include <string>
using namespace Garnet;
namespace {
size_t checks=0,rejections=0,observations=0;
void Need(bool ok,const char* why){if(!ok)throw std::runtime_error(why);++checks;}
template<class F>void Reject(F f){bool rejected=false;try{f();}catch(const std::invalid_argument&){rejected=true;}Need(rejected,"Bad pool input accepted");++rejections;}
void Equal(const LookaheadGramPool& a,const LookaheadGramPool& b,size_t vocab) {
    for(size_t key=0;key<vocab;++key){auto x=a.Lookup(int32_t(key)),y=b.Lookup(int32_t(key));Need(x.count==y.count&&x.grams==y.grams,"Rejected observation mutated candidate table");}
}
void Print(const LookaheadGramPool& p,size_t n,size_t w,size_t g,size_t round) {
    for(int32_t key=0;key<32;++key) {
        const auto c=p.Lookup(key);std::cout<<"STATE "<<n<<' '<<w<<' '<<g<<' '<<round<<' '<<key<<' '<<c.count;
        for(size_t i=0;i<c.count;++i)for(size_t t=0;t<n-1;++t)std::cout<<' '<<c.grams[i][t];std::cout<<'\n';
    }
}
}
int main(){try{
    for(size_t n=2;n<=4;++n)for(size_t w=1;w<=4;++w)for(size_t g=1;g<=4;++g) {
        if((n-1)*(w+g)>16)continue;
        LookaheadLayout layout(4,37,w,n,g);LookaheadGramPool pool(n,g,128);uint32_t seed=uint32_t(79+n*31+w*13+g);
        for(size_t round=0;round<12;++round) {
            auto view=layout.Request(round%4,123+round,3,17,0,n-1);
            std::array<int32_t,16> inputs,predictions;inputs.fill(-1);predictions.fill(-1);
            for(size_t i=0;i<16;++i)if(view.Active()&(uint16_t(1)<<i)) {
                seed=1664525u*seed+1013904223u;inputs[i]=int32_t((seed>>8)%32);
                seed=1664525u*seed+1013904223u;predictions[i]=int32_t((seed>>8)%32);
            }
            // Emit actual inputs so a separately extracted upstream function
            // can check the native table, not a mirrored C++ implementation.
            std::cout<<"OBSERVE "<<n<<' '<<w<<' '<<g<<' '<<round;
            for(size_t i=0;i<(n-1)*w;++i)std::cout<<' '<<inputs[i]<<' '<<predictions[i];std::cout<<'\n';
            Need(pool.Observe(layout,view,inputs,predictions)==w,"Full window insertion count differs");++observations;Print(pool,n,w,g,round);
            const auto before=pool;
            for(size_t node=0;node<(n-1)*w;++node) {
                auto bad=predictions;bad[node]=128;Reject([&]{pool.Observe(layout,view,inputs,bad);});Equal(pool,before,128);
                auto badInput=inputs;badInput[node]=-1;Reject([&]{pool.Observe(layout,view,badInput,predictions);});Equal(pool,before,128);
            }
            auto inactive=inputs;inactive[15]=0;
            if(!(view.Active()&(uint16_t(1)<<15))){Reject([&]{pool.Observe(layout,view,inactive,predictions);});Equal(pool,before,128);}
        }
        auto tail=layout.Request(3,999,36,1,0,n-1);std::array<int32_t,16> inputs,pred;inputs.fill(-1);pred.fill(-1);inputs[0]=7;pred[0]=11;
        Need(pool.Observe(layout,tail,inputs,pred)==(n==2?1:0),"Tail created incomplete ngram");
        pool.Clear();for(int32_t key=0;key<128;++key)Need(pool.Lookup(key).count==0,"Clear retained candidates");
    }
    LookaheadGramPool bounded(2,2,128);
    for(int32_t key=0;key<64;++key)bounded.Insert(key,{1,0,0});
    bounded.Insert(0,{2,0,0});bounded.Insert(64,{3,0,0});
    Need(bounded.Lookup(1).count==0&&bounded.Lookup(0).count==2&&bounded.Lookup(64).count==1,"Bounded key insertion recency differs");
    bounded.Insert(0,{1,0,0});auto c=bounded.Lookup(0);Need(c.count==2&&c.grams[0][0]==2&&c.grams[1][0]==1,"Duplicate tuple LRU differs");
    auto before=bounded;Reject([&]{bounded.Insert(4,{1,2,0});});Equal(bounded,before,128);
    Reject([&]{bounded.Insert(-1,{1,0,0});});Equal(bounded,before,128);
    std::cout<<"PASS checks="<<checks<<" rejections="<<rejections<<" observations="<<observations<<" host_bytes="<<sizeof(LookaheadGramPool)<<" GPU_calls=0 runtime_integration=false\n";
    return 0;
}catch(const std::exception& e){std::cerr<<"FAILED "<<e.what()<<'\n';return 1;}}
