// CPU-only: exercise actual new and retained original geometry/policy headers.
#include "qwen38_decode35_stage_profile.h"
#include "qwen38_full_chunk_stage_profile.h"
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <utility>

struct Config {
    std::vector<int> shape{1,4,5120};std::string layer="model.language_model.layers.35";
    int heads=24,kv=4,dim=256,rotary=64;std::array<int,3> sections{{11,11,10}};
    double theta=1.e7,eps=1.e-6;bool legacy=false,low=true,prefill=false;
    int64_t width=106823680,key=55894016,value=57991168;
    const char* policy="1";std::string precision="modelopt_mixed";
    bool compressed=true,bf16=false,floatInput=true,diagnostic=false,oldWitness=false;
    bool overlay=false,reference=false,fused=true,joined=true;
};
static void validate(const Config& c){
    Garnet::RequireDecode35StagePolicy(c.policy,c.precision,c.compressed,c.bf16,c.floatInput,c.diagnostic,c.oldWitness,c.overlay,c.reference,c.fused,c.joined);
    Garnet::RequireDecode35StageGeometry(c.shape,c.layer,c.heads,c.kv,c.dim,c.rotary,c.sections,c.theta,c.eps,c.legacy,c.low,c.prefill,c.width,c.key,c.value);
}
int main(){
    int valid=0,rejected=0,oldValid=0,oldRejected=0,failures=0;
    auto good=[&](Config c){try{validate(c);++valid;}catch(...){++failures;}};
    auto bad=[&](Config c){try{validate(c);++failures;}catch(const std::invalid_argument&){++rejected;}catch(...){++failures;}};
    Config b1,b4;b4.shape[0]=4;b4.legacy=true;b4.low=false;good(b1);good(b4);
    for(const char* x:std::array<const char*,10>{{nullptr,"","0","00","01","true","1 "," 1","2","-1"}}){auto c=b1;c.policy=x;bad(c);}
    for(const char* x:{"","bf16","modelopt_fp8"}){auto c=b1;c.precision=x;bad(c);}
    for(int i=0;i<9;++i){auto c=b1;switch(i){case 0:c.compressed=false;break;case 1:c.bf16=true;break;case 2:c.floatInput=false;break;case 3:c.diagnostic=true;break;case 4:c.oldWitness=true;break;case 5:c.overlay=true;break;case 6:c.reference=true;break;case 7:c.fused=false;break;case 8:c.joined=false;break;}bad(c);}
    for(const std::vector<int>& x:std::vector<std::vector<int>>{{},{1},{1,4},{1,4,5120,1},{0,4,5120},{2,4,5120},{8,4,5120},{1,1,5120},{1,16,5120},{1,32,5120},{1,4,2560},{4,4,5120}}){auto c=b1;c.shape=x;bad(c);}
    {auto c=b4;c.shape[0]=1;bad(c);}
    for(const char* x:{"","model.language_model.layers.3","model.language_model.layers.34","model.language_model.layers.36","model.layers.35"}){auto c=b1;c.layer=x;bad(c);}
    for(int x:{-1,0,16,32}){auto c=b1;c.heads=x;bad(c);}
    for(int x:{-1,0,2,8}){auto c=b1;c.kv=x;bad(c);}
    for(int x:{-1,0,128,512}){auto c=b1;c.dim=x;bad(c);}
    for(int x:{-1,0,32,128,256}){auto c=b1;c.rotary=x;bad(c);}
    for(auto x:std::vector<std::array<int,3>>{{10,11,11},{11,10,11},{11,11,9},{11,11,11},{-1,11,10}}){auto c=b1;c.sections=x;bad(c);}
    const double inf=std::numeric_limits<double>::infinity(),nan=std::numeric_limits<double>::quiet_NaN();
    for(double x:{-1.,0.,10000.,1.e6,std::nextafter(1.e7,0.),std::nextafter(1.e7,inf),inf,nan}){auto c=b1;c.theta=x;bad(c);}
    for(double x:{-1.,0.,1.e-5,1.e-7,std::nextafter(1.e-6,0.),std::nextafter(1.e-6,inf),inf,nan}){auto c=b1;c.eps=x;bad(c);}
    for(int i=0;i<8;++i){auto c=i<4?b1:b4;switch(i){case 0:c.low=false;break;case 1:c.legacy=true;break;case 2:c.low=false;c.legacy=true;break;case 3:c.prefill=true;break;case 4:c.prefill=true;break;case 5:c.legacy=false;c.low=true;break;case 6:c.low=true;c.legacy=true;break;case 7:c.low=false;c.legacy=false;break;}bad(c);}
    for(int64_t x:{-1ll,0ll,106823679ll,106823681ll,173932544ll}){auto c=b1;c.width=x;bad(c);}
    for(int64_t x:{-1ll,0ll,2482176ll,55894015ll,55894017ll,57991168ll}){auto c=b1;c.key=x;bad(c);}
    for(int64_t x:{-1ll,0ll,4579328ll,57991167ll,57991169ll,55894016ll}){auto c=b1;c.value=x;bad(c);}
    for(int tokens:{32,128}){try{Garnet::RequireFullChunkStageGeometry({1,tokens,5120},"model.language_model.layers.3",24,4,256,64,{11,11,10},1.e7,1.e-6,tokens==32,tokens==128,106823680,2482176,4579328);++oldValid;}catch(...){++failures;}}
    for(const auto& c:{b1,b4}){try{Garnet::RequireFullChunkStageGeometry(c.shape,c.layer,c.heads,c.kv,c.dim,c.rotary,c.sections,c.theta,c.eps,c.low,c.prefill,c.width,c.key,c.value);++failures;}catch(const std::invalid_argument&){++oldRejected;}catch(...){++failures;}}
    if(std::accumulate(Garnet::Decode35StageWidths.begin(),Garnet::Decode35StageWidths.end(),0)!=Garnet::Decode35StageTotalWidth)++failures;
    if(valid!=2||rejected!=103||oldValid!=2||oldRejected!=2)++failures;
    std::cout<<"{\"status\":\""<<(failures?"failed":"passed")<<"\",\"new_valid\":"<<valid<<",\"new_rejected\":"<<rejected<<",\"old_valid\":"<<oldValid<<",\"old_rejected\":"<<oldRejected<<",\"failures\":"<<failures<<",\"cases\":"<<(valid+rejected+oldValid+oldRejected)<<"}\n";
    return failures?1:0;
}
