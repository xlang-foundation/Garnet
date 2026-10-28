// CPU-only actual new policy/geometry guard and retained original headers.
#include "qwen38_rowwise_k_norm_profile.h"
#include "qwen38_decode35_stage_profile.h"
#include "qwen38_full_chunk_stage_profile.h"
#include <iostream>
#include <limits>

struct Config {
    std::vector<int> shape{4,4,5120};std::string layer="model.language_model.layers.35";
    int heads=24,kv=4,dim=256,rotary=64;std::array<int,3> sections{{11,11,10}};
    double theta=1.e7,eps=1.e-6;bool legacy=true,low=false,prefill=false;
    int64_t width=106823680,key=55894016,value=57991168;
    const char* policy="1";std::string precision="modelopt_mixed";
    bool compressed=true,bf16=false,floatInput=true,diagnostic=false,oldWitness=false,decode35Witness=false;
    bool overlay=false,reference=false,fused=true,joined=true;
};
static void validate(const Config& c){
    Garnet::RequireRowwiseKDecodeNormPolicy(c.policy,c.precision,c.compressed,c.bf16,c.floatInput,c.diagnostic,c.oldWitness,c.decode35Witness,c.overlay,c.reference,c.fused,c.joined);
    Garnet::RequireRowwiseKDecodeNormGeometry(c.shape,c.layer,c.heads,c.kv,c.dim,c.rotary,c.sections,c.theta,c.eps,c.legacy,c.low,c.prefill,c.width,c.key,c.value);
}
int main(){
    int valid=0,rejected=0,oldValid=0,oldRejected=0,failures=0;
    auto good=[&](Config c){try{validate(c);++valid;}catch(...){++failures;}};
    auto bad=[&](Config c){try{validate(c);++failures;}catch(const std::invalid_argument&){++rejected;}catch(...){++failures;}};
    Config base;good(base);
    for(const char* x:std::array<const char*,8>{{nullptr,"","0","true","01"," 1","1 ","2"}}){auto c=base;c.policy=x;bad(c);}
    for(const char* x:{"bf16","int4","modelopt_fp8","modelopt_nvfp4"}){auto c=base;c.precision=x;bad(c);}
    for(int i=0;i<10;++i){auto c=base;switch(i){case 0:c.compressed=false;break;case 1:c.bf16=true;break;case 2:c.floatInput=false;break;case 3:c.diagnostic=true;break;case 4:c.oldWitness=true;break;case 5:c.decode35Witness=true;break;case 6:c.overlay=true;break;case 7:c.reference=true;break;case 8:c.fused=false;break;case 9:c.joined=false;break;}bad(c);}
    // Derive cache spans from three GDN states and two packed BF16 FA caches,
    // rather than reuse the guard's offset expression.
    constexpr int64_t linear=10240*4+48*128*128,keyWords=4*4096*256/2;
    for(int group=0;group<16;++group){
        auto c=base;c.layer="model.language_model.layers."+std::to_string(4*group+3);
        c.key=3*linear+group*(3*linear+2*keyWords);c.value=c.key+keyWords;good(c);
        for(int delta:{-1,1}){
            auto d=c;d.key+=delta;bad(d);d=c;d.value+=delta;bad(d);d=c;d.width+=delta;bad(d);
        }
    }
    for(const char* x:{"","model.layers.35","model.language_model.layers.0","model.language_model.layers.1","model.language_model.layers.2","model.language_model.layers.4","model.language_model.layers.035","model.language_model.layers.35 ","model.language_model.layers.64","model.language_model.layers.999999999999999999999"}){auto c=base;c.layer=x;bad(c);}
    for(const auto& x:std::vector<std::vector<int>>{{},{4},{4,4},{4,4,5120,1},{1,4,5120},{2,4,5120},{8,4,5120},{4,1,5120},{4,32,5120},{4,128,5120},{4,4,2560},{0,4,5120}}){auto c=base;c.shape=x;bad(c);}
    for(int x:{0,23}){auto c=base;c.heads=x;bad(c);}
    for(int x:{0,8}){auto c=base;c.kv=x;bad(c);}
    for(int x:{128,512}){auto c=base;c.dim=x;bad(c);}
    for(int x:{0,256}){auto c=base;c.rotary=x;bad(c);}
    for(auto x:std::vector<std::array<int,3>>{{10,11,11},{11,10,11}}){auto c=base;c.sections=x;bad(c);}
    const double inf=std::numeric_limits<double>::infinity(),nan=std::numeric_limits<double>::quiet_NaN();
    for(double x:{1.e6,inf,nan}){auto c=base;c.theta=x;bad(c);}
    for(double x:{1.e-5,inf,nan}){auto c=base;c.eps=x;bad(c);}
    {auto c=base;c.legacy=false;bad(c);c=base;c.low=true;bad(c);c=base;c.prefill=true;bad(c);}
    for(int delta:{-1,1}){auto c=base;c.key+=delta*(3*linear+2*keyWords);bad(c);}
    for(int batch:{1,4}){
        try{Garnet::RequireDecode35StageGeometry({batch,4,5120},base.layer,24,4,256,64,{11,11,10},1.e7,1.e-6,batch==4,batch==1,false,106823680,55894016,57991168);++oldValid;}catch(...){++failures;}
    }
    for(int tokens:{32,128}){
        try{Garnet::RequireFullChunkStageGeometry({1,tokens,5120},"model.language_model.layers.3",24,4,256,64,{11,11,10},1.e7,1.e-6,tokens==32,tokens==128,106823680,2482176,4579328);++oldValid;}catch(...){++failures;}
    }
    try{Garnet::RequireDecode35StageGeometry({8,4,5120},base.layer,24,4,256,64,{11,11,10},1.e7,1.e-6,true,false,false,106823680,55894016,57991168);++failures;}catch(const std::invalid_argument&){++oldRejected;}catch(...){++failures;}
    try{Garnet::RequireFullChunkStageGeometry(base.shape,base.layer,24,4,256,64,{11,11,10},1.e7,1.e-6,false,false,106823680,55894016,57991168);++failures;}catch(const std::invalid_argument&){++oldRejected;}catch(...){++failures;}
    if(valid!=17||rejected!=161||oldValid!=4||oldRejected!=2)++failures;
    std::cout<<"{\"status\":\""<<(failures?"failed":"passed")<<"\",\"new_valid\":"<<valid<<",\"new_rejected\":"<<rejected<<",\"old_valid\":"<<oldValid<<",\"old_rejected\":"<<oldRejected<<",\"failures\":"<<failures<<",\"cases\":"<<(valid+rejected+oldValid+oldRejected)<<"}\n";
    return failures?1:0;
}
