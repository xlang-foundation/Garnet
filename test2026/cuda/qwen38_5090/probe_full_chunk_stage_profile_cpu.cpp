// Actual header-only profile checks. No Garnet/CUDA/TensorRT/model execution.
#include "qwen38_full_chunk_stage_profile.h"
#include <cstdio>
#include <limits>
using namespace Garnet;
struct Profile {
    std::vector<int> shape;std::string layer="model.language_model.layers.3",precision="modelopt_mixed";
    int heads=24,kv=4,dim=256,rotary=64;std::array<int,3> sections{11,11,10};double theta=1.e7,eps=1.e-6;
    bool low=false,pre=false;int64_t width=106823680,key=2482176,value=4579328;
    bool compressed=true,external=false,f32=true,diagnostic=false,legacy=false,overlay=false,reference=false,fused=true,joined=true;
};
int main(){
    int cases=0,selected=0,rejected=0;
    auto check=[&](const char* flag,Profile p,bool valid){
        ++cases;
        try{
            RequireFullChunkStageWitness(flag,p.precision,p.compressed,p.external,p.f32,p.diagnostic,p.legacy,p.overlay,p.reference,p.fused,p.joined);
            RequireFullChunkStageGeometry(p.shape,p.layer,p.heads,p.kv,p.dim,p.rotary,p.sections,p.theta,p.eps,p.low,p.pre,p.width,p.key,p.value);
            if(!valid)throw std::logic_error("Invalid full chunk witness profile admitted");++selected;
        }catch(const std::invalid_argument&){if(valid)throw;++rejected;}
    };
    try{
        for(int tokens:{32,128}){
            Profile base;base.shape={1,tokens,5120};base.low=tokens==32;base.pre=tokens==128;
            for(const char* flag:{static_cast<const char*>(nullptr),"","0","true","01"," 1","1 ","-1","2","1"})check(flag,base,flag&&std::strcmp(flag,"1")==0);
            for(auto shape:{std::vector<int>{},{1,32},{1,32,5120,1},{0,tokens,5120},{2,tokens,5120},{4,tokens,5120},{8,tokens,5120},{1,0,5120},{1,16,5120},{1,129,5120},{1,tokens,5119},{1,tokens,-1}}){auto p=base;p.shape=shape;check("1",p,false);}
            for(const char* layer:{"","model.language_model.layers.2","model.language_model.layers.7"}){auto p=base;p.layer=layer;check("1",p,false);}
            for(const char* precision:{"","bf16","ct_int4_g128"}){auto p=base;p.precision=precision;check("1",p,false);}
            for(double v:{0.,-1.,1.e6,std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity()}){auto p=base;p.theta=v;check("1",p,false);}
            for(double v:{0.,-1.,1.e-5,std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity()}){auto p=base;p.eps=v;check("1",p,false);}
            for(int field=0;field<4;++field)for(int v:{-1,0,25}){auto p=base;int* slots[]={&p.heads,&p.kv,&p.dim,&p.rotary};*slots[field]=v;check("1",p,false);}
            for(int field=0;field<3;++field)for(int delta:{-1,1}){auto p=base;int64_t* slots[]={&p.width,&p.key,&p.value};*slots[field]+=delta;check("1",p,false);}
            for(auto sections:{std::array<int,3>{10,11,11},std::array<int,3>{0,0,0},std::array<int,3>{-1,11,10}}){auto p=base;p.sections=sections;check("1",p,false);}
            {auto p=base;p.low=!p.low;check("1",p,false);}{auto p=base;p.pre=!p.pre;check("1",p,false);}
            for(int field=0;field<9;++field){auto p=base;bool* slots[]={&p.compressed,&p.external,&p.f32,&p.diagnostic,&p.legacy,&p.overlay,&p.reference,&p.fused,&p.joined};*slots[field]=!*slots[field];check("1",p,false);}
        }
        if(cases!=142||selected!=2||rejected!=140)throw std::logic_error("Full chunk witness CPU accounting changed");
        std::printf("{\"status\":\"full_chunk_stage_profile_CPU_contract_passed\",\"cases\":%d,\"selected\":%d,\"rejected\":%d,\"GPU_or_model_executed\":false}\n",cases,selected,rejected);return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
