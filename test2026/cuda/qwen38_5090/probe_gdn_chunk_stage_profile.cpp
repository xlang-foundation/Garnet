// SPDX-License-Identifier: Apache-2.0
// Pure CPU profile contract; no CUDA, TensorRT, Garnet DLL or enqueue calls.
#include "qwen38_gdn_chunk_stage_profile.h"
#include <cstdio>
#include <limits>
#include <utility>
using namespace Garnet;
static void require(bool value){if(!value)throw std::runtime_error("GDN chunk stage contract failed");}
static GDNChunkStageProfile canonical(int tokens,int layer){
    GDNChunkStageProfile p;p.hidden={1,tokens,5120};p.history={1,10240,4};p.state={1,48,128,128};p.active={1,tokens};
    p.layer="model.language_model.layers."+std::to_string(layer);p.weightProfile="modelopt_mixed";p.compressed=true;p.floatInputs={true,true,true,true};
    p.keyHeads=16;p.valueHeads=48;p.keyDim=p.valueDim=128;p.fused=p.register128=p.joinedMLP=1;p.epsilon=1.e-6f;
    for(const char* suffix:{".mlp.gate_up_proj.weight",".mlp.down_proj.weight",".linear_attn.in_proj_qkv.weight",".linear_attn.in_proj_z.weight",".linear_attn.out_proj.weight"})p.externalNames.push_back(p.layer+suffix);
    return p;
}
int main(){
    try{
        int profiles=0,rejections=0,flagRejections=0;
        for(int tokens:{32,128})for(int layer:{0,1}){auto p=canonical(tokens,layer);require(SelectGDNChunkStageProfile(true,1,false,p));require(!SelectGDNChunkStageProfile(true,0,false,p));++profiles;}
        const auto base=canonical(128,1);std::vector<GDNChunkStageProfile> bad;
        for(auto dims:{std::vector<int>{},std::vector<int>{1,128},std::vector<int>{4,32,5120},std::vector<int>{1,16,5120},std::vector<int>{1,128,5121}}){auto p=base;p.hidden=dims;bad.push_back(p);}
        for(auto dims:{std::vector<int>{},std::vector<int>{1,10240,3},std::vector<int>{2,10240,4}}){auto p=base;p.history=dims;bad.push_back(p);}
        for(auto dims:{std::vector<int>{},std::vector<int>{1,48,128,127},std::vector<int>{1,16,128,128}}){auto p=base;p.state=dims;bad.push_back(p);}
        for(auto dims:{std::vector<int>{},std::vector<int>{1,32},std::vector<int>{2,128}}){auto p=base;p.active=dims;bad.push_back(p);}
        for(int i=0;i<4;++i){auto p=base;p.floatInputs[i]=false;bad.push_back(p);}
        for(const char* layer:{"model.language_model.layers.2","model.language_model.layers.3","0"}){auto p=base;p.layer=layer;bad.push_back(p);}
        for(const char* profile:{"bf16","ct_int4_g128",""}){auto p=base;p.weightProfile=profile;bad.push_back(p);}
        for(int fault=0;fault<5;++fault){auto p=base;if(fault==0)p.externalNames.pop_back();else if(fault==1)p.externalNames[0]="wrong";else if(fault==2)p.externalNames[0]=p.externalNames[1];else if(fault==3)p.externalNames[0]="model.language_model.layers.0.mlp.gate_up_proj.weight";else p.externalNames.push_back("extra");bad.push_back(p);}
        {auto p=base;p.compressed=false;bad.push_back(p);}{auto p=base;p.bf16Bindings=true;bad.push_back(p);}
        for(int field=0;field<4;++field){auto p=base;int* values[]={&p.keyHeads,&p.valueHeads,&p.keyDim,&p.valueDim};++*values[field];bad.push_back(p);}
        for(int field=0;field<6;++field){auto p=base;int* values[]={&p.fused,&p.retain,&p.register128,&p.joinedInputs,&p.joinedMLP,&p.workspace};++*values[field];bad.push_back(p);}
        for(int field=0;field<3;++field){auto p=base;int* values[]={&p.fused,&p.register128,&p.joinedMLP};*values[field]=0;bad.push_back(p);}
        {auto p=base;p.workspace=-1;bad.push_back(p);}
        for(float epsilon:{0.f,-1.f,1.e-5f,std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity()}){auto p=base;p.epsilon=epsilon;bad.push_back(p);}
        require(bad.size()==50);
        for(const auto& p:bad){bool rejected=false;try{SelectGDNChunkStageProfile(true,1,false,p);}catch(const std::invalid_argument&){rejected=true;}require(rejected);++rejections;}
        for(auto flag:{int64_t(-1),int64_t(2),int64_t(128),(std::numeric_limits<int64_t>::max)()}){bool rejected=false;try{SelectGDNChunkStageProfile(true,flag,false,base);}catch(const std::invalid_argument&){rejected=true;}require(rejected);++flagRejections;}
        for(int64_t flag:{0,1}){bool rejected=false;try{SelectGDNChunkStageProfile(false,flag,false,base);}catch(const std::invalid_argument&){rejected=true;}require(rejected);++flagRejections;}
        bool crossed=false;try{SelectGDNChunkStageProfile(true,1,true,base);}catch(const std::invalid_argument&){crossed=true;}require(crossed);
        require(!SelectGDNChunkStageProfile(true,0,true,GDNChunkStageProfile{}));
        std::printf("{\"status\":\"gdn_chunk_stage_cpu_profile_passed\",\"profiles\":%d,\"profile_rejections\":%d,\"flag_rejections\":%d,\"legacy_conflict_rejections\":1,\"legacy_default_unchanged\":true,\"local_or_remote_gpu_calls\":0,\"native_frontend_exercised\":false,\"primary_witness_qualified\":false,\"model_or_performance_qualified\":false}\n",profiles,rejections,flagRejections);return 0;
    }catch(const std::exception& error){std::fprintf(stderr,"%s\n",error.what());return 1;}
}
