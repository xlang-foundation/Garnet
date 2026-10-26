// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
namespace Garnet {
struct GDNChunkStageProfile {
    std::vector<int> hidden,history,state,active;
    std::string layer,weightProfile;
    std::vector<std::string> externalNames;
    std::array<bool,4> floatInputs{};
    bool compressed=false,bf16Bindings=false;
    int keyHeads=0,valueHeads=0,keyDim=0,valueDim=0;
    int fused=0,retain=0,register128=0,joinedInputs=0,joinedMLP=0,workspace=0;
    float epsilon=0;
};
inline bool SelectGDNChunkStageProfile(bool integerFlag,int64_t enabled,bool legacyDiagnostic,const GDNChunkStageProfile& p){
    if(!integerFlag||(enabled!=0&&enabled!=1))throw std::invalid_argument("GDN chunk stage opt-in requires integer0 or1");
    if(!enabled)return false;
    const bool layer=p.layer=="model.language_model.layers.0"||p.layer=="model.language_model.layers.1";
    const bool shape=p.hidden==std::vector<int>{1,32,5120}||p.hidden==std::vector<int>{1,128,5120};
    if(legacyDiagnostic||!layer||!shape||p.weightProfile!="modelopt_mixed"||!p.compressed||p.bf16Bindings||
       p.history!=std::vector<int>{1,10240,4}||p.state!=std::vector<int>{1,48,128,128}||
       p.active!=std::vector<int>{1,p.hidden[1]}||p.floatInputs!=std::array<bool,4>{true,true,true,true}||
       p.keyHeads!=16||p.valueHeads!=48||p.keyDim!=128||p.valueDim!=128||p.epsilon!=1.e-6f||
       p.fused!=1||p.retain!=0||p.register128!=1||p.joinedInputs!=0||p.joinedMLP!=1||p.workspace!=0)
        throw std::invalid_argument("GDN chunk stage requires pinned shared joined B1 T32/T128 layer0/1 profile");
    const std::array<const char*,5> suffixes={".mlp.gate_up_proj.weight",".mlp.down_proj.weight",".linear_attn.in_proj_qkv.weight",".linear_attn.in_proj_z.weight",".linear_attn.out_proj.weight"};
    if(p.externalNames.size()!=suffixes.size())throw std::invalid_argument("GDN chunk stage needs five exact external projections");
    for(const auto* suffix:suffixes){
        const auto wanted=p.layer+suffix;int found=0;
        for(const auto& name:p.externalNames)if(name==wanted)++found;
        if(found!=1)throw std::invalid_argument("GDN chunk stage external projection identity differs");
    }
    return true;
}
}
