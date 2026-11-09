// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <set>
#include <vector>
namespace Garnet {
// Slot IDs are immutable logical roots, never inferred from pooled GPU storage.
struct KVHoistNode {
    enum Kind {Other,Attention,Commit};Kind kind=Other;bool resident=true;
    std::vector<size_t> reads,writes,args,dims;
};
struct KVHoistPlan {size_t attention=SIZE_MAX;std::array<size_t,2> commits{SIZE_MAX,SIZE_MAX};bool Valid()const{return attention!=SIZE_MAX;}};
inline KVHoistPlan ProveKVCommitHoist(const std::vector<KVHoistNode>& nodes,
    const std::array<size_t,6>& inputs,const std::array<std::vector<size_t>,2>& outputs,size_t batch,size_t chunk,size_t maximumChunk=128) {
    KVHoistPlan none,plan;
    if((maximumChunk!=128&&maximumChunk!=512)||!batch||batch>4||chunk<2||chunk>maximumChunk||batch>512/chunk||
       std::set<size_t>(inputs.begin(),inputs.end()).size()!=6)return none;
    auto has=[](const auto& a,size_t id){return std::find(a.begin(),a.end(),id)!=a.end();};
    for(size_t i=0;i<nodes.size();++i)if(nodes[i].resident&&nodes[i].kind==KVHoistNode::Attention) {
        if(plan.Valid())return none;const auto& n=nodes[i];
        if(n.dims!=std::vector<size_t>{32,8,128,4096,chunk,0}||n.args.size()!=8||n.writes.size()!=1||
           n.args[3]!=inputs[1]||n.args[4]!=inputs[2]||n.args[5]!=inputs[4]||n.args[6]!=inputs[5])return none;
        plan.attention=i;
    }
    if(!plan.Valid())return none;const auto& a=nodes[plan.attention];
    if(std::set<size_t>{a.args[0],a.args[1],a.args[2]}.size()!=3)return none;
    for(size_t part=0;part<2;++part) {
        if(outputs[part].empty())return none;
        for(auto id:outputs[part])if(has(inputs,id)||has(outputs[1-part],id)||has(a.args,id))return none;
        size_t producers=0;
        for(size_t i=0;i<nodes.size();++i)if(nodes[i].resident) {
            const auto& n=nodes[i];
            if(has(n.writes,inputs[part+1]))return none;
            for(auto id:n.writes)if(has(outputs[part],id)) {
                if(++producers!=1||n.kind!=KVHoistNode::Commit||i<=plan.attention||n.writes.size()!=1||
                   n.dims!=std::vector<size_t>{8,128,4096,chunk}||
                   n.args!=std::vector<size_t>{a.args[part+1],inputs[part+1],inputs[4],inputs[5],inputs[3]})return none;
                plan.commits[part]=i;
            }
        }
        if(producers!=1)return none;
    }
    for(size_t role=0;role<3;++role){size_t producers=0;
        if(has(inputs,a.args[role]))return none;
        for(size_t i=0;i<nodes.size();++i)if(nodes[i].resident&&has(nodes[i].writes,a.args[role])) {
            if(i>=plan.attention||++producers!=1)return none;
        }
        if(producers!=1)return none;
    }
    for(size_t i=0;i<nodes.size();++i)if(nodes[i].resident) {
        const auto& n=nodes[i];
        for(size_t part=0;part<2;++part) {
            if(i>=plan.attention&&has(n.reads,inputs[part+1])&&i!=plan.attention&&i!=plan.commits[part])return none;
            if(i<plan.commits[part])for(auto id:outputs[part])if(has(n.reads,id))return none;
        }
        if(i>=plan.attention&&i<=std::max(plan.commits[0],plan.commits[1]))
            for(auto id:n.writes)if(id==a.args[1]||id==a.args[2]||id==inputs[3]||id==inputs[4]||id==inputs[5])return none;
    }
    return plan;
}
}
