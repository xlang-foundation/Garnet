// SPDX-License-Identifier: Apache-2.0
#include "graph_policy_fingerprint.h"
#include <iostream>
#include <map>
#include <set>
int main(){
    using namespace Garnet;using Env=std::map<std::string,std::string>;int checks=0,rejected=0;
    auto require=[&](bool value){++checks;if(!value)throw std::runtime_error("Policy fingerprint contract failed");};
    auto key=[&](const Env& env){return TRTGraphPolicyFingerprint([&](const char* name)->const char*{auto it=env.find(name);return it==env.end()?nullptr:it->second.c_str();});};
    const auto baseline=key({});require(baseline.find("qwen38_graph_policy_v1\n")==0);
    Env all0,empty;for(auto* name:kTRTCanonicalFP8Policies){all0[name]="0";empty[name]="";}
    for(auto* name:kTRTAdditionalGraphPolicies){all0[name]="0";empty[name]="";}
    require(key(all0)==baseline);require(key(empty)==baseline);std::set<std::string> keys{baseline};
    for(auto* name:kTRTCanonicalFP8Policies){Env env{{name,"1"}};if(std::string(name)=="GARNET_FP8_GDN_Z128_RECONSTRUCT")env["GARNET_FP8_GDN_Z128_CANONICAL"]="1";require(keys.insert(key(env)).second);require(key(env)==key(env));}
    for(auto* name:kTRTAdditionalGraphPolicies){require(keys.insert(key({{name,"1"}})).second);require(key({{name,""}})==baseline);}
    for(auto* name:kTRTCanonicalFP8Policies)for(auto* value:{"2","true","01"," 1","1\n"}){
        try{key({{name,value}});return 2;}catch(const std::invalid_argument& error){require(std::string(error.what())==std::string(name)+" must be0 or1");++rejected;}
    }
    for(const char* canonical:{static_cast<const char*>(nullptr),"","0"}){
        Env env{{"GARNET_FP8_GDN_Z128_RECONSTRUCT","1"}};if(canonical)env["GARNET_FP8_GDN_Z128_CANONICAL"]=canonical;
        try{key(env);return 3;}catch(const std::invalid_argument& error){require(std::string(error.what())=="Z128 reconstruction requires explicit canonical Z128 policy");++rejected;}
    }
    Env a{{"GARNET_PARALLEL_BATCH_TOP1","x\nGARNET_QWEN38_BF16_LOW_BATCH_TILED_SCORES:1:1"}},b{{"GARNET_PARALLEL_BATCH_TOP1","x"},{"GARNET_QWEN38_BF16_LOW_BATCH_TILED_SCORES","1"}};
    require(key(a)!=key(b));Env reverse;for(auto it=all0.rbegin();it!=all0.rend();++it)reverse.insert(*it);require(key(reverse)==baseline);
    std::cout<<"{\"status\":\"TRT_graph_policy_header_CPU_contract_passed\",\"checks\":"<<checks<<",\"strict_policies\":8,\"additional_policies\":4,\"distinct_keys\":"<<keys.size()<<",\"rejections\":"<<rejected<<",\"GPU_or_model_executed\":false}\n";
}
