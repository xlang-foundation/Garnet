// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <array>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
namespace Garnet {
// Bump this schema when a policy's serialized engine meaning changes.
inline constexpr const char* kTRTGraphPolicySchema="qwen38_graph_policy_v1";
inline constexpr std::array<const char*,8> kTRTCanonicalFP8Policies={{
    "GARNET_FP8_FULL_ATTN_M32_CANONICAL","GARNET_FP8_GDN_M32_CANONICAL",
    "GARNET_FP8_GDN_M128_CANONICAL","GARNET_FP8_GDN_Z128_CANONICAL",
    "GARNET_FP8_GDN_Z128_RECONSTRUCT","GARNET_FP8_FULL_ATTN_KV128_CANONICAL",
    "GARNET_FP8_FULL_ATTN_Q128_CANONICAL","GARNET_FP8_FULL_ATTN_O128_CANONICAL"}};
inline constexpr std::array<const char*,4> kTRTAdditionalGraphPolicies={{
    "GARNET_FULL_CHUNK_STAGE_WITNESS","GARNET_PARALLEL_BATCH_TOP1",
    "GARNET_QWEN38_BF16_LOW_BATCH_TILED_SCORES","GARNET_NVFP4_DOWN_CANDIDATE"}};
template<class Lookup> std::string TRTGraphPolicyFingerprint(Lookup lookup){
    std::ostringstream material;material<<kTRTGraphPolicySchema<<'\n';
    std::array<bool,8> enabled{};
    for(size_t i=0;i<kTRTCanonicalFP8Policies.size();++i){
        const auto* name=kTRTCanonicalFP8Policies[i];const auto* value=lookup(name);
        if(value&&*value&&std::strcmp(value,"0")&&std::strcmp(value,"1"))
            throw std::invalid_argument(std::string(name)+" must be0 or1");
        enabled[i]=value&&!std::strcmp(value,"1");
        material<<name<<':'<<(enabled[i]?'1':'0')<<'\n';
    }
    if(enabled[4]&&!enabled[3])
        throw std::invalid_argument("Z128 reconstruction requires explicit canonical Z128 policy");
    // These controls retain their existing parser semantics. Length prefixes
    // make arbitrary environment strings unambiguous without reinterpreting them.
    for(const auto* name:kTRTAdditionalGraphPolicies){
        const auto* value=lookup(name);if(!value||!*value)value="0";
        material<<name<<':'<<std::strlen(value)<<':'<<value<<'\n';
    }
    return material.str();
}
inline std::string TRTGraphPolicyFingerprint(){
    return TRTGraphPolicyFingerprint([](const char* name){return std::getenv(name);});
}
}
