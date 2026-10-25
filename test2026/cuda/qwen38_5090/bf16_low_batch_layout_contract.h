// SPDX-License-Identifier: Apache-2.0
#pragma once
// Pure CPU metadata gate. No CUDA/runtime imports, allocations or writes.
#include <nlohmann/json.hpp>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
namespace Garnet {
struct BF16LowBatchKVLayoutSpec {int batch=0,chunk=0;int64_t canonical=0,target=0,replay=0;};
namespace BF16LowBatchKVContract {
using Json=nlohmann::json;
inline void require(bool v,const char* text){if(!v)throw std::invalid_argument(text);}
inline void integer(const Json& j,const char* key,int64_t wanted){
    const auto& v=j.at(key);require(v.is_number_integer()&&v==wanted,"BF16 KV integer field mismatch");
}
inline void string(const Json& j,const char* key,const char* wanted){
    const auto& v=j.at(key);require(v.is_string()&&v==wanted,"BF16 KV string field mismatch");
}
inline void shape(const Json& j,const char* key,std::initializer_list<int64_t> wanted){
    const auto& v=j.at(key);require(v.is_array()&&v.size()==wanted.size(),"BF16 KV shape rank mismatch");
    size_t i=0;for(auto n:wanted){require(v[i].is_number_integer()&&v[i]==n,"BF16 KV shape dimension mismatch");++i;}
}
inline BF16LowBatchKVLayoutSpec Parse(const Json& j){
    require(j.is_object()&&j.size()==17,"BF16 KV summary schema mismatch");
    string(j,"contract","garnet_qwen38_bf16_low_batch_delta_study_v1");
    require(j.at("batch").is_number_integer()&&j.at("batch")==1,"BF16 KV batch unsupported");
    require(j.at("chunk").is_number_integer()&&(j.at("chunk")==4||j.at("chunk")==32),"BF16 KV chunk unsupported");
    const int b=j.at("batch").get<int>(),t=j.at("chunk").get<int>();const bool retain=t==4;
    const int64_t stateEnd=int64_t(t)*5120+39714816+int64_t(16)*2*t*1024;
    const int64_t payload=retain?3164160:0,targetWidth=stateEnd+payload;
    integer(j,"capacity",4096);integer(j,"hidden",5120);integer(j,"hidden_width",int64_t(t)*5120);
    integer(j,"canonical_state_width",106823680);integer(j,"linear_replay_width",39714816);
    integer(j,"target_state_end",stateEnd);integer(j,"payload_width",payload);integer(j,"output_width",targetWidth);
    string(j,"canonical_dtype","fp32_carrier_with_bf16_kv");string(j,"kv_dtype","bf16");string(j,"linear_dtype","fp32");
    require(j.at("retain_linear_inputs").is_boolean()&&j.at("retain_linear_inputs")==retain,"BF16 KV replay mode mismatch");
    // Documentary flag is deliberately false in this research-layout dialect;
    // building a helper never qualifies a model or enables a serving profile.
    require(j.at("runtime_qualified").is_boolean()&&j.at("runtime_qualified")==false,"BF16 KV study flag mismatch");
    const auto& layers=j.at("layers");require(layers.is_array()&&layers.size()==64,"BF16 KV layer count mismatch");
    int64_t canonical=0,target=int64_t(t)*5120,replay=0,payloadAt=stateEnd;
    for(int i=0;i<64;++i){
        const auto& l=layers[i];const bool full=i%4==3;
        require(l.is_object()&&l.size()==(full?15:14),"BF16 KV layer schema mismatch");
        const int64_t first=full?2097152:40960,second=full?2097152:786432;
        integer(l,"index",i);string(l,"kind",full?"full_attention":"linear_attention");
        integer(l,"offset",canonical);integer(l,"state1_width",first);integer(l,"state2_width",second);
        integer(l,"target_offset",target);integer(l,"output_state1_width",full?int64_t(t)*1024:first);
        integer(l,"output_state2_width",full?int64_t(t)*1024:second);
        string(l,"state_precision",full?"packed_bf16":"fp32");
        if(full){
            shape(l,"state1_shape",{b,4,4096,128});shape(l,"state2_shape",{b,4,4096,128});shape(l,"logical_kv_shape",{b,4,4096,256});
            require(l.at("replay_offset").is_null()&&l.at("payload_offset").is_null(),"BF16 KV full layer must not carry replay data");integer(l,"payload_width",0);
        }else{
            shape(l,"state1_shape",{b,10240,4});shape(l,"state2_shape",{b,48,128,128});integer(l,"replay_offset",replay);
            integer(l,"payload_width",retain?65920:0);
            if(retain){integer(l,"payload_offset",payloadAt);payloadAt+=65920;}
            else require(l.at("payload_offset").is_null(),"BF16 KV prefill must not carry replay payload");
            replay+=first+second;
        }
        canonical+=first+second;target+=full?int64_t(2)*t*1024:first+second;
    }
    require(canonical==106823680&&replay==39714816&&target==stateEnd&&payloadAt==targetWidth,"BF16 KV complete interval mismatch");
    return {b,t,canonical,targetWidth,replay};
}
}
}
