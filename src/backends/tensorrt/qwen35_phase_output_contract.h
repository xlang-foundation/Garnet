// SPDX-License-Identifier: Apache-2.0
#pragma once
// Pure metadata validation. No allocation, launch, IO, alias permission or
// serving integration. Validate the complete contract before device work.
#include "../../cuda/qwen35_packed_commit.cuh"
#include <nlohmann/json.hpp>
#include <climits>
#include <stdexcept>
#include <string>
#include <vector>

namespace Garnet {
struct Qwen35PhaseOutputContract {
    Qwen35PackedCommitConfig config{};
    std::vector<Qwen35PackedLinearLayout> linear;
    std::vector<int64_t> full;
    std::vector<int> parameter_layers;
    int first_layer=0,layer_count=0;
    bool replay_only=false;
};
namespace PhaseOutputDetail {
inline void require(bool okay,const char* why) {
    if(!okay)throw std::invalid_argument(why);
}
inline int64_t integer(const nlohmann::json& j,int64_t lo,int64_t hi,const char* why) {
    require(j.is_number_integer(),why);
    if(j.is_number_unsigned())require(j.get<uint64_t>()<=uint64_t(hi),why);
    auto v=j.get<int64_t>();require(v>=lo&&v<=hi,why);return v;
}
inline bool boolean(const nlohmann::json& j,const char* why) {
    require(j.is_boolean(),why);return j.get<bool>();
}
inline void keys(const nlohmann::json& j,const std::vector<std::string>& allowed) {
    require(j.is_object(),"Phase metadata must be an object");
    for(auto it=j.begin();it!=j.end();++it){
        bool found=false;for(const auto& key:allowed)found|=it.key()==key;
        require(found,"Unknown phase metadata field");
    }
}
inline void shape(const nlohmann::json& j,const std::vector<int64_t>& expected) {
    require(j.is_array()&&j.size()==expected.size(),"Phase canonical shape rank mismatch");
    for(size_t i=0;i<expected.size();++i)
        require(integer(j[i],1,INT32_MAX,"Phase shape must contain bounded integers")==expected[i],"Phase canonical shape mismatch");
}
struct Layer {std::string kind;int64_t offset,w1,w2;};
}

inline Qwen35PhaseOutputContract ParseQwen35PhaseOutputContract(const nlohmann::json& j) {
    using namespace PhaseOutputDetail;
    keys(j,{"phase_output_layout","config","layout","batch","tokens","capacity","state_width","hidden_width",
        "output_state_width","status_offset","packed_width","compact_kv_outputs","copy_only","replay_only_linear_state","copy_layer_range"});
    require(j.at("phase_output_layout").is_string()&&j.at("phase_output_layout")=="qwen35-phase-outputs-v1","Unknown phase output protocol");
    Qwen35PhaseOutputContract out;auto& c=out.config;const auto& m=j.at("config");
    c.batch=int(integer(j.at("batch"),1,8,"Invalid phase batch"));
    require(c.batch==1||c.batch==2||c.batch==4||c.batch==8,"Unsupported phase batch");
    c.capacity=int(integer(j.at("capacity"),32,8192,"Invalid phase capacity"));
    c.key_heads=int(integer(m.at("linear_num_key_heads"),1,64,"Invalid phase key heads"));
    c.value_heads=int(integer(m.at("linear_num_value_heads"),1,64,"Invalid phase value heads"));
    require(c.value_heads%c.key_heads==0,"Invalid phase head grouping");
    require(integer(m.at("linear_key_head_dim"),128,128,"Phase replay requires128-wide keys")==128&&
        integer(m.at("linear_value_head_dim"),128,128,"Phase replay requires128-wide values")==128,"Invalid phase linear dimensions");
    c.kernel=int(integer(m.at("linear_conv_kernel_dim"),1,4,"Invalid phase convolution kernel"));
    c.kv_heads=int(integer(m.at("num_key_value_heads"),1,32,"Invalid phase KV heads"));
    c.head_dim=int(integer(m.at("head_dim"),1,256,"Invalid phase head dimension"));
    int hidden=int(integer(m.at("hidden_size"),1,16384,"Invalid phase hidden size"));
    out.replay_only=boolean(j.at("replay_only_linear_state"),"Phase replay mode must be boolean");
    c.copy_only=boolean(j.at("copy_only"),"Phase copy mode must be boolean");
    require(c.copy_only==!out.replay_only,"Inconsistent phase commit mode");
    require(boolean(j.at("compact_kv_outputs"),"Phase compact mode must be boolean"),"Phase outputs require compact KV");c.compact_kv=1;
    c.tokens=int(integer(j.at("tokens"),1,c.copy_only?128:16,"Invalid phase token count"));
    require(c.tokens<=c.capacity,"Phase token count exceeds cache capacity");
    const auto& kinds=m.at("layer_types");
    require(kinds.is_array()&&kinds.size()>=1&&kinds.size()<=128,"Invalid phase layer list");
    require(integer(m.at("num_hidden_layers"),1,128,"Invalid phase layer count")==int64_t(kinds.size()),"Phase layer count mismatch");
    std::vector<Layer> canonical;int64_t state=0;const int channels=(2*c.key_heads+c.value_heads)*128;
    for(const auto& kindValue:kinds){
        require(kindValue.is_string(),"Phase layer kind must be a string");auto kind=kindValue.get<std::string>();
        int64_t w1,w2;
        if(kind=="linear_attention"){w1=int64_t(channels)*c.kernel;w2=int64_t(c.value_heads)*128*128;}
        else{require(kind=="full_attention","Unknown phase layer kind");w1=w2=int64_t(c.kv_heads)*c.capacity*c.head_dim;}
        canonical.push_back({kind,state,w1,w2});state+=w1+w2;require(state<=INT32_MAX,"Phase canonical width overflow");
    }
    c.state_width=integer(j.at("state_width"),1,INT32_MAX,"Invalid phase state width");
    require(c.state_width==state,"Phase canonical width mismatch");
    c.hidden_width=integer(j.at("hidden_width"),1,INT32_MAX,"Invalid phase hidden width");
    require(c.hidden_width==int64_t(hidden)*c.tokens,"Phase hidden width mismatch");
    if(out.replay_only){
        require(!j.contains("copy_layer_range"),"Decode replay cannot select a partial layer range");
        out.first_layer=0;out.layer_count=int(kinds.size());
    }else{
        const auto& range=j.at("copy_layer_range");require(range.is_array()&&range.size()==2,"Invalid phase copy range");
        out.first_layer=int(integer(range[0],0,int64_t(kinds.size())-1,"Invalid phase first layer"));
        out.layer_count=int(integer(range[1],1,int64_t(kinds.size())-out.first_layer,"Invalid phase layer span"));
    }
    const auto& layers=j.at("layout");require(layers.is_array()&&layers.size()==size_t(out.layer_count),"Incomplete phase layout");
    int64_t output=0;int fullCount=0;
    for(int local=0;local<out.layer_count;++local){
        const int index=out.first_layer+local;const auto& expected=canonical[index];const auto& l=layers[local];
        keys(l,{"index","kind","offset","state1_shape","state2_shape","state1_width","state2_width",
            "output_offset","output_state1_width","output_state2_width","control_index","payload_width","payload_offset"});
        require(integer(l.at("index"),0,127,"Invalid phase layer index")==index&&l.at("kind").is_string()&&l.at("kind")==expected.kind,"Phase layer order/kind mismatch");
        require(integer(l.at("offset"),0,INT32_MAX,"Invalid phase canonical offset")==expected.offset,"Phase canonical offset mismatch");
        require(integer(l.at("state1_width"),1,INT32_MAX,"Invalid phase state1 width")==expected.w1&&
            integer(l.at("state2_width"),1,INT32_MAX,"Invalid phase state2 width")==expected.w2,"Phase state width mismatch");
        if(expected.kind=="linear_attention"){
            shape(l.at("state1_shape"),{c.batch,channels,c.kernel});shape(l.at("state2_shape"),{c.batch,c.value_heads,128,128});
            require(!l.contains("control_index"),"Linear phase state cannot own KV status");
        }else{
            shape(l.at("state1_shape"),{c.batch,c.kv_heads,c.capacity,c.head_dim});shape(l.at("state2_shape"),{c.batch,c.kv_heads,c.capacity,c.head_dim});
            require(integer(l.at("control_index"),0,127,"Invalid phase status index")==fullCount,"Phase status index mismatch");++fullCount;
        }
        int64_t w1=expected.kind=="full_attention"?int64_t(c.kv_heads)*c.tokens*c.head_dim:out.replay_only?0:expected.w1;
        int64_t w2=expected.kind=="full_attention"?w1:out.replay_only?0:expected.w2;
        require(integer(l.at("output_offset"),0,INT32_MAX,"Invalid phase output offset")==output&&
            integer(l.at("output_state1_width"),0,INT32_MAX,"Invalid phase output1 width")==w1&&
            integer(l.at("output_state2_width"),0,INT32_MAX,"Invalid phase output2 width")==w2,"Phase output interval mismatch");
        if(expected.kind=="full_attention"){out.full.push_back(expected.offset);out.full.push_back(c.hidden_width+output);}
        output+=w1+w2;require(output<=INT32_MAX,"Phase output width overflow");
    }
    c.output_state_width=integer(j.at("output_state_width"),1,INT32_MAX,"Invalid phase output-state width");
    require(output==c.output_state_width&&output<=c.state_width,"Phase output-state coverage mismatch");
    int64_t payload=c.hidden_width+output;
    for(int local=0;local<out.layer_count;++local){
        const auto& l=layers[local];int index=out.first_layer+local;const auto& expected=canonical[index];
        const bool linear=expected.kind=="linear_attention";
        int64_t width=linear&&out.replay_only?int64_t(c.tokens)*(channels+c.value_heads*128+2*c.value_heads):0;
        require(integer(l.at("payload_width"),0,INT32_MAX,"Invalid phase payload width")==width,"Phase retained payload size mismatch");
        if(width){
            require(integer(l.at("payload_offset"),0,INT32_MAX,"Invalid phase payload offset")==payload,"Phase payload interval mismatch");
            out.linear.push_back({expected.offset,payload});out.parameter_layers.push_back(index);payload+=width;
        }else{
            require(!l.contains("payload_offset"),"Unexpected phase payload offset");
            if(linear)out.linear.push_back({expected.offset,c.hidden_width+l.at("output_offset").get<int64_t>()});
        }
        require(payload<=INT32_MAX,"Phase payload width overflow");
    }
    c.linear_layers=int(out.linear.size());c.full_layers=fullCount;
    require(c.linear_layers>=1&&c.linear_layers<=64&&c.full_layers<=64&&(!out.replay_only||c.full_layers>=1),"Unsupported phase layer counts");
    c.status_offset=integer(j.at("status_offset"),1,INT32_MAX,"Invalid phase status offset");
    c.packed_width=integer(j.at("packed_width"),1,INT32_MAX,"Invalid phase packed width");
    require(c.status_offset==payload&&c.packed_width==payload+c.full_layers,"Incomplete phase payload/status coverage");
    return out;
}
}
