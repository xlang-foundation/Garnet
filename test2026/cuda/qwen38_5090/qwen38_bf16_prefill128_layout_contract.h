// SPDX-License-Identifier: Apache-2.0
#pragma once
// Independent CPU contract only. No old parser/profile/serving admission changes.
#include "qwen38_bf16_low_batch_kv_layout_contract.h"
namespace Garnet {
struct BF16Prefill128KVLayoutSpec {int batch=0,chunk=0;int64_t canonical=0,target=0,replay=0;};
namespace BF16Prefill128KVContract {
using namespace BF16LowBatchKVContract;
inline BF16Prefill128KVLayoutSpec Parse(const Json& j){
    require(j.is_object()&&j.size()==17,"T128 layout schema mismatch");
    string(j,"contract","garnet_qwen38_bf16_prefill128_delta_study_v1");
    integer(j,"batch",1);integer(j,"chunk",128);integer(j,"capacity",4096);
    integer(j,"hidden",5120);integer(j,"hidden_width",655360);
    integer(j,"canonical_state_width",106823680);integer(j,"linear_replay_width",39714816);
    integer(j,"target_state_end",44564480);integer(j,"payload_width",0);integer(j,"output_width",44564480);
    string(j,"canonical_dtype","fp32_carrier_with_bf16_kv");string(j,"kv_dtype","bf16");string(j,"linear_dtype","fp32");
    require(j.at("retain_linear_inputs").is_boolean()&&!j.at("retain_linear_inputs").get<bool>(),"T128 prefill excludes retained replay payload");
    require(j.at("runtime_qualified").is_boolean()&&!j.at("runtime_qualified").get<bool>(),"Metadata never qualifies a runtime");
    const auto& layers=j.at("layers");require(layers.is_array()&&layers.size()==64,"T128 layer count mismatch");
    int64_t canonical=0,target=655360,replay=0;
    for(int i=0;i<64;++i){
        const auto& l=layers[i];const bool full=i%4==3;
        require(l.is_object()&&l.size()==(full?15:14),"T128 layer schema mismatch");
        const int64_t first=full?2097152:40960,second=full?2097152:786432;
        const int64_t out1=full?131072:first,out2=full?131072:second;
        integer(l,"index",i);string(l,"kind",full?"full_attention":"linear_attention");
        integer(l,"offset",canonical);integer(l,"state1_width",first);integer(l,"state2_width",second);
        integer(l,"target_offset",target);integer(l,"output_state1_width",out1);integer(l,"output_state2_width",out2);
        integer(l,"payload_width",0);require(l.at("payload_offset").is_null(),"T128 payload must be absent");
        string(l,"state_precision",full?"packed_bf16":"fp32");
        if(full){
            shape(l,"state1_shape",{1,4,4096,128});shape(l,"state2_shape",{1,4,4096,128});shape(l,"logical_kv_shape",{1,4,4096,256});
            require(l.at("replay_offset").is_null(),"T128 full attention has no linear replay");
        }else{
            shape(l,"state1_shape",{1,10240,4});shape(l,"state2_shape",{1,48,128,128});integer(l,"replay_offset",replay);
            replay+=first+second;
        }
        canonical+=first+second;target+=out1+out2;
    }
    require(canonical==106823680&&replay==39714816&&target==44564480,"T128 complete layout interval mismatch");
    return {1,128,canonical,target,replay};
}
}
}
