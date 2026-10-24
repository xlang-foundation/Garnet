// SPDX-License-Identifier: Apache-2.0
#if !defined(GARNET_CPU_ONLY) && !defined(GARNET_METAL)
#include "cuda_kv_delta_commit.h"
#include "tensor_helper.h"
#include <nlohmann/json.hpp>
#include <cmath>
#include <vector>
namespace Garnet { namespace {
void require(bool okay,const char* message){if(!okay)throw X::Error(message);}
void tensor(const X3TensorInfo& v,int64_t width){
    require(!v.symbolic&&!v.readonly&&v.device_type==TensorHelper::CudaDevice&&v.dtype==X3_TENSOR_FLOAT32&&v.data&&
        v.rank==2&&v.shape[0]==4&&v.shape[1]==width&&v.byte_size>=uint64_t(4)*width*4,"KV delta requires matching FP32 CUDA matrices");
    require(!v.strides||(v.strides[0]==width*4&&v.strides[1]==4),"KV delta matrices must be contiguous");
}
}
void CUDAKVDeltaCommit::CloseUnlocked(){ready_=false;linear_=full_=X::Tensor();}
CUDAKVDeltaCommit::~CUDAKVDeltaCommit(){Close();}
void CUDAKVDeltaCommit::Close(){std::lock_guard<std::mutex> lock(mutex_);CloseUnlocked();}
bool CUDAKVDeltaCommit::Build(const std::string& specification){
    std::lock_guard<std::mutex> lock(mutex_);require(!ready_,"Close KV delta commit before rebuilding");
    const auto j=nlohmann::json::parse(specification);
    require(j.at("batch")==4&&j.at("capacity")==4096&&j.at("chunk")==4&&j.at("hidden_width")==20480&&
        j.at("canonical_state_width")==Qwen35KVCanonicalWidth&&j.at("output_width")==Qwen35KVTargetWidth&&
        j.at("linear_replay_width")==Qwen35KVReplayWidth&&j.at("payload_width")==3164160,"Unqualified KV delta summary");
    const auto& layers=j.at("layers");require(layers.is_array()&&layers.size()==64,"Unqualified KV delta layer list");
    std::vector<Qwen35KVDeltaLinearLayout> linear;std::vector<Qwen35KVDeltaFullLayout> full;
    int64_t canonical=0,target=20480,replay=0,payload=39866368;
    for(int i=0;i<64;++i){
        const auto& l=layers[i];const bool attention=i%4==3;
        const int64_t first=attention?4194304:40960,second=attention?4194304:786432;
        require(l.at("index")==i&&l.at("kind")== (attention?"full_attention":"linear_attention")&&l.at("canonical_offset")==canonical&&
            l.at("canonical_state1_width")==first&&l.at("canonical_state2_width")==second&&l.at("output_offset")==target&&
            l.at("output_state1_width")== (attention?4096:first)&&l.at("output_state2_width")== (attention?4096:second),"KV delta layer interval mismatch");
        if(attention){
            require(l.at("key_offset")==canonical&&l.at("value_offset")==canonical+first&&l.at("cache_batch_stride")==Qwen35KVCanonicalWidth&&l.at("cache_row_width")==first,"KV delta cache interval mismatch");
            full.push_back({canonical,target});
        }else{
            require(l.at("replay_offset")==replay&&l.at("payload_offset")==payload&&l.at("payload_width")==65920,"KV delta replay/payload mismatch");
            linear.push_back({canonical,target,replay,first+second});replay+=first+second;payload+=65920;
        }
        canonical+=first+second;target+=attention?8192:first+second;
    }
    require(canonical==Qwen35KVCanonicalWidth&&replay==Qwen35KVReplayWidth&&payload==Qwen35KVTargetWidth&&target==39866368,"KV delta complete interval mismatch");
    auto a=TensorHelper::CreateGPU(Host(),X3_TENSOR_INT64,{48,4},linear.data());
    auto b=TensorHelper::CreateGPU(Host(),X3_TENSOR_INT64,{16,2},full.data());
    linear_=a;full_=b;ready_=true;return true;
}
void CUDAKVDeltaCommit::Commit(X::Value packed,X::Value replayed,X::Value state,X::Value spans){
    std::lock_guard<std::mutex> lock(mutex_);require(ready_,"KV delta commit is closed");
    X::Tensor target(packed),linear(replayed),canonical(state);tensor(target.Info(),Qwen35KVTargetWidth);tensor(canonical.Info(),Qwen35KVCanonicalWidth);
    require(linear.Info().rank==2,"KV delta linear source rank mismatch");const int64_t width=linear.Info().shape[1];
    require(width==Qwen35KVTargetWidth||width==Qwen35KVReplayWidth,"KV delta linear source width mismatch");tensor(linear.Info(),width);
    require(width!=Qwen35KVTargetWidth||linear.Info().data==target.Info().data,"KV delta complete linear source must be its target");
    require(spans.IsList()&&spans.Size()==8,"KV delta span count mismatch");Qwen35KVDeltaSpans s{};
    for(int b=0;b<4;++b){
        const double start=spans.Get(uint64_t(2*b)).ToDouble(),count=spans.Get(uint64_t(2*b+1)).ToDouble();
        require(std::isfinite(start)&&std::isfinite(count)&&start==std::floor(start)&&count==std::floor(count)&&start>=0&&start<=4096&&count>=0&&count<=4&&count<=4096-start,"KV delta spans out of range");
        s.start[b]=int(start);s.count[b]=int(count);
    }
    std::vector<std::pair<X::Tensor,X3TensorAccess>> tensors{{target,X3_TENSOR_READ},{canonical,X3_TENSOR_WRITE},{linear_,X3_TENSOR_READ},{full_,X3_TENSOR_READ}};
    if(linear.Info().data!=target.Info().data)tensors.emplace_back(linear,X3_TENSOR_READ);
    auto use=TensorHelper::AcquireGPU(tensors);
    const int status=RunQwen35KVDeltaCommit((const float*)target.Info().data,(const float*)linear.Info().data,width,(float*)canonical.Info().data,
        (const Qwen35KVDeltaLinearLayout*)linear_.Info().data,(const Qwen35KVDeltaFullLayout*)full_.Info().data,s,CurrentExecutionStream());
    require(status==0,"KV delta final write failed; canonical completion is unknown");use.Finish();
}
std::string CUDAKVDeltaCommit::StatsJson() const{
    std::lock_guard<std::mutex> lock(mutex_);
    return nlohmann::json{{"ready",ready_},{"canonical_state_width",Qwen35KVCanonicalWidth},{"target_width",Qwen35KVTargetWidth},{"linear_replay_width",Qwen35KVReplayWidth},
        {"owned_layout_bytes",ready_?linear_.Info().byte_size+full_.Info().byte_size:0},{"scope","Metadata only; borrowed canonical/target/replay storage excluded. No timing or peak-memory claim."}}.dump();
}
}
#endif
