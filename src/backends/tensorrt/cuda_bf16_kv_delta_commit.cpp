// SPDX-License-Identifier: Apache-2.0
#if !defined(GARNET_CPU_ONLY) && !defined(GARNET_METAL)
#include "cuda_bf16_kv_delta_commit.h"
#include "../../cuda/qwen38_bf16_kv_commit.cuh"
#include "tensor_helper.h"
#include <limits>
#include <vector>
namespace Garnet { namespace {
void require(bool v,const char* s){if(!v)throw X::Error(s);}
void matrix(const X3TensorInfo& v,int batch,int64_t width,bool writable){
    require(!v.symbolic&&(!writable||!v.readonly)&&v.device_type==TensorHelper::CudaDevice&&v.device_id==0&&
        v.dtype==X3_TENSOR_FLOAT32&&v.data&&v.rank==2&&v.shape&&v.shape[0]==batch&&v.shape[1]==width&&
        v.byte_size>=uint64_t(batch)*uint64_t(width)*4&&reinterpret_cast<uintptr_t>(v.data)%alignof(float)==0,
        "BF16 KV commit requires matching CUDA FP32 carrier matrices and writable destination");
    require(!v.strides||(v.strides[0]==width*4&&v.strides[1]==4),"BF16 KV carrier matrices must be contiguous");
}
bool overlap(const void* a,size_t n,const void* b,size_t m){
    const auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b),maximum=(std::numeric_limits<uintptr_t>::max)();
    return !a||!b||x>maximum-n||y>maximum-m||(x<y+m&&y<x+n);
}
uint64_t spanInteger(const X::Value& v){
    require(v.IsInt64()||v.IsUInt64(),"BF16 KV spans require integer values, excluding booleans/floats");
    const uint64_t value=v.ToUInt64();require(value<=4096,"BF16 KV span integer out of range");return value;
}
}
bool CUDABF16KVDeltaCommit::Build(const std::string& specification){
    std::lock_guard<std::mutex> lock(mutex_);require(!ready_,"Close BF16 KV helper before rebuilding");
    BF16KVLayoutSpec candidate;
    try{candidate=BF16KVContract::Parse(nlohmann::json::parse(specification));}catch(const std::exception& e){throw X::Error(e.what());}
    layout_=candidate;poisoned_=false;commits_=0;ready_=true;return true;
}
void CUDABF16KVDeltaCommit::Close(){std::lock_guard<std::mutex> lock(mutex_);ready_=false;poisoned_=false;layout_={};commits_=0;}
void CUDABF16KVDeltaCommit::Commit(X::Value packed,X::Value replayed,X::Value state,X::Value spans){
    std::lock_guard<std::mutex> lock(mutex_);require(ready_&&!poisoned_,"BF16 KV helper is closed or poisoned");
    require(packed.runtime()==Host()->runtime&&replayed.runtime()==Host()->runtime&&state.runtime()==Host()->runtime&&spans.runtime()==Host()->runtime,"BF16 KV values must belong to the current runtime");
    X::Tensor target(packed),linear(replayed),canonical(state);
    matrix(target.Info(),layout_.batch,layout_.target,false);matrix(canonical.Info(),layout_.batch,layout_.canonical,true);
    const auto info=linear.Info();require(info.rank==2&&info.shape,"BF16 KV linear source rank mismatch");const int64_t width=info.shape[1];
    require(width==layout_.target||(layout_.chunk==4&&width==layout_.replay),"BF16 KV linear source width mismatch");
    matrix(info,layout_.batch,width,false);require(width!=layout_.target||info.data==target.Info().data,"BF16 KV complete linear source must be its target");
    const auto dst=canonical.Info();const size_t stateBytes=size_t(layout_.batch)*size_t(layout_.canonical)*4;
    require(!overlap(dst.data,stateBytes,target.Info().data,size_t(layout_.batch)*size_t(layout_.target)*4)&&
        !overlap(dst.data,stateBytes,info.data,size_t(layout_.batch)*size_t(width)*4),"BF16 KV destination must not overlap a source");
    require(spans.IsList()&&spans.Size()==uint64_t(2*layout_.batch),"BF16 KV span count mismatch");Qwen38BF16KVSpans controls{};
    for(int b=0;b<layout_.batch;++b){
        const auto start=spanInteger(spans.Get(uint64_t(2*b))),count=spanInteger(spans.Get(uint64_t(2*b+1)));
        require(count<=uint64_t(layout_.chunk)&&count<=4096-start,"BF16 KV span exceeds chunk/cache");controls.start[b]=int(start);controls.count[b]=int(count);
    }
    std::vector<std::pair<X::Tensor,X3TensorAccess>> tensors{{target,X3_TENSOR_READ},{canonical,X3_TENSOR_WRITE}};
    if(info.data!=target.Info().data)tensors.emplace_back(linear,X3_TENSOR_READ);
    auto use=TensorHelper::AcquireGPU(tensors);
    try{
        const int status=RunQwen38BF16KVCommit((const float*)target.Info().data,(const float*)info.data,width,(float*)dst.data,
            layout_.batch,layout_.chunk,controls,CurrentExecutionStream());
        require(status==0,"BF16 KV final write failed; discard canonical state because completion is unknown");use.Finish();++commits_;
    }catch(...){poisoned_=true;throw;}
}
std::string CUDABF16KVDeltaCommit::StatsJson() const{
    std::lock_guard<std::mutex> lock(mutex_);
    return nlohmann::json{{"ready",ready_},{"poisoned",poisoned_},{"batch",layout_.batch},{"chunk",layout_.chunk},
        {"canonical_state_width",layout_.canonical},{"target_width",layout_.target},{"linear_replay_width",layout_.replay},
        {"commits",commits_},{"owned_layout_bytes",0},{"research_only",true},
        {"scope","Metadata only; borrowed storage excluded. No model quality, timing or peak-memory claim."}}.dump();
}
}
#endif
