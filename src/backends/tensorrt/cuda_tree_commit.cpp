// SPDX-License-Identifier: Apache-2.0
// PREPARED ONLY: not installed, compiled, tested or exposed by GarnetAPI yet.
#if !defined(GARNET_CPU_ONLY) && !defined(GARNET_METAL)
#include "cuda_tree_commit.h"
#include "safetensors_index.h"
#include "tensor_helper.h"
#include <nlohmann/json.hpp>
#include <cmath>
#include <cstring>
#include <fstream>
#include <vector>
#include <climits>
namespace Garnet {namespace {
void require(bool good,const char* message){if(!good)throw X::Error(message);}
int64_t integer(const nlohmann::json& j){require(j.is_number_integer(),"Tree commit metadata requires bounded integers");if(j.is_number_unsigned())require(j.get<uint64_t>()<=INT32_MAX,"Tree integer overflow");int64_t v=j.get<int64_t>();require(v>=0&&v<=INT32_MAX,"Tree integer out of range");return v;}
std::vector<float> weight(const SafeTensorsIndex& index,const std::string& name,const std::vector<long long>& shape){
    const auto* e=index.Find(name);require(e&&e->shape==shape,"Packed prefix weight shape mismatch");
    size_t n=1;for(auto d:shape)n*=size_t(d);
    require(n<=65536&&(e->dataType=="F32"||e->dataType=="BF16"),"Packed prefix weight dtype/size unsupported");
    size_t item=e->dataType=="F32"?4:2;require(e->dataSize==n*item,"Packed prefix weight byte size mismatch");
    std::vector<unsigned char> bytes(n*item);std::ifstream file(e->filePath,std::ios::binary);
    require(bool(file),"Packed prefix weight file open failed");file.seekg(e->dataOffset);file.read((char*)bytes.data(),bytes.size());
    require(bool(file),"Packed prefix weight read failed");std::vector<float> result(n);
    for(size_t i=0;i<n;++i){
        if(item==4)std::memcpy(&result[i],bytes.data()+4*i,4);
        else{uint16_t small;std::memcpy(&small,bytes.data()+2*i,2);uint32_t bits=uint32_t(small)<<16;std::memcpy(&result[i],&bits,4);}
        require(std::isfinite(result[i]),"Packed prefix weights must be finite");
    }return result;
}
void tensor(const X3TensorInfo& v,int b,int64_t width,bool writable){
    require(!v.symbolic&&(!writable||!v.readonly)&&v.device_type==TensorHelper::CudaDevice&&v.dtype==X3_TENSOR_FLOAT32&&v.data&&v.rank==2&&v.shape[0]==b&&v.shape[1]==width&&v.byte_size>=uint64_t(b)*width*4,"Tree commit requires matching contiguous FP32 CUDA matrices");
    require(!v.strides||(v.strides[0]==width*4&&v.strides[1]==4),"Tree commit matrices must be contiguous");
}
}
void CUDATreeCommit::CloseUnlocked(){ready_=false;linear_=full_=parameters_=workspace_=status_=selection_=X::Tensor();config_={};}
CUDATreeCommit::~CUDATreeCommit(){Close();}
void CUDATreeCommit::Close(){std::lock_guard<std::mutex> lock(mutex_);CloseUnlocked();}
bool CUDATreeCommit::Build(const std::string& checkpoint,const std::string& specification){
    std::lock_guard<std::mutex> lock(mutex_);require(!ready_,"Close tree commit helper before rebuilding");
    auto j=nlohmann::json::parse(specification);const auto& m=j.at("config");Qwen35TreeCommitConfig c{};
    c.batch=int(integer(j.at("batch")));c.nodes=int(integer(j.at("nodes")));c.capacity=int(integer(j.at("capacity")));
    c.key_heads=int(integer(m.at("linear_num_key_heads")));c.value_heads=int(integer(m.at("linear_num_value_heads")));c.kernel=int(integer(m.at("linear_conv_kernel_dim")));
    c.kv_heads=int(integer(m.at("num_key_value_heads")));c.head_dim=int(integer(m.at("head_dim")));
    c.state_width=integer(j.at("state_width"));c.packed_width=integer(j.at("packed_width"));c.hidden_width=integer(j.at("hidden_width"));
    c.status_offset=integer(j.at("status_offset"));c.metadata_offset=integer(j.at("metadata_offset"));
    require(m.at("linear_key_head_dim")==128&&m.at("linear_value_head_dim")==128,"Tree commit requires128-wide recurrent state");
    int hidden=int(integer(m.at("hidden_size")));
    require(hidden>=1&&hidden<=16384&&c.nodes>=1&&c.nodes<=64&&c.hidden_width==int64_t(c.nodes)*hidden,"Tree commit hidden geometry invalid");
    require(c.batch==1||c.batch==2||c.batch==4||c.batch==8,"Tree commit batch invalid");
    require(c.capacity>=32&&c.capacity<=8192&&c.key_heads>=1&&c.key_heads<=64&&c.value_heads>=1&&c.value_heads<=64&&c.value_heads%c.key_heads==0&&c.kernel>=1&&c.kernel<=4&&c.kv_heads>=1&&c.kv_heads<=32&&c.head_dim>=32&&c.head_dim<=256&&c.head_dim%32==0,"Tree commit dimensions unsupported");
    auto& layout=j.at("layout");auto& kinds=m.at("layer_types");
    require(layout.is_array()&&kinds.is_array()&&layout.size()==kinds.size()&&!layout.empty()&&layout.size()<=128,"Tree commit layer list invalid");
    SafeTensorsIndex index;std::string error;require(index.Open(checkpoint,error),"Tree commit checkpoint open failed");
    std::vector<Qwen35TreeCommitLayer> linear,full;std::vector<float> parameters;
    int ch=(2*c.key_heads+c.value_heads)*128;int64_t state=0,payload=c.hidden_width;
    for(size_t i=0;i<layout.size();++i){const auto& l=layout[i];std::string kind=kinds[i];
        require(l.at("index")==i&&l.at("kind")==kind&&l.at("offset")==state&&l.at("tree_payload_offset")==payload,"Tree commit layer order/offset invalid");
        int64_t w1=0,w2=0,pw=0;std::vector<int64_t> shape1,shape2;
        if(kind=="linear_attention"){
            w1=int64_t(ch)*c.kernel;w2=int64_t(c.value_heads)*128*128;pw=int64_t(c.nodes)*(ch+2*c.value_heads);
            shape1={c.batch,ch,c.kernel};shape2={c.batch,c.value_heads,128,128};linear.push_back({state,payload,int64_t(i)});
            auto p="model.language_model.layers."+std::to_string(i)+".linear_attn.";
            for(auto item:{std::make_pair("conv1d.weight",std::vector<long long>{ch,1,c.kernel}),std::make_pair("dt_bias",std::vector<long long>{c.value_heads}),std::make_pair("A_log",std::vector<long long>{c.value_heads})}){
                auto values=weight(index,p+item.first,item.second);parameters.insert(parameters.end(),values.begin(),values.end());}
        }else{
            require(kind=="full_attention","Tree commit layer kind unsupported");w1=w2=int64_t(c.kv_heads)*c.capacity*c.head_dim;pw=int64_t(2)*c.nodes*c.kv_heads*c.head_dim;
            shape1=shape2={c.batch,c.kv_heads,c.capacity,c.head_dim};full.push_back({state,payload,int64_t(i)});
        }
        require(l.at("state1_width")==w1&&l.at("state2_width")==w2&&l.at("tree_payload_width")==pw&&l.at("state1_shape")==shape1&&l.at("state2_shape")==shape2,"Tree commit state/payload shapes invalid");
        state+=w1+w2;payload+=pw;
    }
    c.linear_layers=int(linear.size());c.full_layers=int(full.size());
    require(state==c.state_width&&payload==c.status_offset&&c.metadata_offset==payload+layout.size()&&ValidQwen35TreeCommitConfig(c),"Tree commit complete packet layout invalid");
    auto a=TensorHelper::CreateGPU(Host(),X3_TENSOR_INT64,{int64_t(linear.size()),3},linear.data());
    X::Tensor b;if(!full.empty())b=TensorHelper::CreateGPU(Host(),X3_TENSOR_INT64,{int64_t(full.size()),3},full.data());
    auto p=TensorHelper::CreateGPU(Host(),X3_TENSOR_FLOAT32,{int64_t(parameters.size())},parameters.data());
    auto w=TensorHelper::CreateGPU(Host(),X3_TENSOR_INT32,{c.batch,836});
    auto s=TensorHelper::CreateGPU(Host(),X3_TENSOR_INT32,{1});
    auto selected=TensorHelper::CreateGPU(Host(),X3_TENSOR_INT32,{16});
    linear_=a;full_=b;parameters_=p;workspace_=w;status_=s;selection_=selected;config_=c;ready_=true;return true;
}
int CUDATreeCommit::Commit(X::Value packed,X::Value canonical,X::Value parents,X::Value active,X::Value prefix,X::Value selection){
    std::lock_guard<std::mutex> lock(mutex_);require(ready_,"Tree commit helper is closed");
    X::Tensor packet(packed),state(canonical),par(parents),mask(active),start(prefix);auto c=config_;
    tensor(packet.Info(),c.batch,c.packed_width,false);tensor(state.Info(),c.batch,c.state_width,true);tensor(par.Info(),c.batch,c.nodes,false);tensor(mask.Info(),c.batch,c.nodes,false);tensor(start.Info(),c.batch,1,false);
    require(selection.IsList()&&selection.Size()==uint64_t(2*c.batch),"Tree selected prefix/last-node count mismatch");Qwen35TreeCommitSelection selected{};
    for(int b=0;b<c.batch;++b){double p=selection.Get(uint64_t(2*b)).ToDouble(),n=selection.Get(uint64_t(2*b+1)).ToDouble();
        require(std::isfinite(p)&&p==std::floor(p)&&p>=0&&p<=c.capacity&&std::isfinite(n)&&n==std::floor(n)&&n>=-1&&n<c.nodes,"Tree selected prefix/last node invalid");selected.prefix[b]=int(p);selected.last[b]=int(n);}
    std::vector<std::pair<X::Tensor,X3TensorAccess>> tensors{{packet,X3_TENSOR_READ},{state,X3_TENSOR_WRITE},{par,X3_TENSOR_READ},{mask,X3_TENSOR_READ},{start,X3_TENSOR_READ},{linear_,X3_TENSOR_READ},{parameters_,X3_TENSOR_READ},{workspace_,X3_TENSOR_WRITE},{status_,X3_TENSOR_WRITE}};
    tensors.emplace_back(selection_,X3_TENSOR_WRITE);
    if(full_.IsValid())tensors.emplace_back(full_,X3_TENSOR_READ);auto use=TensorHelper::AcquireGPU(tensors);
    require(cudaMemcpyAsync(selection_.Info().data,&selected,sizeof(selected),cudaMemcpyHostToDevice,CurrentExecutionStream())==cudaSuccess,"Tree selection upload failed; canonical completion unknown");
    int rc=RunQwen35TreeCommit(c,(const Qwen35TreeCommitSelection*)selection_.Info().data,(const float*)packet.Info().data,(float*)state.Info().data,(const float*)par.Info().data,(const float*)mask.Info().data,(const float*)start.Info().data,
        (const Qwen35TreeCommitLayer*)linear_.Info().data,full_.IsValid()?(const Qwen35TreeCommitLayer*)full_.Info().data:nullptr,(const float*)parameters_.Info().data,workspace_.Info().data,(int*)status_.Info().data,CurrentExecutionStream());
    require(rc==0,"Tree commit launch failed; canonical completion unknown");int result=-1;
    require(cudaMemcpyAsync(&result,status_.Info().data,4,cudaMemcpyDeviceToHost,CurrentExecutionStream())==cudaSuccess,"Tree commit status read failed; canonical completion unknown");
    require(cudaStreamSynchronize(CurrentExecutionStream())==cudaSuccess,"Tree commit completion failed; canonical completion unknown");
    require(result==0||result==1,"Tree commit status corrupted; canonical completion unknown");use.Finish();return result;
}
std::string CUDATreeCommit::StatsJson() const{
    std::lock_guard<std::mutex> lock(mutex_);auto bytes=[](const X::Tensor& t)->uint64_t{return t.IsValid()?t.Info().byte_size:0;};
    return nlohmann::json{{"ready",ready_},{"parameter_bytes",bytes(parameters_)},{"layout_bytes",bytes(linear_)+bytes(full_)},{"metadata_workspace_bytes",bytes(workspace_)},{"status_bytes",bytes(status_)},
        {"selection_bytes",bytes(selection_)},{"owned_bytes",bytes(parameters_)+bytes(linear_)+bytes(full_)+bytes(workspace_)+bytes(status_)+bytes(selection_)},{"method","validated selected tree in-place recurrent/conv/KV commit"},{"scope","Helper-owned storage only; borrowed packet/canonical/controls excluded. Not process VRAM or peak."}}.dump();
}
}
#endif
