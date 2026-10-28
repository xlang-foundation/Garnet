// SPDX-License-Identifier: Apache-2.0
#if !defined(GARNET_CPU_ONLY) && !defined(GARNET_METAL)
#include "cuda_packed_prefix.h"
#include "qwen35_phase_output_contract.h"
#include "safetensors_index.h"
#include "tensor_helper.h"
#include <nlohmann/json.hpp>
#include <cmath>
#include <cstring>
#include <fstream>
#include <vector>
#include <unordered_set>
#include <climits>
namespace Garnet { namespace {
void require(bool okay,const char* message){if(!okay)throw X::Error(message);}
int64_t integer(const nlohmann::json& item){
    require(item.is_number_integer(),"Packed commit metadata must use bounded integers");
    if(item.is_number_unsigned())require(item.get<uint64_t>()<=uint64_t(INT32_MAX),"Packed commit integer overflow");
    int64_t value=item.get<int64_t>();require(value>=0&&value<=INT32_MAX,"Packed commit integer out of range");return value;
}
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
void tensor(const X3TensorInfo& v,int b,int64_t width){
    require(!v.symbolic&&!v.readonly&&v.device_type==TensorHelper::CudaDevice&&v.dtype==X3_TENSOR_FLOAT32&&
        v.data&&v.rank==2&&v.shape[0]==b&&v.shape[1]==width&&v.byte_size>=uint64_t(b)*width*4,
        "Packed prefix requires matching writable FP32 CUDA matrices");
    require(!v.strides||(v.strides[0]==width*4&&v.strides[1]==4),"Packed prefix matrices must be contiguous");
}
bool overlap(const X3TensorInfo& a,const X3TensorInfo& b){auto x=reinterpret_cast<uintptr_t>(a.data),y=reinterpret_cast<uintptr_t>(b.data);return x>=y?x-y<b.byte_size:y-x<a.byte_size;}
}
void CUDAPackedPrefix::CloseUnlocked(){ready_=false;linear_=full_=parameters_=workspace_=X::Tensor();config_={};}
CUDAPackedPrefix::~CUDAPackedPrefix(){Close();}
void CUDAPackedPrefix::Close(){std::lock_guard<std::mutex> lock(mutex_);CloseUnlocked();}
bool CUDAPackedPrefix::BuildPhaseOutput(const std::string& checkpoint,const std::string& specification){
    // Build() holds the object lock. Parse the complete source/destination
    // contract before reading parameters or creating any device storage.
    const auto plan=ParseQwen35PhaseOutputContract(nlohmann::json::parse(specification));
    const auto& c=plan.config;require(ValidQwen35PackedCommitConfig(c),"Phase prefix config outside admitted kernel range");
    std::vector<float> parameters;
    if(!c.copy_only){
        SafeTensorsIndex index;std::string error;require(index.Open(checkpoint,error),"Phase prefix checkpoint open failed");
        const int channels=(2*c.key_heads+c.value_heads)*128;
        for(int layer:plan.parameter_layers){
            auto prefix="model.language_model.layers."+std::to_string(layer)+".linear_attn.";
            for(auto pair:{std::make_pair("conv1d.weight",std::vector<long long>{channels,1,c.kernel}),
                    std::make_pair("dt_bias",std::vector<long long>{c.value_heads}),
                    std::make_pair("A_log",std::vector<long long>{c.value_heads})}){
                auto data=weight(index,prefix+pair.first,pair.second);parameters.insert(parameters.end(),data.begin(),data.end());
            }
        }
    }
    const size_t workspaceBytes=Qwen35PackedCommitWorkspace(c);
    require(c.copy_only||workspaceBytes>0,"Phase replay workspace invalid");
    auto linear=TensorHelper::CreateGPU(Host(),X3_TENSOR_INT64,{int64_t(plan.linear.size()),2},plan.linear.data());
    X::Tensor full,p,w;
    if(!plan.full.empty())full=TensorHelper::CreateGPU(Host(),X3_TENSOR_INT64,{int64_t(plan.full.size())},plan.full.data());
    if(!parameters.empty())p=TensorHelper::CreateGPU(Host(),X3_TENSOR_FLOAT32,{int64_t(parameters.size())},parameters.data());
    if(workspaceBytes)w=TensorHelper::CreateGPU(Host(),X3_TENSOR_FLOAT32,{int64_t(workspaceBytes/4)});
    linear_=linear;full_=full;parameters_=p;workspace_=w;config_=c;ready_=true;return true;
}
bool CUDAPackedPrefix::Build(const std::string& checkpoint,const std::string& specification){
    std::lock_guard<std::mutex> lock(mutex_);require(!ready_,"Close packed prefix before rebuilding");
    auto j=nlohmann::json::parse(specification);
    if(j.contains("phase_output_layout"))return BuildPhaseOutput(checkpoint,specification);
    auto m=j.at("config");Qwen35PackedCommitConfig c{};
    c.batch=int(integer(j.at("batch")));c.tokens=int(integer(j.at("tokens")));c.capacity=int(integer(j.at("capacity")));
    c.key_heads=int(integer(m.at("linear_num_key_heads")));c.value_heads=int(integer(m.at("linear_num_value_heads")));c.kernel=int(integer(m.at("linear_conv_kernel_dim")));
    c.kv_heads=int(integer(m.at("num_key_value_heads")));c.head_dim=int(integer(m.at("head_dim")));
    c.state_width=integer(j.at("state_width"));c.packed_width=integer(j.at("packed_width"));c.hidden_width=integer(j.at("hidden_width"));
    c.compact_kv=j.value("compact_kv_outputs",false);c.copy_only=j.value("copy_only",false);
    require(!c.copy_only||c.compact_kv,"Copy-only commit requires compact KV output");
    c.output_state_width=c.compact_kv?integer(j.at("output_state_width")):0;
    c.status_offset=c.compact_kv?integer(j.at("status_offset")):0;
    require(m.at("linear_key_head_dim")==128&&m.at("linear_value_head_dim")==128,"Packed prefix requires128-wide K/V state");
    int hidden=int(integer(m.at("hidden_size")));require(hidden>=1&&hidden<=16384&&c.tokens>=1&&c.tokens<=(c.copy_only?128:16)&&c.hidden_width==int64_t(hidden)*c.tokens,
        "Packed prefix hidden layout invalid");
    // Validate every interval before creating device storage or launching work.
    const auto& layers=j.at("layout");const auto& kinds=m.at("layer_types");
    require(layers.is_array()&&layers.size()==kinds.size()&&layers.size()<=128,"Packed prefix layer list invalid");
    std::vector<Qwen35PackedLinearLayout> linear;std::vector<int64_t> full;std::vector<float> parameters;
    SafeTensorsIndex index;std::string error;if(!c.copy_only)require(index.Open(checkpoint,error),"Packed prefix checkpoint open failed");
    require(c.state_width>0&&c.state_width<=INT32_MAX&&c.packed_width>0&&c.packed_width<=INT32_MAX,"Packed prefix widths out of range");
    if(c.compact_kv)require(c.output_state_width>0&&c.output_state_width<=c.state_width,"Compact output-state width invalid");
    int64_t cursor=0,outputCursor=0,payload=c.hidden_width+(c.compact_kv?c.output_state_width:c.state_width);
    require(c.key_heads>=1&&c.key_heads<=64&&c.value_heads>=1&&c.value_heads<=64&&c.value_heads%c.key_heads==0&&c.kernel>=1&&c.kernel<=4,
        "Packed prefix linear dimensions invalid");
    int channels=2*c.key_heads*128+c.value_heads*128;
    require(c.batch>=1&&c.batch<=8&&c.capacity>=32&&c.capacity<=8192&&c.kv_heads>=1&&c.kv_heads<=32&&c.head_dim>=1&&c.head_dim<=256,
        "Packed prefix batch/cache dimensions invalid");
    for(size_t i=0;i<layers.size();++i){
        const auto& l=layers[i];std::string kind=kinds[i];require(l.at("index")==i&&l.at("kind")==kind&&l.at("offset")==cursor,"Packed prefix layer order invalid");
        int64_t w1=0,w2=0;
        if(kind=="linear_attention"){
            w1=int64_t(channels)*c.kernel;w2=int64_t(c.value_heads)*128*128;
            int64_t n=int64_t(c.tokens)*(channels+c.value_heads*128+2*c.value_heads);
            if(c.copy_only){
                int64_t retained=integer(l.at("payload_width"));require(retained==0||retained==n,"Compact copy retained payload invalid");
                if(retained)require(l.at("payload_offset")==payload,"Compact copy payload offset invalid");
                linear.push_back({cursor,c.hidden_width+outputCursor});payload+=retained;
            }else{
                require(l.at("payload_offset")==payload&&l.at("payload_width")==n,"Packed prefix payload invalid");
                linear.push_back({cursor,payload});payload+=n;
            }
            auto prefix="model.language_model.layers."+std::to_string(i)+".linear_attn.";
            if(!c.copy_only)for(auto pair: {std::make_pair("conv1d.weight",std::vector<long long>{channels,1,c.kernel}),
                            std::make_pair("dt_bias",std::vector<long long>{c.value_heads}),
                            std::make_pair("A_log",std::vector<long long>{c.value_heads})}){
                auto data=weight(index,prefix+pair.first,pair.second);parameters.insert(parameters.end(),data.begin(),data.end());
            }
        }else{
            require(kind=="full_attention","Packed prefix layer kind unsupported");
            w1=w2=int64_t(c.kv_heads)*c.capacity*c.head_dim;full.push_back(cursor);
            if(c.compact_kv)full.push_back(c.hidden_width+outputCursor);
        }
        require(l.at("state1_width")==w1&&l.at("state2_width")==w2,"Packed prefix state widths invalid");
        int64_t out1=(kind=="full_attention"&&c.compact_kv)?int64_t(c.kv_heads)*c.tokens*c.head_dim:w1;
        int64_t out2=(kind=="full_attention"&&c.compact_kv)?out1:w2;
        if(c.compact_kv)require(l.at("output_offset")==outputCursor&&l.at("output_state1_width")==out1&&l.at("output_state2_width")==out2,"Compact output offsets/widths invalid");
        cursor+=w1+w2;outputCursor+=out1+out2;
    }
    c.linear_layers=int(linear.size());c.full_layers=int(full.size()/(c.compact_kv?2:1));
    require(cursor==c.state_width,"Packed prefix canonical layout mismatch");
    if(c.compact_kv)require(outputCursor==c.output_state_width&&payload==c.status_offset&&payload+c.full_layers==c.packed_width,"Compact complete output/payload/status layout mismatch");
    else require(payload==c.packed_width,"Packed prefix complete layout mismatch");
    require(ValidQwen35PackedCommitConfig(c),"Packed prefix config outside admitted range");
    auto workspaceBytes=Qwen35PackedCommitWorkspace(c);require(c.copy_only||workspaceBytes>0,"Packed prefix workspace invalid");
    // Metadata uses integer storage and the exact POD byte representation.
    auto a=TensorHelper::CreateGPU(Host(),X3_TENSOR_INT64,{int64_t(linear.size()),2},linear.data());
    X::Tensor b;if(!full.empty())b=TensorHelper::CreateGPU(Host(),X3_TENSOR_INT64,{int64_t(full.size())},full.data());
    X::Tensor p,w;if(!parameters.empty())p=TensorHelper::CreateGPU(Host(),X3_TENSOR_FLOAT32,{int64_t(parameters.size())},parameters.data());
    if(workspaceBytes)w=TensorHelper::CreateGPU(Host(),X3_TENSOR_FLOAT32,{int64_t(workspaceBytes/4)});
    linear_=a;full_=b;parameters_=p;workspace_=w;config_=c;ready_=true;return true;
}
void CUDAPackedPrefix::CheckCompactStatus(const X::Tensor& source) const {
    if(!config_.compact_kv||!config_.full_layers)return;
    auto info=source.Info();std::vector<float> flags(size_t(config_.batch)*config_.full_layers);
    require(cudaMemcpy2DAsync(flags.data(),size_t(config_.full_layers)*4,(const float*)info.data+config_.status_offset,size_t(config_.packed_width)*4,size_t(config_.full_layers)*4,config_.batch,cudaMemcpyDeviceToHost,CurrentExecutionStream())==cudaSuccess,"Compact status read failed");
    require(cudaStreamSynchronize(CurrentExecutionStream())==cudaSuccess,"Compact status synchronization failed");
    for(float flag:flags)require(flag==0,"Compact KV control error; canonical state was not committed");
}
void CUDAPackedPrefix::Commit(X::Value packed,X::Value canonical,X::Value spans){
    std::lock_guard<std::mutex> lock(mutex_);require(ready_,"Packed prefix is closed");
    require(!config_.copy_only,"Copy-only helper requires commit_all");
    X::Tensor source(packed),state(canonical);tensor(source.Info(),config_.batch,config_.packed_width);tensor(state.Info(),config_.batch,config_.state_width);
    require(!overlap(source.Info(),state.Info()),"Packed/canonical buffers must not overlap");
    require(spans.IsList()&&spans.Size()==uint64_t(2*config_.batch),"Packed prefix span count mismatch");
    Qwen35PackedCommitSpans s{};
    for(int b=0;b<config_.batch;++b){
        double start=spans.Get(uint64_t(2*b)).ToDouble(),count=spans.Get(uint64_t(2*b+1)).ToDouble();
        require(std::isfinite(start)&&std::isfinite(count)&&start==std::floor(start)&&count==std::floor(count)&&
            start>=0&&start<=config_.capacity&&count>=0&&count<=config_.tokens&&count<=config_.capacity-start,"Packed prefix spans out of range");
        s.start[b]=int(start);s.count[b]=int(count);
    }
    std::vector<std::pair<X::Tensor,X3TensorAccess>> tensors{{source,X3_TENSOR_READ},{state,X3_TENSOR_WRITE},
        {linear_,X3_TENSOR_READ},{parameters_,X3_TENSOR_READ},{workspace_,X3_TENSOR_WRITE}};
    if(full_.IsValid())tensors.emplace_back(full_,X3_TENSOR_READ);
    auto use=TensorHelper::AcquireGPU(tensors);
    CheckCompactStatus(source);
    int status=RunQwen35PackedCommit(config_,s,(const float*)source.Info().data,(float*)state.Info().data,
        (const Qwen35PackedLinearLayout*)linear_.Info().data,full_.IsValid()?(const int64_t*)full_.Info().data:nullptr,
        (const float*)parameters_.Info().data,(float*)workspace_.Info().data,CurrentExecutionStream());
    require(status==0,"Packed prefix kernel launch failed; canonical completion is unknown");use.Finish();
}
void CUDAPackedPrefix::CommitAll(X::Value packed,X::Value canonical,X::Value slots,X::Value active){
    std::lock_guard<std::mutex> lock(mutex_);require(ready_&&config_.compact_kv&&config_.copy_only,"Compact copy helper is closed or replay-only");
    X::Tensor source(packed),state(canonical),slot(slots),flag(active);tensor(source.Info(),config_.batch,config_.packed_width);tensor(state.Info(),config_.batch,config_.state_width);tensor(slot.Info(),config_.batch,config_.tokens);tensor(flag.Info(),config_.batch,config_.tokens);
    for(auto* input:{&source,&slot,&flag})require(!overlap(input->Info(),state.Info()),"Compact canonical storage must not overlap its inputs");
    std::vector<std::pair<X::Tensor,X3TensorAccess>> tensors{{source,X3_TENSOR_READ},{state,X3_TENSOR_WRITE},{slot,X3_TENSOR_READ},{flag,X3_TENSOR_READ},{linear_,X3_TENSOR_READ}};if(full_.IsValid())tensors.emplace_back(full_,X3_TENSOR_READ);
    auto use=TensorHelper::AcquireGPU(tensors);CheckCompactStatus(source);size_t n=size_t(config_.batch)*config_.tokens;std::vector<float> indices(n),flags(n);
    require(cudaMemcpyAsync(indices.data(),slot.Info().data,n*4,cudaMemcpyDeviceToHost,CurrentExecutionStream())==cudaSuccess&&cudaMemcpyAsync(flags.data(),flag.Info().data,n*4,cudaMemcpyDeviceToHost,CurrentExecutionStream())==cudaSuccess,"Compact slot/active read failed");
    require(cudaStreamSynchronize(CurrentExecutionStream())==cudaSuccess,"Compact slot/active synchronization failed");
    for(int b=0;b<config_.batch;++b){std::unordered_set<int> seen;for(int t=0;t<config_.tokens;++t){size_t i=size_t(b)*config_.tokens+t;require(flags[i]==0||flags[i]==1,"Compact active controls must be binary");require(std::isfinite(indices[i])&&indices[i]==std::floor(indices[i])&&indices[i]>=0&&indices[i]<=config_.capacity,"Compact slots invalid");if(flags[i])require(indices[i]<config_.capacity&&seen.insert(int(indices[i])).second,"Compact active slots must be unique and in range");}}
    require(RunQwen35PackedCommitAll(config_,(const float*)source.Info().data,(float*)state.Info().data,(const Qwen35PackedLinearLayout*)linear_.Info().data,full_.IsValid()?(const int64_t*)full_.Info().data:nullptr,(const float*)slot.Info().data,(const float*)flag.Info().data,CurrentExecutionStream())==0,"Compact canonical commit failed; completion unknown");use.Finish();
}
std::string CUDAPackedPrefix::StatsJson() const{
    std::lock_guard<std::mutex> lock(mutex_);auto bytes=[](const X::Tensor& t)->uint64_t{return t.IsValid()?t.Info().byte_size:0;};
    return nlohmann::json{{"ready",ready_},{"workspace_bytes",bytes(workspace_)},{"parameter_bytes",bytes(parameters_)},
        {"layout_bytes",bytes(linear_)+bytes(full_)},{"owned_bytes",bytes(workspace_)+bytes(parameters_)+bytes(linear_)+bytes(full_)},
        {"compact_kv_outputs",bool(config_.compact_kv)},{"copy_only",bool(config_.copy_only)},
        {"method","native state-only in-place prefix commit"},{"scope","Owned helper buffers only; borrowed packed/canonical state excluded. Not process VRAM or peak."}}.dump();
}
}
#endif
