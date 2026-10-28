// SPDX-License-Identifier: Apache-2.0
#if !defined(GARNET_CPU_ONLY) && !defined(GARNET_METAL)
#include "cuda_mtp_draft.h"
#include "cuda_compiled_graph.h"
#include "cuda_tensor_io.h"
#include "tensor_helper.h"
#include "../../cuda/qwen35_indexed_head.cuh"
#include <nlohmann/json.hpp>
#include <climits>
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace Garnet {
cudaError_t CopyQwen35LastChunk(const float*,const float*,float*,int,int,int,cudaStream_t);
namespace {
void Check(cudaError_t status){if(status!=cudaSuccess)throw X::Error(cudaGetErrorString(status));}
void Require(bool okay,const char* message){if(!okay)throw X::Error(message);}
std::vector<int64_t> Dimensions(const X3TensorInfo& x){return {x.shape,x.shape+x.rank};}
size_t Bytes(const X3TensorInfo& x,X3TensorDType type,int device,bool writable){
    Require(!x.symbolic&&x.data&&x.device_type==TensorHelper::CudaDevice&&x.device_id==device&&x.dtype==type&&x.rank>0&&x.rank<=4&&(!writable||!x.readonly),"MTP draft tensor type/device/ownership invalid");
    size_t size=TensorHelper::ItemSize(type);
    for(int i=int(x.rank)-1;i>=0;--i){
        Require(x.shape[i]>0&&x.shape[i]<=INT_MAX&&size<=INT64_MAX&&(!x.strides||x.strides[i]==int64_t(size))&&size<=SIZE_MAX/size_t(x.shape[i]),"MTP draft tensor shape/stride invalid");size*=size_t(x.shape[i]);
    }
    Require(size<=x.byte_size,"MTP draft tensor exceeds accessible backing storage");
    cudaPointerAttributes p{};Check(cudaPointerGetAttributes(&p,x.data));Require((p.type==cudaMemoryTypeDevice||p.type==cudaMemoryTypeManaged)&&p.device==device,"MTP draft pointer is not on the selected device");return size;
}
bool Overlap(const X::Tensor& a,const X::Tensor& b){
    auto x=a.Info(),y=b.Info();size_t nx=TensorHelper::ItemSize(x.dtype),ny=TensorHelper::ItemSize(y.dtype);
    for(uint32_t i=0;i<x.rank;++i)nx*=size_t(x.shape[i]);for(uint32_t i=0;i<y.rank;++i)ny*=size_t(y.shape[i]);
    auto px=reinterpret_cast<uintptr_t>(x.data),py=reinterpret_cast<uintptr_t>(y.data);return px>=py?px-py<ny:py-px<nx;
}
void CheckCaptured(cudaGraph_t graph,int device,int depth=0){
    Require(depth<=8,"MTP draft captured graph nesting exceeds8");size_t count=0;Check(cudaGraphGetNodes(graph,nullptr,&count));std::vector<cudaGraphNode_t> nodes(count);Check(cudaGraphGetNodes(graph,nodes.data(),&count));
    auto pointer=[&](const void* p){cudaPointerAttributes x{};Check(cudaPointerGetAttributes(&x,p));Require((x.type==cudaMemoryTypeDevice||x.type==cudaMemoryTypeManaged)&&x.device==device,"MTP draft captured copy must remain on device");};
    for(auto node:nodes){cudaGraphNodeType type;Check(cudaGraphNodeGetType(node,&type));
        if(type==cudaGraphNodeTypeKernel||type==cudaGraphNodeTypeEmpty)continue;
        if(type==cudaGraphNodeTypeGraph){cudaGraph_t child=nullptr;Check(cudaGraphChildGraphNodeGetGraph(node,&child));CheckCaptured(child,device,depth+1);continue;}
        if(type==cudaGraphNodeTypeMemcpy){cudaMemcpy3DParms p{};Check(cudaGraphMemcpyNodeGetParams(node,&p));Require(!p.srcArray&&!p.dstArray,"MTP draft array copies unsupported");pointer(p.srcPtr.ptr);pointer(p.dstPtr.ptr);continue;}
        if(type==cudaGraphNodeTypeMemset){cudaMemsetParams p{};Check(cudaGraphMemsetNodeGetParams(node,&p));pointer(p.dst);continue;}
        throw X::Error("MTP draft child has unsupported host/event/allocation/control node");
    }
}
void SliceCopy(const X::Tensor& packed,int start,const X::Tensor& output,int batch,cudaStream_t stream){
    auto a=packed.Info(),b=output.Info();size_t bytes=TensorHelper::ItemSize(b.dtype);for(uint32_t i=0;i<b.rank;++i)bytes*=size_t(b.shape[i]);size_t width=bytes/size_t(batch);
    Require(start>=0&&uint64_t(start)+width/4<=uint64_t(a.shape[1]),"MTP draft packed slice exceeds row");
    Check(cudaMemcpy2DAsync(b.data,width,static_cast<float*>(a.data)+start,size_t(a.shape[1])*4,width,size_t(batch),cudaMemcpyDeviceToDevice,stream));
}
int Integer(X::Value value,int low,int high){double x=value.ToDouble();Require(std::isfinite(x)&&x==std::floor(x)&&x>=low&&x<=high,"MTP draft control requires a bounded integer");return int(x);}
}
CUDAMTPDraft::~CUDAMTPDraft(){Close();}
void CUDAMTPDraft::RequireAffinity() const{
    int device=-1;Check(cudaGetDevice(&device));Require(device_>=0&&device==device_&&CurrentExecutionStream()==stream_&&std::this_thread::get_id()==thread_,"MTP draft must use its original device, stream and host thread");
}
void CUDAMTPDraft::RequireReady() const{Require(ready_&&!closed_&&!poisoned_&&child_&&child_->Executable(),"MTP draft is closed, poisoned or not built");RequireAffinity();}
void CUDAMTPDraft::CloseUnlocked() noexcept{
    int previous=-1;cudaGetDevice(&previous);if(device_>=0)cudaSetDevice(device_);
    if(pending_){if(stream_==cudaStreamPerThread&&thread_!=std::this_thread::get_id())cudaDeviceSynchronize();else cudaStreamSynchronize(stream_);}
    pending_=false;ready_=false;closed_=true;
    child_.reset();graphParent_=X::Value();ioParent_=X::Value();tensors_.clear();inputs_.clear();owned_.clear();
    table_=candidates_=hidden_=last_=packed_=controls_=first_=next_=proposals_=state_=scores_=values_=X::Tensor();ownedBytes_=0;
    if(previous>=0&&previous!=device_)cudaSetDevice(previous);
}
void CUDAMTPDraft::Close(){std::lock_guard<std::mutex> lock(mutex_);CloseUnlocked();}
void CUDAMTPDraft::ValidateInputs(X::Value values) const{
    Require(values.runtime()==Host()->runtime&&values.IsList()&&values.Size()==inputs_.size(),"MTP child launch requires its exact seven fixed bindings");
    for(uint64_t i=0;i<values.Size();++i){Require(X::Tensor::IsTensor(values.Get(i)),"MTP child binding is not a tensor");X::Tensor t(values.Get(i));auto a=t.Info(),b=inputs_[i].Info();Bytes(a,X3_TENSOR_FLOAT32,device_,true);Require(a.data==b.data&&Dimensions(a)==Dimensions(b),"MTP child input binding changed after reservation");}
}
bool CUDAMTPDraft::Build(X::Value graphValue,X::Value ioValue,X::Value bindings,X::Value views,X::Value candidates,const std::string& description){
    std::lock_guard<std::mutex> lock(mutex_);Require(!closed_&&!ready_&&!child_,"Build a fresh MTP draft owner once");
    try{
        for(auto value:{graphValue,ioValue,bindings,views,candidates})Require(value.runtime()==Host()->runtime,"MTP draft values must belong to this runtime");
        Require(bindings.IsList()&&bindings.Size()==7&&views.IsList()&&views.Size()==2,"MTP draft requires seven inputs and hidden/last views");
        auto* parent=static_cast<CUDACompiledGraph*>(Host()->instance_get_native_data(graphValue.raw(),X::detail::native_type_name<CUDACompiledGraph>()));
        auto* io=static_cast<CUDATensorIO*>(Host()->instance_get_native_data(ioValue.raw(),X::detail::native_type_name<CUDATensorIO>()));
        Require(parent&&io,"MTP draft needs native graph and tensor I/O objects");
        Check(cudaGetDevice(&device_));stream_=CurrentExecutionStream();thread_=std::this_thread::get_id();cudaDeviceProp properties{};Check(cudaGetDeviceProperties(&properties,device_));
        Require(properties.major==8&&properties.minor==9,"Experimental MTP draft requires SM89");cudaStreamCaptureStatus capturing;Check(cudaStreamIsCapturing(stream_,&capturing));Require(capturing==cudaStreamCaptureStatusNone,"Build MTP draft outside graph capture");
        for(uint64_t i=0;i<7;++i){Require(X::Tensor::IsTensor(bindings.Get(i)),"MTP draft binding is not a tensor");inputs_.emplace_back(bindings.Get(i));Bytes(inputs_.back().Info(),X3_TENSOR_FLOAT32,device_,true);}
        auto first=inputs_[0].Info();Require(first.rank==3&&first.shape[1]==4&&(first.shape[0]==1||first.shape[0]==2||first.shape[0]==4||first.shape[0]==8)&&(first.shape[2]==2048||first.shape[2]==2560),"MTP draft requires Qwen3.5 2B/4B batch1/2/4/8 and T4");
        config_.batch=int(first.shape[0]);config_.hidden=int(first.shape[2]);const int b=config_.batch,h=config_.hidden;
        auto shape=[&](const X::Tensor& x,std::vector<int64_t> expected){Require(Dimensions(x.Info())==expected,"MTP draft tensor geometry differs from fixed profile");};
        shape(inputs_[1],{b,4,h});auto key=inputs_[2].Info();Require(key.rank==4&&key.shape[0]==b&&key.shape[1]>=1&&key.shape[1]<=32&&key.shape[2]>=32&&key.shape[2]<=8192&&key.shape[3]>=1&&key.shape[3]<=256,"MTP draft KV geometry invalid");
        shape(inputs_[3],Dimensions(key));config_.capacity=int(key.shape[2]);cacheWidth_=int(key.shape[1]*key.shape[2]*key.shape[3]);shape(inputs_[4],{b,4,3});shape(inputs_[5],{b,4});shape(inputs_[6],{b,4});
        Require(X::Tensor::IsTensor(views.Get(uint64_t(0)))&&X::Tensor::IsTensor(views.Get(uint64_t(1)))&&X::Tensor::IsTensor(candidates),"MTP draft views/candidates must be tensors");hidden_=X::Tensor(views.Get(uint64_t(0)));last_=X::Tensor(views.Get(uint64_t(1)));candidates_=X::Tensor(candidates);
        Bytes(hidden_.Info(),X3_TENSOR_FLOAT32,device_,true);Bytes(last_.Info(),X3_TENSOR_FLOAT32,device_,true);Bytes(candidates_.Info(),X3_TENSOR_INT32,device_,false);shape(hidden_,{b,4,h});shape(last_,{b,h});shape(candidates_,{b,8192});
        {std::lock_guard<std::mutex> ioLock(io->mutex_);Require(io->table_.IsValid()&&io->width_==h&&io->vocab_>=8192&&io->vocab_<=1<<24,"MTP draft requires the existing BF16 tied table");table_=io->table_;config_.vocab=io->vocab_;}
        Bytes(table_.Info(),X3_TENSOR_BFLOAT16,device_,false);shape(table_,{config_.vocab,h});
        auto descriptionJSON=nlohmann::json::parse(description);Require(descriptionJSON.is_object()&&descriptionJSON.size()==2&&descriptionJSON.at("protocol")=="qwen35-mtp-draft-v1"&&descriptionJSON.at("stops").is_array(),"MTP draft description protocol invalid");auto stops=descriptionJSON.at("stops");Require(stops.size()>=1&&stops.size()<=4,"MTP draft requires1..4 stop IDs");config_.stopCount=int(stops.size());
        for(int i=0;i<config_.stopCount;++i){Require(stops[i].is_number_integer(),"MTP stop ID must be integer");int64_t id=stops[i].get<int64_t>();Require(id>=0&&id<config_.vocab,"MTP stop ID outside vocabulary");config_.stops[i]=int(id);}
        std::vector<X::Tensor> writable=inputs_;writable.push_back(hidden_);writable.push_back(last_);
        for(size_t i=0;i<writable.size();++i){Require(!Overlap(writable[i],table_)&&!Overlap(writable[i],candidates_),"MTP draft writable tensor aliases readonly data");for(size_t j=0;j<i;++j)Require(!Overlap(writable[i],writable[j]),"MTP draft writable buffers overlap");}
        child_=parent->ReserveForBundle(bindings);graphParent_=graphValue;ioParent_=ioValue;Require(child_->Device()==device_&&child_->Stream()==stream_&&child_->Executable(),"MTP reserved context affinity/executable invalid");CheckCaptured(child_->Graph(),device_);packed_=X::Tensor(child_->Output());
        Bytes(packed_.Info(),X3_TENSOR_FLOAT32,device_,true);packedWidth_=4*h+2*cacheWidth_;shape(packed_,{b,packedWidth_});for(const auto& x:writable)Require(!Overlap(packed_,x),"MTP graph output aliases its request buffers");
        auto allocate=[&](X3TensorDType type,std::vector<int64_t> dims){auto tensor=TensorHelper::CreateGPU(Host(),type,dims);ownedBytes_+=Bytes(tensor.Info(),type,device_,true);owned_.push_back(tensor);return tensor;};
        controls_=allocate(X3_TENSOR_INT32,{b,4});first_=allocate(X3_TENSOR_INT64,{b});next_=allocate(X3_TENSOR_INT64,{b});proposals_=allocate(X3_TENSOR_INT64,{b,3});state_=allocate(X3_TENSOR_INT32,{int64_t(sizeof(MTPDraftDeviceState)/4)});scores_=allocate(X3_TENSOR_BFLOAT16,{b,8192});values_=allocate(X3_TENSOR_FLOAT32,{b});
        tensors_=child_->Tensors();tensors_.emplace_back(table_,X3_TENSOR_READ);tensors_.emplace_back(candidates_,X3_TENSOR_READ);
        for(auto tensor:writable){tensors_.emplace_back(tensor,X3_TENSOR_READ);tensors_.emplace_back(tensor,X3_TENSOR_WRITE);}
        for(auto tensor:owned_){tensors_.emplace_back(tensor,X3_TENSOR_READ);tensors_.emplace_back(tensor,X3_TENSOR_WRITE);}
        ready_=true;return true;
    }catch(...){CloseUnlocked();throw;}
}
X::Value CUDAMTPDraft::RunChild(X::Value bindings){
    std::lock_guard<std::mutex> lock(mutex_);RequireReady();ValidateInputs(bindings);
    try{auto use=TensorHelper::AcquireGPU(child_->Tensors(),stream_);pending_=true;Check(cudaGraphLaunch(child_->Executable(),stream_));use.Finish();++childSubmissions_;return child_->Output();}catch(...){poisoned_=true;throw;}
}
std::string CUDAMTPDraft::Run(X::Value firstDraft,X::Value controlRows){
    std::lock_guard<std::mutex> lock(mutex_);RequireReady();Require(firstDraft.runtime()==Host()->runtime&&controlRows.runtime()==Host()->runtime&&firstDraft.IsList()&&controlRows.IsList()&&firstDraft.Size()==uint64_t(config_.batch)&&controlRows.Size()==uint64_t(config_.batch),"MTP draft controls require one row per batch member");
    std::vector<long long> ids(config_.batch);std::vector<int> controls(size_t(config_.batch)*4);
    bool any=false;
    for(int b=0;b<config_.batch;++b){ids[b]=Integer(firstDraft.Get(uint64_t(b)),0,config_.vocab-1);auto row=controlRows.Get(uint64_t(b));Require(row.IsList()&&row.Size()==4,"MTP draft control row requires active,remaining,position,slot");controls[b*4]=Integer(row.Get(uint64_t(0)),0,1);controls[b*4+1]=Integer(row.Get(uint64_t(1)),0,128);controls[b*4+2]=Integer(row.Get(uint64_t(2)),0,1048576-128);controls[b*4+3]=Integer(row.Get(uint64_t(3)),0,config_.capacity);if(controls[b*4]){Require(controls[b*4+1]>=2,"Active MTP draft row needs at least2 remaining outputs");any=true;}}
    Require(any,"MTP draft requires an active row");std::vector<long long> proposals(size_t(config_.batch)*3);
    try{
        const int batch=config_.batch,h=config_.hidden;int count=0;
        std::vector<int> hostSlots(batch),advance1(batch),advance2(batch);std::vector<char> stopped(batch);std::vector<long long> intermediate(batch);
        auto isStop=[&](long long id){for(int i=0;i<config_.stopCount;++i)if(id==config_.stops[i])return true;return false;};
        for(int row=0;row<batch;++row){hostSlots[row]=controls[row*4+3];if(controls[row*4])count=std::max(count,std::min(3,controls[row*4+1]-1));}
        bool firstAdvance=false;
        for(int row=0;row<batch;++row){stopped[row]=controls[row*4]&&isStop(ids[row]);advance1[row]=count>1&&controls[row*4]&&!stopped[row]&&controls[row*4+1]>2;firstAdvance|=advance1[row]!=0;}
        auto validateCapacity=[&](const std::vector<int>& advance){for(int row=0;row<batch;++row)Require(!advance[row]||hostSlots[row]<config_.capacity,"MTP draft cache capacity exhausted; reset required");};
        validateCapacity(advance1);
        auto use=TensorHelper::AcquireGPU(tensors_,stream_);pending_=true;
        auto* deviceState=static_cast<MTPDraftDeviceState*>(state_.Info().data);auto* proposalIds=static_cast<long long*>(proposals_.Info().data);auto* nextIds=static_cast<long long*>(next_.Info().data);
        Check(cudaMemcpyAsync(first_.Info().data,ids.data(),ids.size()*8,cudaMemcpyHostToDevice,stream_));Check(cudaMemcpyAsync(controls_.Info().data,controls.data(),controls.size()*4,cudaMemcpyHostToDevice,stream_));
        Check(cudaMemcpyAsync(nextIds,first_.Info().data,ids.size()*8,cudaMemcpyDeviceToDevice,stream_));Check(InitMTPDraftDevice(config_,static_cast<const int*>(controls_.Info().data),deviceState,proposalIds,stream_));
        int physicalCalls=0;
        auto prepare=[&](int stage){
            Check(PrepareMTPDraftDevice(config_,stage,nextIds,proposalIds,deviceState,0,stream_));
            Check(InputsMTPDraftDevice(config_,stage,table_.Info().data,proposalIds,deviceState,static_cast<const float*>(last_.Info().data),static_cast<float*>(inputs_[0].Info().data),static_cast<float*>(inputs_[1].Info().data),static_cast<float*>(inputs_[4].Info().data),static_cast<float*>(inputs_[5].Info().data),static_cast<float*>(inputs_[6].Info().data),stream_));
        };
        auto learnedStep=[&](const std::vector<int>& advance){
            Check(cudaGraphLaunch(child_->Executable(),stream_));
            SliceCopy(packed_,0,hidden_,batch,stream_);SliceCopy(packed_,4*h,inputs_[2],batch,stream_);SliceCopy(packed_,4*h+cacheWidth_,inputs_[3],batch,stream_);
            Check(CopyQwen35LastChunk(static_cast<const float*>(hidden_.Info().data),static_cast<const float*>(inputs_[6].Info().data),static_cast<float*>(last_.Info().data),batch,4,h,stream_));
            Require(RunQwen35IndexedHead({batch,h,config_.vocab,8192},static_cast<const float*>(last_.Info().data),table_.Info().data,static_cast<const int*>(candidates_.Info().data),scores_.Info().data,nextIds,static_cast<float*>(values_.Info().data),stream_)==0,"MTP native staged indexed head failed");
            for(int row=0;row<batch;++row)hostSlots[row]+=advance[row];++physicalCalls;
        };
        prepare(1);if(firstAdvance)learnedStep(advance1);
        bool secondAdvance=false;
        if(count>2&&firstAdvance){
            Check(cudaMemcpyAsync(intermediate.data(),nextIds,intermediate.size()*8,cudaMemcpyDeviceToHost,stream_));Check(cudaStreamSynchronize(stream_));++decisionTransfers_;
            for(int row=0;row<batch;++row){Require(intermediate[row]>=0&&intermediate[row]<config_.vocab,"MTP native staged head returned invalid token; reset required");if(controls[row*4]&&!stopped[row]&&2<controls[row*4+1]&&isStop(intermediate[row]))stopped[row]=true;
                advance2[row]=controls[row*4]&&!stopped[row]&&controls[row*4+1]>3;secondAdvance|=advance2[row]!=0;}
        }
        validateCapacity(advance2);prepare(2);if(secondAdvance)learnedStep(advance2);
        Check(FinishMTPDraftDevice(config_,nextIds,proposalIds,deviceState,stream_));
        Check(cudaMemcpyAsync(&lastState_,state_.Info().data,sizeof(lastState_),cudaMemcpyDeviceToHost,stream_));Check(cudaMemcpyAsync(proposals.data(),proposals_.Info().data,proposals.size()*8,cudaMemcpyDeviceToHost,stream_));Check(cudaStreamSynchronize(stream_));use.Finish();++submissions_;
        Require(lastState_.error==0&&lastState_.stage==3&&lastState_.count==count&&lastState_.modelCalls==physicalCalls&&physicalCalls>=0&&physicalCalls<=2,"MTP native staged control or model prediction failed; reset required");
        for(int row=0;row<batch;++row)Require(lastState_.slots[row]==hostSlots[row],"MTP host/device staged cache slots differ; reset required");
        nlohmann::json packet=nlohmann::json::array();for(int t=0;t<lastState_.count;++t){nlohmann::json row=nlohmann::json::array();for(int b=0;b<config_.batch;++b)row.push_back(proposals[size_t(b)*3+t]);packet.push_back(row);}
        return nlohmann::json{{"draft_proposals",packet},{"slots",std::vector<int>(lastState_.slots,lastState_.slots+config_.batch)},{"model_calls",lastState_.modelCalls},{"proposed_tokens",lastState_.proposed}}.dump();
    }catch(...){poisoned_=true;throw;}
}
std::string CUDAMTPDraft::StatsJson() const{
    std::lock_guard<std::mutex> lock(mutex_);
    return nlohmann::json{{"ready",ready_},{"closed",closed_},{"poisoned",poisoned_},{"device",device_},{"batch",config_.batch},{"hidden",config_.hidden},{"capacity",config_.capacity},{"reserved_model_contexts",child_?1:0},{"new_tensor_bytes",ownedBytes_},{"new_large_weight_bytes",0},{"borrowed_BF16_table_bytes_excluded",table_.IsValid()?uint64_t(config_.vocab)*config_.hidden*2:0},{"conditional_IF_nodes",0},{"new_CUDA_graph_executables",0},{"execution_strategy","native_host_dispatch_original_MTP_capture"},{"intermediate_host_decision_transfers",decisionTransfers_},{"intermediate_host_decision_transfers_per_window_max",1},{"host_synchronizations_per_window_max",2},{"draft_submissions",submissions_},{"child_submissions",childSubmissions_},{"last_model_calls",lastState_.modelCalls},{"CUDA_graph_driver_bytes_measured",false},{"scope","Opt-in host-dispatched retained-context learned D3/T4 draft/head/embedding execution; original warmed MTP executable only, no conditional graph composition. Target acceptance/verification unchanged. Existing model weights/table and retained input/cache/output tensors excluded from new allocations. Driver graph memory untracked; no primary speed or memory qualification."}}.dump();
}
}
#endif
