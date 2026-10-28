// SPDX-License-Identifier: Apache-2.0
#if !defined(GARNET_CPU_ONLY) && !defined(GARNET_METAL)
#include "cuda_execution_bundle.h"
#include "cuda_compiled_graph.h"
#include "tensor_helper.h"
#include <nlohmann/json.hpp>
#include <climits>
#include <stdexcept>
namespace Garnet {
cudaError_t CopyQwen35LastChunk(const float*,const float*,float*,int,int,int,cudaStream_t);
namespace {
void Check(cudaError_t status){if(status!=cudaSuccess)throw X::Error(cudaGetErrorString(status));}
size_t RequireTensor(const X3TensorInfo& info,int device,bool writable){
    if(info.symbolic||!info.data||info.device_type!=TensorHelper::CudaDevice||info.device_id!=device||
       info.dtype!=X3_TENSOR_FLOAT32||!info.rank||info.rank>8||(writable&&info.readonly))
        throw X::Error("Bundle copy requires a contiguous FP32 tensor on its device");
    size_t bytes=4;
    for(int axis=int(info.rank)-1;axis>=0;--axis){
        if(info.shape[axis]<=0||info.shape[axis]>INT_MAX||bytes>INT64_MAX||
           (info.strides&&info.strides[axis]!=int64_t(bytes))||bytes>SIZE_MAX/size_t(info.shape[axis]))
            throw X::Error("Bundle tensor shape/stride overflow");
        bytes*=size_t(info.shape[axis]);
    }
    if(bytes>info.byte_size)throw X::Error("Bundle tensor logical bytes exceed backing storage");
    return bytes;
}
bool Overlap(const X3TensorInfo& a,size_t an,const X3TensorInfo& b,size_t bn){
    auto x=reinterpret_cast<uintptr_t>(a.data),y=reinterpret_cast<uintptr_t>(b.data);
    return x>=y?x-y<bn:y-x<an;
}
void CheckDevicePointer(const void* pointer,int device){
    cudaPointerAttributes info{};Check(cudaPointerGetAttributes(&info,pointer));
    if((info.type!=cudaMemoryTypeDevice&&info.type!=cudaMemoryTypeManaged)||info.device!=device)
        throw X::Error("Bundle captured copy requires same-device device/managed operands");
}
void CheckGraph(cudaGraph_t graph,int device,int depth=0){
    if(depth>8)throw X::Error("Bundle child graph nesting exceeds8");
    size_t count=0;Check(cudaGraphGetNodes(graph,nullptr,&count));
    std::vector<cudaGraphNode_t> nodes(count);Check(cudaGraphGetNodes(graph,nodes.data(),&count));
    for(auto node:nodes){
        cudaGraphNodeType type;Check(cudaGraphNodeGetType(node,&type));
        if(type==cudaGraphNodeTypeKernel||type==cudaGraphNodeTypeEmpty)continue;
        if(type==cudaGraphNodeTypeGraph){cudaGraph_t child=nullptr;Check(cudaGraphChildGraphNodeGetGraph(node,&child));CheckGraph(child,device,depth+1);continue;}
        if(type==cudaGraphNodeTypeMemcpy){
            cudaMemcpy3DParms p{};Check(cudaGraphMemcpyNodeGetParams(node,&p));
            if(p.srcArray||p.dstArray)throw X::Error("Bundle captured array copies are unsupported");
            CheckDevicePointer(p.srcPtr.ptr,device);CheckDevicePointer(p.dstPtr.ptr,device);continue;
        }
        if(type==cudaGraphNodeTypeMemset){cudaMemsetParams p{};Check(cudaGraphMemsetNodeGetParams(node,&p));CheckDevicePointer(p.dst,device);continue;}
        throw X::Error("Bundle graph contains an unsupported host/event/allocation/control node");
    }
}
}
CUDAExecutionBundle::~CUDAExecutionBundle(){Close();}
void CUDAExecutionBundle::RequireBuilding() const {
    if(closed_||sealed_||poisoned_)throw X::Error("Execution bundle is closed, sealed or poisoned");
}
void CUDAExecutionBundle::RequireAffinity() const {
    int current=-1;Check(cudaGetDevice(&current));
    if(device_<0||current!=device_||CurrentExecutionStream()!=stream_||std::this_thread::get_id()!=thread_)
        throw X::Error("Execution bundle must use its original device, stream and host thread");
}
void CUDAExecutionBundle::CloseUnlocked() noexcept {
    int previous=-1;cudaGetDevice(&previous);if(device_>=0)cudaSetDevice(device_);
    if(pending_){
        if(stream_==cudaStreamPerThread&&thread_!=std::this_thread::get_id())cudaDeviceSynchronize();
        else cudaStreamSynchronize(stream_);
    }
    if(executable_)cudaGraphExecDestroy(executable_);
    if(graph_)cudaGraphDestroy(graph_);
    executable_=nullptr;graph_=nullptr;last_=nullptr;pending_=false;sealed_=false;closed_=true;
    // Graph pointers retire before the context leases and every buffer owner.
    children_.clear();parents_.clear();parentPointers_.clear();tensors_.clear();
    if(previous>=0&&previous!=device_)cudaSetDevice(previous);
}
void CUDAExecutionBundle::Close(){std::lock_guard<std::mutex> lock(mutex_);CloseUnlocked();}
X::Value CUDAExecutionBundle::AddGraph(X::Value value,X::Value inputs){
    std::lock_guard<std::mutex> lock(mutex_);RequireBuilding();
    if(value.runtime()!=Host()->runtime||inputs.runtime()!=Host()->runtime)
        throw X::Error("Bundle graphs and inputs must belong to this runtime");
    auto* parent=static_cast<CUDACompiledGraph*>(Host()->instance_get_native_data(value.raw(),X::detail::native_type_name<CUDACompiledGraph>()));
    if(!parent||parentPointers_.count(parent))throw X::Error("Bundle requires a distinct native compiled graph");
    auto child=parent->ReserveForBundle(inputs);
    try {
        if(device_<0){device_=child->Device();stream_=child->Stream();thread_=std::this_thread::get_id();}
        RequireAffinity();
        if(child->Device()!=device_||child->Stream()!=stream_)throw X::Error("Bundle child graph affinity mismatch");
        CheckGraph(child->Graph(),device_);
        if(!graph_)Check(cudaGraphCreate(&graph_,0));
        parents_.push_back(value);children_.push_back(child);parentPointers_.insert(parent);
        tensors_.insert(tensors_.end(),child->Tensors().begin(),child->Tensors().end());
        cudaGraphNode_t node;Check(cudaGraphAddChildGraphNode(&node,graph_,last_?&last_:nullptr,last_?1:0,child->Graph()));last_=node;
        return child->Output();
    }catch(...){CloseUnlocked();throw;}
}
X::Value CUDAExecutionBundle::ReplayGraph(int64_t index){
    std::lock_guard<std::mutex> lock(mutex_);RequireBuilding();RequireAffinity();
    if(index<0||static_cast<uint64_t>(index)>=children_.size())
        throw X::Error("Bundle replay requires an existing reserved graph index");
    auto& child=children_[static_cast<size_t>(index)];
    try {
        // Reuse its existing context and fixed bindings. The preceding node
        // completes before this replay; the same retained owners and outer
        // tensor fence cover both uses, even after the source object closes.
        CheckGraph(child->Graph(),device_);
        cudaGraphNode_t node;Check(cudaGraphAddChildGraphNode(&node,graph_,last_?&last_:nullptr,last_?1:0,child->Graph()));
        last_=node;++replayNodes_;return child->Output();
    }catch(...){CloseUnlocked();throw;}
}
void CUDAExecutionBundle::SliceInto(X::Value source,int64_t start,X::Value destination){
    std::lock_guard<std::mutex> lock(mutex_);RequireBuilding();RequireAffinity();
    if(source.runtime()!=Host()->runtime||destination.runtime()!=Host()->runtime||!X::Tensor::IsTensor(source)||!X::Tensor::IsTensor(destination))
        throw X::Error("Bundle slice requires tensors in this runtime");
    X::Tensor src(source),dst(destination);auto a=src.Info(),b=dst.Info();
    size_t an=RequireTensor(a,device_,false),bn=RequireTensor(b,device_,true);
    if(a.rank!=2||a.shape[0]!=b.shape[0]||start<0)throw X::Error("Bundle slice requires packed rows and matching batch");
    size_t width=bn/size_t(b.shape[0]);
    if(uint64_t(start)>uint64_t(a.shape[1])||width/4>uint64_t(a.shape[1]-start)||Overlap(a,an,b,bn))
        throw X::Error("Bundle packed slice overlaps or exceeds source");
    try {
        tensors_.emplace_back(src,X3_TENSOR_READ);tensors_.emplace_back(dst,X3_TENSOR_WRITE);
        cudaMemcpy3DParms p{};
        p.srcPtr=make_cudaPitchedPtr(static_cast<float*>(a.data)+start,size_t(a.shape[1])*4,size_t(a.shape[1])*4,size_t(a.shape[0]));
        p.dstPtr=make_cudaPitchedPtr(b.data,width,width,size_t(b.shape[0]));p.extent=make_cudaExtent(width,size_t(b.shape[0]),1);p.kind=cudaMemcpyDeviceToDevice;
        cudaGraphNode_t node;Check(cudaGraphAddMemcpyNode(&node,graph_,last_?&last_:nullptr,last_?1:0,&p));last_=node;++copies_;
    }catch(...){CloseUnlocked();throw;}
}
void CUDAExecutionBundle::CopyLastChunk(X::Value source,X::Value active,X::Value destination){
    std::lock_guard<std::mutex> lock(mutex_);RequireBuilding();RequireAffinity();
    for(auto value:{source,active,destination})if(value.runtime()!=Host()->runtime||!X::Tensor::IsTensor(value))
        throw X::Error("Bundle last-chunk operands must be tensors in this runtime");
    X::Tensor src(source),flags(active),dst(destination);auto a=src.Info(),b=flags.Info(),c=dst.Info();
    size_t an=RequireTensor(a,device_,false),bn=RequireTensor(b,device_,false),cn=RequireTensor(c,device_,true);
    if(a.rank!=3||b.rank!=2||c.rank!=2||a.shape[0]!=b.shape[0]||a.shape[1]!=b.shape[1]||a.shape[0]!=c.shape[0]||a.shape[2]!=c.shape[1]||
       Overlap(a,an,c,cn)||Overlap(b,bn,c,cn)||a.shape[0]>8||a.shape[1]>128||a.shape[2]>16384)
        throw X::Error("Bundle last-chunk shapes/aliases exceed admitted profile");
    cudaGraph_t captured=nullptr;
    try {
        tensors_.emplace_back(src,X3_TENSOR_READ);tensors_.emplace_back(flags,X3_TENSOR_READ);tensors_.emplace_back(dst,X3_TENSOR_WRITE);
        Check(cudaStreamBeginCapture(stream_,cudaStreamCaptureModeThreadLocal));
        auto launched=CopyQwen35LastChunk(static_cast<const float*>(a.data),static_cast<const float*>(b.data),static_cast<float*>(c.data),int(a.shape[0]),int(a.shape[1]),int(a.shape[2]),stream_);
        auto ended=cudaStreamEndCapture(stream_,&captured);Check(launched);Check(ended);
        cudaGraphNode_t node;Check(cudaGraphAddChildGraphNode(&node,graph_,last_?&last_:nullptr,last_?1:0,captured));last_=node;
        Check(cudaGraphDestroy(captured));captured=nullptr;++lastChunks_;
    }catch(...){if(captured)cudaGraphDestroy(captured);CloseUnlocked();throw;}
}
void CUDAExecutionBundle::Seal(){
    std::lock_guard<std::mutex> lock(mutex_);RequireBuilding();RequireAffinity();
    if(!graph_||children_.empty())throw X::Error("Bundle needs a captured model graph");
    try{CheckGraph(graph_,device_);Check(cudaGraphInstantiate(&executable_,graph_,0));sealed_=true;}
    catch(...){CloseUnlocked();throw;}
}
void CUDAExecutionBundle::Run(){
    std::lock_guard<std::mutex> lock(mutex_);
    if(!sealed_||closed_||poisoned_)throw X::Error("Execution bundle is not runnable");
    RequireAffinity();
    try{auto use=TensorHelper::AcquireGPU(tensors_,stream_);pending_=true;Check(cudaGraphLaunch(executable_,stream_));use.Finish();++submissions_;}
    catch(...){poisoned_=true;throw;}
}
std::string CUDAExecutionBundle::StatsJson() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return nlohmann::json{{"sealed",sealed_},{"closed",closed_},{"poisoned",poisoned_},{"device",device_},
        {"reserved_model_contexts",children_.size()},{"child_replay_nodes",replayNodes_},{"packed_slices",copies_},{"last_chunk_copies",lastChunks_},{"submissions",submissions_},
        {"new_tensor_allocations",0},{"scope","Opt-in linear model/copy bundle; retained tensors and existing context workspaces excluded from new allocations. CUDA graph driver allocation is untracked here. No GPU conditional control or speed/memory acceptance."}}.dump();
}
}
#endif
