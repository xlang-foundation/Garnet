// SPDX-License-Identifier: Apache-2.0
#if !defined(GARNET_CPU_ONLY) && !defined(GARNET_METAL)
#include "cuda_compiled_graph.h"
#include "trt_builder.h"
#include "qwen35_delta_plugin.h"
#include "qwen35_tree_plugin.h"
#include "tensor_helper.h"
#include <atomic>
#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <nlohmann/json.hpp>
namespace Garnet {
CUDACompiledGraph::CUDACompiledGraph() {
    if(!EnsureQwen35TreePluginsRegistered())throw X::Error("Native tree plugin registration failed");
    static std::atomic<uint64_t> next{0};
    const auto stamp=std::chrono::steady_clock::now().time_since_epoch().count();
    cacheRoot_=(std::filesystem::temp_directory_path()/
        ("garnet-cuda-graph-"+std::to_string(stamp)+"-"+std::to_string(++next))).string()+"/";
    enginePath_=cacheRoot_+"model.engine";
}
CUDACompiledGraph::~CUDACompiledGraph(){Close();}
void CUDACompiledGraph::RequireUnreserved() const {
    if(reservations_->load(std::memory_order_acquire))throw X::Error("CUDA graph context is reserved by an execution bundle");
}
std::shared_ptr<TRTGraphReservation> CUDACompiledGraph::ReserveForBundle(X::Value inputs) {
    std::lock_guard<std::mutex> lock(mutex_);RequireUnreserved();
    if(!ready_||!useCudaGraph_||!deviceOutput_.IsValid()||(userManagedWorkspace_&&!workspaceBound_))
        throw X::Error("Warm and capture a ready device graph before reserving it");
    if(inputs.runtime()!=Host()->runtime)throw X::Error("Reserved CUDA inputs must belong to this runtime");
    std::string error;
    auto reservation=TRTBuilder::ReserveCapturedExecution(execution_,inputs,deviceOutput_,reservations_,error);
    if(!reservation)throw X::Error(error);
    return reservation;
}
void CUDACompiledGraph::CloseUnlocked(){deviceOutput_=X::Value();externalOutput_=false;execution_.reset();TRTBuilder::ReleaseCachedExecutions(cacheRoot_);ready_=false;workspaceBound_=false;executionStarted_=false;externalWeightCount_=externalWeightBytes_=joinedProjectionInputs_=internalPackedStores_=internalPackedRoots_=0;}
void CUDACompiledGraph::Close(){std::lock_guard<std::mutex> lock(mutex_);CloseUnlocked();}
bool CUDACompiledGraph::Ready() const{std::lock_guard<std::mutex> lock(mutex_);return ready_;}
double CUDACompiledGraph::CompileMs() const{std::lock_guard<std::mutex> lock(mutex_);return compileMs_;}
double CUDACompiledGraph::LastInferenceMs() const{std::lock_guard<std::mutex> lock(mutex_);return lastMs_;}
void CUDACompiledGraph::SetCudaGraph(bool enabled){std::lock_guard<std::mutex> lock(mutex_);RequireUnreserved();useCudaGraph_=enabled;}
void CUDACompiledGraph::SetSerialGraphReuse(bool enabled){std::lock_guard<std::mutex> lock(mutex_);RequireUnreserved();serialGraphReuse_=enabled;}
void CUDACompiledGraph::SetUserManagedWorkspace(bool enabled){
    std::lock_guard<std::mutex> lock(mutex_);
    RequireUnreserved();
    if(ready_)throw X::Error("Select workspace allocation policy before building a CUDA graph");
    userManagedWorkspace_=enabled;
}
void CUDACompiledGraph::SetWorkspaceStorage(X::Value storage){
    std::lock_guard<std::mutex> lock(mutex_);
    RequireUnreserved();
    if(!ready_||!userManagedWorkspace_||executionStarted_)throw X::Error("Bind managed workspace once after build and before execution");
    if(storage.runtime()!=Host()->runtime)throw X::Error("CUDA workspace must belong to this runtime");
    std::string error;if(!TRTBuilder::BindCapturedWorkspace(execution_,storage,error))throw X::Error(error);
    workspaceBound_=true;
}
void CUDACompiledGraph::SetBuilderWorkspaceBytes(long long bytes){
    std::lock_guard<std::mutex> lock(mutex_);
    RequireUnreserved();
    if(ready_)throw X::Error("Set builder workspace before building a CUDA graph");
    if(bytes<(1LL<<20)||bytes>(1LL<<30))throw X::Error("CUDA graph builder workspace requires 1MiB..1GiB");
    builderWorkspaceBytes_=static_cast<unsigned long long>(bytes);
}
void CUDACompiledGraph::SetOutputStorage(X::Value output){
    std::lock_guard<std::mutex> lock(mutex_);
    RequireUnreserved();
    if(!ready_||deviceOutput_.IsValid())throw X::Error("Bind external output once after build and before first execution");
    if(output.runtime()!=Host()->runtime)throw X::Error("External CUDA output must belong to this runtime");
    std::string error;if(!TRTBuilder::CapturedOutputCompatible(execution_,output,error))throw X::Error(error);
    deviceOutput_=output;externalOutput_=true;
}
std::string CUDACompiledGraph::StatsJson() const{
    std::lock_guard<std::mutex> lock(mutex_);auto info=nlohmann::json::parse(TRTBuilder::RetainedExecutionStats(execution_));
    info["cuda_graph_requested"]=useCudaGraph_;
    info["execution_bundle_reservations"]=reservations_->load(std::memory_order_acquire);
    info["user_managed_workspace_requested"]=userManagedWorkspace_;
    info["qwen35_whole_program_cuda"]=Qwen35WholeProgramCUDA();
    info["serial_graph_reuse_requested"]=serialGraphReuse_;
    info["builder_workspace_limit_bytes"]=builderWorkspaceBytes_;
    info["external_bf16_weight_inputs"]=externalWeightCount_;
    info["joined_projection_inputs"]=joinedProjectionInputs_;
    info["internal_packed_stores"]=internalPackedStores_;
    info["internal_packed_root_fills"]=internalPackedRoots_;
    info["external_bf16_weight_bytes"]=externalWeightBytes_;
    info["external_weight_scope"]="Caller-owned symbolic BF16 inputs; excluded from graph-owned allocations";
    uint64_t span=0,footprint=0;
    if(deviceOutput_.IsValid()){
        auto value=X::Tensor(deviceOutput_).Info();span=value.byte_size;footprint=TensorHelper::ItemSize(value.dtype);
        for(uint32_t i=0;i<value.rank;++i)footprint*=value.shape[i];
    }
    info["device_output_bytes"]=footprint;
    info["owned_device_output_bytes"]=externalOutput_?0:span;
    info["external_output_storage"]=externalOutput_;
    info["external_output_storage_span_bytes"]=externalOutput_?span:0;
    info["output_storage_scope"]="Logical output footprint; externally bound storage is excluded from owned bytes and must be counted once by its caller";
    return info.dump();
}
bool CUDACompiledGraph::LoadRefittedSingleInputEngine(const std::string& checkpoint,const std::string& artifact) {
    std::lock_guard<std::mutex> lock(mutex_);RequireUnreserved();CloseUnlocked();
    const auto start=std::chrono::steady_clock::now();std::string error;
    try{
        if(!weights_.Open(checkpoint,error))throw std::runtime_error(error);
        const std::filesystem::path source(artifact);
        if(!std::filesystem::is_regular_file(source)||std::filesystem::file_size(source)==0)
            throw std::invalid_argument("Engine artifact requires a nonempty regular file");
        // Each graph owns its copied artifact and runtime context. Callers retain
        // the original immutable plan; refit uses the existing checkpoint path.
        std::filesystem::create_directories(cacheRoot_);
        std::filesystem::copy_file(source,enginePath_,std::filesystem::copy_options::overwrite_existing);
        execution_=TRTBuilder::RetainCachedExecution(enginePath_,&weights_,error,userManagedWorkspace_);
        if(!execution_)throw std::runtime_error(error);
        const auto info=nlohmann::json::parse(TRTBuilder::RetainedEngineInfo(execution_));
        // The restricted contract excludes caller-owned symbolic weight inputs
        // and multi-input packed-state programs, preserving their normal builder.
        if(!info.at("single_static_float32_input_and_output").get<bool>())
            throw std::invalid_argument("Artifact loader requires one static FLOAT32 input and one nonaliased FLOAT32 output");
        compileMs_=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        ready_=true;return true;
    }catch(...){CloseUnlocked();throw;}
}
std::string CUDACompiledGraph::EngineInfoJson() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if(!ready_||!execution_)throw X::Error("Engine inspection requires a ready graph");
    return TRTBuilder::RetainedEngineInfo(execution_);
}
bool CUDACompiledGraph::Build(X::Value graph,X::Value symbols,const std::string& checkpoint) {
    std::lock_guard<std::mutex> lock(mutex_);RequireUnreserved();CloseUnlocked();
    if(graph.runtime()!=Host()->runtime||symbols.runtime()!=Host()->runtime||!symbols.IsList())
        throw X::Error("CUDA graph and symbols must belong to this runtime");
    std::string error;if(!weights_.Open(checkpoint,error))throw X::Error(error);
    std::filesystem::create_directories(cacheRoot_);
    X::ARGS inputs;for(uint64_t i=0;i<symbols.Size();++i)inputs.push_back(symbols.Get(i));
    auto arguments=inputs;TRTBuilder builder(Host());builder.SetCapturedStrictFP32(true);
    builder.SetCapturedWorkspaceBytes(builderWorkspaceBytes_);
    auto start=std::chrono::steady_clock::now();
    if(!builder.BuildCapturedGraph(graph,X::Value(),arguments,inputs,&weights_,enginePath_,{},error))
        throw X::Error(error);
    execution_=TRTBuilder::RetainCachedExecution(enginePath_,&weights_,error,userManagedWorkspace_);
    if(!execution_)throw X::Error(error);
    externalWeightCount_=builder.ExternalBF16WeightInputs();externalWeightBytes_=builder.ExternalBF16WeightBytes();
    joinedProjectionInputs_=builder.JoinedProjectionInputs();
    internalPackedStores_=builder.InternalPackedStoreCount();internalPackedRoots_=builder.InternalPackedRootCount();
    if(internalPackedStores_&&(internalPackedRoots_!=1||nlohmann::json::parse(TRTBuilder::RetainedExecutionStats(execution_)).at("network_output_aliases_input").get<bool>())){
        CloseUnlocked();throw X::Error("Internal packed stores require one internal root and nonaliased public output");
    }
    compileMs_=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    ready_=true;return true;
}
X::Value CUDACompiledGraph::Run(X::Value inputs) {
    std::lock_guard<std::mutex> lock(mutex_);RequireUnreserved();if(!ready_)throw X::Error("CUDA graph has not been built");
    if(userManagedWorkspace_)throw X::Error("Managed CUDA workspace requires run_device");
    if(externalOutput_)throw X::Error("Externally bound output requires run_device");
    TRTBuilder builder(Host());std::string error;auto start=std::chrono::steady_clock::now();
    auto result=builder.RunCapturedEngine(enginePath_,inputs,&weights_,X::Value(),false,error);
    if(!result.IsValid())throw X::Error(error);
    // Explicit host boundary for existing request-state orchestration. A GPU
    // resident session is a separate optimization, not hidden CPU execution.
    X::Tensor tensor(result);X::Value host=TensorHelper::CopyToCPU(tensor);
    lastMs_=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    return host;
}
X::Value CUDACompiledGraph::RunDevice(X::Value inputs) {
    std::lock_guard<std::mutex> lock(mutex_);RequireUnreserved();if(!ready_)throw X::Error("CUDA graph has not been built");
    if(userManagedWorkspace_&&!workspaceBound_)throw X::Error("Bind managed workspace before CUDA execution");
    if(externalOutput_){
        if(!inputs.IsList())throw X::Error("External CUDA output requires tensor input list");
        auto output=X::Tensor(deviceOutput_).Info();auto out=reinterpret_cast<uintptr_t>(output.data);
        for(uint64_t i=0;i<inputs.Size();++i)if(X::Tensor::IsTensor(inputs.Get(i))){
            auto input=X::Tensor(inputs.Get(i)).Info();if(input.device_type!=TensorHelper::CudaDevice||!input.data)continue;
            auto in=reinterpret_cast<uintptr_t>(input.data);
            // Remaining storage spans intentionally reject even ambiguous views
            // of the same backing allocation. Inputs and output stay separate.
            if(out>=in?out-in<input.byte_size:in-out<output.byte_size)
                throw X::Error("External CUDA output overlaps input storage");
        }
    }
    TRTBuilder builder(Host());std::string error;auto start=std::chrono::steady_clock::now();
    executionStarted_=true;
    auto result=builder.RunCapturedEngine(enginePath_,inputs,&weights_,deviceOutput_,useCudaGraph_,error,{},serialGraphReuse_);
    if(!result.IsValid())throw X::Error(error);
    if(externalOutput_&&X::Tensor(result).Info().data!=X::Tensor(deviceOutput_).Info().data)
        throw X::Error("Configured CUDA output storage was not reused");
    deviceOutput_=result;
    lastMs_=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    // Reused device storage: consume/copy its outputs before the next call.
    return result;
}
}
#endif
