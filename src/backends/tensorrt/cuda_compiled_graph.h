// SPDX-License-Identifier: Apache-2.0
#pragma once
#if !defined(GARNET_CPU_ONLY) && !defined(GARNET_METAL)
#include "xlang3/xlang3.h"
#include "safetensors_index.h"
#include "trt_graph_reservation.h"
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
namespace Garnet {
// CUDA implementation of the backend-neutral compiled_graph boundary used by
// Qwen3.5 tensor programs. One graph owns one immutable refitted engine.
class CUDACompiledGraph {
    friend class CUDAExecutionBundle;
    friend class CUDAMTPDraft;
    SafeTensorsIndex weights_;
    std::string cacheRoot_,enginePath_;
    std::shared_ptr<void> execution_;
    X::Value deviceOutput_;
    bool ready_=false,useCudaGraph_=false,serialGraphReuse_=false,externalOutput_=false;
    bool userManagedWorkspace_=false,workspaceBound_=false,executionStarted_=false;
    double compileMs_=0,lastMs_=0;
    unsigned long long builderWorkspaceBytes_=64ULL<<20;
    size_t externalWeightCount_=0,externalWeightBytes_=0,joinedProjectionInputs_=0;
    size_t internalPackedStores_=0,internalPackedRoots_=0;
    mutable std::mutex mutex_;
    std::shared_ptr<std::atomic<size_t>> reservations_=std::make_shared<std::atomic<size_t>>(0);
    void CloseUnlocked();
    void RequireUnreserved() const;
    std::shared_ptr<TRTGraphReservation> ReserveForBundle(X::Value inputs);
public:
    CUDACompiledGraph();
    ~CUDACompiledGraph();
    bool Build(X::Value graph,X::Value symbols,const std::string& checkpoint);
    bool LoadRefittedSingleInputEngine(const std::string& checkpoint,const std::string& artifact);
    std::string EngineInfoJson() const;
    X::Value Run(X::Value inputs);
    X::Value RunDevice(X::Value inputs);
    void Close();
    void SetCudaGraph(bool enabled);
    void SetSerialGraphReuse(bool enabled);
    void SetBuilderWorkspaceBytes(long long bytes);
    void SetOutputStorage(X::Value output);
    void SetUserManagedWorkspace(bool enabled);
    void SetWorkspaceStorage(X::Value storage);
    std::string StatsJson() const;
    bool Ready() const;
    double CompileMs() const;
    double LastInferenceMs() const;
    BEGIN_PACKAGE(CUDACompiledGraph)
        APISET().AddFunc<3>("build",&CUDACompiledGraph::Build);
        APISET().AddFunc<2>("load_refitted_single_input_engine",&CUDACompiledGraph::LoadRefittedSingleInputEngine);
        APISET().AddFunc<0>("engine_info_json",&CUDACompiledGraph::EngineInfoJson);
        APISET().AddFunc<1>("run",&CUDACompiledGraph::Run);
        APISET().AddFunc<1>("run_device",&CUDACompiledGraph::RunDevice);
        APISET().AddFunc<0>("close",&CUDACompiledGraph::Close);
        APISET().AddFunc<1>("enable_cuda_graph",&CUDACompiledGraph::SetCudaGraph);
        APISET().AddFunc<1>("enable_serial_graph_reuse",&CUDACompiledGraph::SetSerialGraphReuse);
        APISET().AddFunc<1>("set_builder_workspace_bytes",&CUDACompiledGraph::SetBuilderWorkspaceBytes);
        APISET().AddFunc<1>("set_output_storage",&CUDACompiledGraph::SetOutputStorage);
        APISET().AddFunc<1>("enable_user_managed_workspace",&CUDACompiledGraph::SetUserManagedWorkspace);
        APISET().AddFunc<1>("set_workspace_storage",&CUDACompiledGraph::SetWorkspaceStorage);
        APISET().AddFunc<0>("stats_json",&CUDACompiledGraph::StatsJson);
        APISET().AddProp("ready",&CUDACompiledGraph::Ready);
        APISET().AddProp("compile_ms",&CUDACompiledGraph::CompileMs);
        APISET().AddProp("last_inference_ms",&CUDACompiledGraph::LastInferenceMs);
    END_PACKAGE
};
}
#endif
