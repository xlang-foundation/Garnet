// SPDX-License-Identifier: Apache-2.0
#pragma once
#if !defined(GARNET_CPU_ONLY) && !defined(GARNET_METAL)
#include "xlang3/xlang3.h"
#include "trt_graph_reservation.h"
#include <cuda_runtime_api.h>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>
namespace Garnet {
class CUDACompiledGraph;
// Opt-in linear device execution. Contexts are reserved once; tensor fences
// bracket the entire submission. No host/event/allocation nodes in the graph.
class CUDAExecutionBundle {
    cudaGraph_t graph_=nullptr;
    cudaGraphExec_t executable_=nullptr;
    cudaGraphNode_t last_=nullptr;
    cudaStream_t stream_=nullptr;
    int device_=-1;
    std::thread::id thread_;
    bool sealed_=false,closed_=false,pending_=false,poisoned_=false;
    uint64_t submissions_=0,copies_=0,lastChunks_=0,replayNodes_=0;
    std::vector<std::shared_ptr<TRTGraphReservation>> children_;
    std::vector<X::Value> parents_;
    std::unordered_set<CUDACompiledGraph*> parentPointers_;
    std::vector<std::pair<X::Tensor,X3TensorAccess>> tensors_;
    mutable std::mutex mutex_;
    void CloseUnlocked() noexcept;
    void RequireBuilding() const;
    void RequireAffinity() const;
public:
    ~CUDAExecutionBundle();
    X::Value AddGraph(X::Value graph,X::Value inputs);
    X::Value ReplayGraph(int64_t index);
    void SliceInto(X::Value source,int64_t start,X::Value destination);
    void CopyLastChunk(X::Value source,X::Value active,X::Value destination);
    void Seal();
    void Run();
    void Close();
    std::string StatsJson() const;
    BEGIN_PACKAGE(CUDAExecutionBundle)
        APISET().AddFunc<2>("add_graph",&CUDAExecutionBundle::AddGraph);
        APISET().AddFunc<1>("replay_graph",&CUDAExecutionBundle::ReplayGraph);
        APISET().AddFunc<3>("slice_into",&CUDAExecutionBundle::SliceInto);
        APISET().AddFunc<3>("copy_last_chunk",&CUDAExecutionBundle::CopyLastChunk);
        APISET().AddFunc<0>("seal",&CUDAExecutionBundle::Seal);
        APISET().AddFunc<0>("run",&CUDAExecutionBundle::Run);
        APISET().AddFunc<0>("close",&CUDAExecutionBundle::Close);
        APISET().AddFunc<0>("stats_json",&CUDAExecutionBundle::StatsJson);
    END_PACKAGE
};
}
#endif
