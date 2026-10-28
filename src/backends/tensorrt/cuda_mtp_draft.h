// SPDX-License-Identifier: Apache-2.0
#pragma once
#if !defined(GARNET_CPU_ONLY) && !defined(GARNET_METAL)
#include "xlang3/xlang3.h"
#include "trt_graph_reservation.h"
#include "../../cuda/qwen35_mtp_draft_control.cuh"
#include <cuda_runtime_api.h>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
namespace Garnet {
// Opt-in D3/T4 native host dispatch. Target acceptance/verification stay
// authoritative. One original context lease/table; no conditional CUDA graph.
class CUDAMTPDraft {
    int device_=-1,cacheWidth_=0,packedWidth_=0;
    cudaStream_t stream_=nullptr;
    std::thread::id thread_;
    bool ready_=false,closed_=false,poisoned_=false,pending_=false;
    std::shared_ptr<TRTGraphReservation> child_;
    X::Value graphParent_,ioParent_;
    MTPDraftDeviceConfig config_{};
    MTPDraftDeviceState lastState_{};
    std::vector<X::Tensor> inputs_,owned_;
    std::vector<std::pair<X::Tensor,X3TensorAccess>> tensors_;
    X::Tensor table_,candidates_,hidden_,last_,packed_;
    X::Tensor controls_,first_,next_,proposals_,state_,scores_,values_;
    uint64_t ownedBytes_=0,submissions_=0,childSubmissions_=0,decisionTransfers_=0;
    mutable std::mutex mutex_;
    void RequireAffinity() const;
    void RequireReady() const;
    void CloseUnlocked() noexcept;
    void ValidateInputs(X::Value) const;
public:
    ~CUDAMTPDraft();
    bool Build(X::Value graph,X::Value io,X::Value inputs,X::Value views,X::Value candidates,const std::string& config);
    X::Value RunChild(X::Value inputs);
    std::string Run(X::Value firstDraft,X::Value controls);
    void Close();
    std::string StatsJson() const;
    BEGIN_PACKAGE(CUDAMTPDraft)
        APISET().AddFunc<6>("build",&CUDAMTPDraft::Build);
        APISET().AddFunc<1>("run_child",&CUDAMTPDraft::RunChild);
        APISET().AddFunc<2>("run",&CUDAMTPDraft::Run);
        APISET().AddFunc<0>("close",&CUDAMTPDraft::Close);
        APISET().AddFunc<0>("stats_json",&CUDAMTPDraft::StatsJson);
    END_PACKAGE
};
}
#endif
